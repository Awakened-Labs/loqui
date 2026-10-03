//! Decoding uploaded audio for transcription.
//!
//! The container is sniffed from the bytes, never taken from a filename or
//! content type a client supplied. Channels are averaged to mono. Length is
//! capped while decoding, so a small compressed upload cannot expand into
//! hours of samples in memory.

use std::io::Cursor;

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::opus::{self, Opus};
use crate::{Error, Pcm};

/// The sample rate Whisper consumes.
pub const WHISPER_RATE: u32 = 16_000;

/// Decodes any supported container to mono at its native rate. With
/// `max_secs`, stops with [`Error::TooLong`] as soon as the audio exceeds it.
pub fn decode(bytes: &[u8], max_secs: Option<u64>) -> Result<Pcm, Error> {
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(&Hint::new(), source, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| Error::Decode(format!("unrecognised container: {e}")))?;
    let (track_id, params) = {
        let track = format.default_track(TrackType::Audio).ok_or_else(|| Error::Decode("no audio track".into()))?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| Error::Decode("audio track has no codec parameters".into()))?
            .clone();
        (track.id, params)
    };
    let mut decoder = if params.codec == CODEC_ID_OPUS {
        let channels = params.channels.as_ref().map_or(1, |c| c.count());
        Decoder::Opus(Box::new(Opus::new(channels, params.extra_data.as_deref())?))
    } else {
        Decoder::Symphonia(
            symphonia::default::get_codecs()
                .make_audio_decoder(&params, &AudioDecoderOptions::default())
                .map_err(|e| Error::Decode(format!("unsupported codec: {e}")))?,
        )
    };

    let mut mono: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();
    let mut rate: Option<u32> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(Error::Decode(e.to_string())),
        };
        if packet.track_id != track_id {
            continue;
        }
        // A corrupt packet is skipped, as every player skips it.
        let (samples, channels, packet_rate): (&[f32], usize, u32) = match &mut decoder {
            Decoder::Symphonia(decoder) => {
                let decoded = match decoder.decode(&packet) {
                    Ok(decoded) => decoded,
                    Err(SymphoniaError::DecodeError(_)) => continue,
                    Err(SymphoniaError::ResetRequired) => {
                        decoder.reset();
                        continue;
                    }
                    Err(e) => return Err(Error::Decode(e.to_string())),
                };
                decoded.copy_to_vec_interleaved::<f32>(&mut interleaved);
                (&interleaved, decoded.num_planes(), decoded.spec().rate())
            }
            Decoder::Opus(opus) => {
                let channels = opus.channels();
                // Priming comes from the OpusHead, so the demuxer's
                // trim_start (the same priming, where Ogg sets it) is not
                // applied twice. trim_end is in 48 kHz samples where set.
                match opus.decode(&packet.data, packet.trim_end.get() as usize) {
                    Some(samples) => (samples, channels, opus::RATE),
                    None => continue,
                }
            }
        };
        if channels == 0 {
            continue;
        }
        match rate {
            None => rate = Some(packet_rate),
            Some(r) if r != packet_rate => return Err(Error::Decode("sample rate changes mid-stream".into())),
            Some(_) => {}
        }
        let scale = 1.0 / channels as f32;
        mono.extend(samples.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() * scale));
        if let Some(max) = max_secs
            && mono.len() as u64 > max.saturating_mul(u64::from(packet_rate))
        {
            return Err(Error::TooLong { max_secs: max });
        }
    }
    match rate {
        Some(rate) if !mono.is_empty() => Ok(Pcm { samples: mono, rate }),
        _ => Err(Error::Decode("no audio frames".into())),
    }
}

/// What turns packets into samples: symphonia's own codecs, or Opus, which
/// symphonia demuxes but cannot decode.
enum Decoder {
    Symphonia(Box<dyn AudioDecoder>),
    Opus(Box<Opus>),
}

/// Resamples to `target` Hz with a band-limited FFT resampler.
pub fn resample(pcm: Pcm, target: u32) -> Result<Pcm, Error> {
    if pcm.rate == target || pcm.samples.is_empty() {
        return Ok(Pcm { rate: target, ..pcm });
    }
    let mut resampler = Fft::<f32>::new(pcm.rate as usize, target as usize, 1024, 1, FixedSync::Input)
        .map_err(|e| Error::Decode(format!("cannot resample {} Hz to {target} Hz: {e}", pcm.rate)))?;
    let len = pcm.samples.len();
    let capacity = resampler.process_all_needed_output_len(len);
    let mut out = vec![0.0f32; capacity];
    let input = InterleavedSlice::new(&pcm.samples[..], 1, len).map_err(|e| Error::Decode(e.to_string()))?;
    let mut output = InterleavedSlice::new_mut(&mut out[..], 1, capacity).map_err(|e| Error::Decode(e.to_string()))?;
    let (_, written) = resampler.process_all_into_buffer(&input, &mut output, len, None).map_err(|e| Error::Decode(e.to_string()))?;
    out.truncate(written);
    Ok(Pcm { samples: out, rate: target })
}

/// What Whisper wants: mono, 16 kHz, clamped to `[-1, 1]`.
pub fn decode_mono_16k(bytes: &[u8], max_secs: Option<u64>) -> Result<Pcm, Error> {
    let mut pcm = resample(decode(bytes, max_secs)?, WHISPER_RATE)?;
    for s in &mut pcm.samples {
        *s = s.clamp(-1.0, 1.0);
    }
    Ok(pcm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Format, encode};

    #[test]
    fn resamples_to_whisper_rate_preserving_duration() {
        let pcm = Pcm { samples: vec![0.25; 24_000], rate: 24_000 };
        let wav = encode(&pcm, Format::Wav).unwrap();
        let out = decode_mono_16k(&wav, None).unwrap();
        assert_eq!(out.rate, WHISPER_RATE);
        assert!((out.duration_secs() - 1.0).abs() < 0.01, "{}", out.duration_secs());
    }

    #[test]
    fn enforces_the_length_cap_while_decoding() {
        let pcm = Pcm { samples: vec![0.1; 16_000 * 3], rate: 16_000 };
        let wav = encode(&pcm, Format::Wav).unwrap();
        assert!(matches!(decode(&wav, Some(2)), Err(Error::TooLong { max_secs: 2 })));
        assert!(decode(&wav, Some(u64::MAX)).is_ok(), "a ceiling too large to scale must not overflow");
        assert!(decode(&wav, Some(3)).is_ok());
    }

    /// Frequency by zero crossings, and RMS, of a mono signal.
    fn tone(pcm: &Pcm) -> (f32, f32) {
        let crossings = pcm.samples.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        let freq = crossings as f32 / 2.0 / pcm.duration_secs() as f32;
        let rms = (pcm.samples.iter().map(|s| s * s).sum::<f32>() / pcm.samples.len() as f32).sqrt();
        (freq, rms)
    }

    // Fixtures: half a second of a 440 Hz sine at amplitude 1/8 (RMS 0.088),
    // encoded by ffmpeg's libopus.
    fn assert_is_the_sine(pcm: &Pcm) {
        assert_eq!(pcm.rate, 48_000);
        assert!((0.49..0.52).contains(&pcm.duration_secs()), "duration {}", pcm.duration_secs());
        let (freq, rms) = tone(pcm);
        assert!((420.0..460.0).contains(&freq), "frequency {freq}");
        assert!((0.06..0.12).contains(&rms), "rms {rms}");
    }

    #[test]
    fn decodes_ogg_opus_as_voice_notes_arrive() {
        assert_is_the_sine(&decode(include_bytes!("../tests/data/sine440-mono.ogg"), None).unwrap());
    }

    #[test]
    fn decodes_webm_opus_as_browsers_record_it_downmixing_stereo() {
        assert_is_the_sine(&decode(include_bytes!("../tests/data/sine440-stereo.webm"), None).unwrap());
    }

    #[test]
    fn decodes_every_frame_of_multi_frame_silk_packets() {
        // 40 ms SILK packets: opus-rs before 0.1.33 decoded their second
        // 20 ms frame as silence (restsend/opus-rs#27).
        let pcm = decode(include_bytes!("../tests/data/sine440-silk40ms.ogg"), None).unwrap();
        assert_is_the_sine(&pcm);
        let window = 960; // 20 ms at 48 kHz
        for (i, frame) in pcm.samples.chunks_exact(window).enumerate().skip(3) {
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / window as f32).sqrt();
            assert!(rms > 0.03, "20 ms window {i} is silent (rms {rms})");
        }
    }

    #[test]
    fn opus_drops_encoder_priming_and_padding_exactly_as_libopus_does() {
        // libopus (through ffmpeg) decodes every fixture to 24,000 samples.
        for bytes in [&include_bytes!("../tests/data/sine440-mono.ogg")[..], &include_bytes!("../tests/data/sine440-silk40ms.ogg")[..]] {
            assert_eq!(decode(bytes, None).unwrap().samples.len(), 24_000);
        }
        // WebM's end padding (DiscardPadding, 648 samples here) is parsed by
        // symphonia 0.6 but not passed on, so up to 13.5 ms of trailing
        // near-silence remains. The priming is still dropped exactly.
        let webm = decode(include_bytes!("../tests/data/sine440-stereo.webm"), None).unwrap();
        assert!((24_000..=24_648).contains(&webm.samples.len()), "{}", webm.samples.len());
    }

    #[test]
    fn enforces_the_length_cap_on_opus_too() {
        let ogg = include_bytes!("../tests/data/sine440-mono.ogg");
        assert!(matches!(decode(ogg, Some(0)), Err(Error::TooLong { max_secs: 0 })));
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode(b"definitely not audio", None).is_err());
        assert!(decode(b"", None).is_err());
    }
}
