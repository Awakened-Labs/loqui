//! Decoding uploaded audio for transcription.
//!
//! The container is sniffed from the bytes, never taken from a filename or
//! content type a client supplied. Channels are averaged to mono. Length is
//! capped while decoding, so a small compressed upload cannot expand into
//! hours of samples in memory.

use std::io::Cursor;

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

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
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| Error::Decode(format!("unsupported codec: {e}")))?;

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
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A corrupt packet is skipped, as every player skips it.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(Error::Decode(e.to_string())),
        };
        let channels = decoded.num_planes();
        if channels == 0 {
            continue;
        }
        let packet_rate = decoded.spec().rate();
        match rate {
            None => rate = Some(packet_rate),
            Some(r) if r != packet_rate => return Err(Error::Decode("sample rate changes mid-stream".into())),
            Some(_) => {}
        }
        decoded.copy_to_vec_interleaved::<f32>(&mut interleaved);
        let scale = 1.0 / channels as f32;
        mono.extend(interleaved.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() * scale));
        if let Some(max) = max_secs
            && mono.len() as u64 > max * u64::from(packet_rate)
        {
            return Err(Error::TooLong { max_secs: max });
        }
    }
    match rate {
        Some(rate) if !mono.is_empty() => Ok(Pcm { samples: mono, rate }),
        _ => Err(Error::Decode("no audio frames".into())),
    }
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
        assert!(decode(&wav, Some(3)).is_ok());
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode(b"definitely not audio", None).is_err());
        assert!(decode(b"", None).is_err());
    }
}
