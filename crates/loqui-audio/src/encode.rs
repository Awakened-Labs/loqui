//! Encoders for synthesized speech.

use flacenc::component::BitRepr;
use flacenc::error::Verify;

use crate::{Error, Format, Pcm};

pub fn encode(pcm: &Pcm, format: Format) -> Result<Vec<u8>, Error> {
    match format {
        Format::Wav => Ok(wav(pcm)),
        Format::Flac => flac(pcm),
        Format::Opus => crate::opus::encode_ogg(pcm),
        Format::Pcm => Ok(to_i16(&pcm.samples).flat_map(i16::to_le_bytes).collect()),
    }
}

fn to_i16(samples: &[f32]) -> impl Iterator<Item = i16> + '_ {
    samples.iter().map(|s| (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16)
}

/// A canonical 44-byte-header WAV: mono, 16-bit PCM.
fn wav(pcm: &Pcm) -> Vec<u8> {
    let data_len = u32::try_from(pcm.samples.len() * 2).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(44 + pcm.samples.len() * 2);
    out.extend(b"RIFF");
    out.extend(data_len.saturating_add(36).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes()); // PCM
    out.extend(1u16.to_le_bytes()); // mono
    out.extend(pcm.rate.to_le_bytes());
    out.extend((pcm.rate * 2).to_le_bytes()); // byte rate
    out.extend(2u16.to_le_bytes()); // block align
    out.extend(16u16.to_le_bytes()); // bits per sample
    out.extend(b"data");
    out.extend(data_len.to_le_bytes());
    out.extend(to_i16(&pcm.samples).flat_map(i16::to_le_bytes));
    out
}

fn flac(pcm: &Pcm) -> Result<Vec<u8>, Error> {
    let samples: Vec<i32> = to_i16(&pcm.samples).map(i32::from).collect();
    let config = flacenc::config::Encoder::default().into_verified().map_err(|(_, e)| Error::Encode(format!("FLAC config: {e:?}")))?;
    let source = flacenc::source::MemSource::from_samples(&samples, 1, 16, pcm.rate as usize);
    let mut stream =
        flacenc::encode_with_fixed_block_size(&config, source, config.block_size).map_err(|e| Error::Encode(format!("FLAC: {e:?}")))?;
    // flacenc 0.5 lets the short final block lower STREAMINFO's minimum
    // block size. The spec excludes the last block from that minimum, and
    // min != max marks the stream variable-blocksize, contradicting every
    // frame header: libFLAC warns, symphonia refuses the file. Restore it.
    stream.stream_info_mut().set_block_sizes(config.block_size, config.block_size).map_err(|e| Error::Encode(format!("FLAC: {e:?}")))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream.write(&mut sink).map_err(|e| Error::Encode(format!("FLAC: {e:?}")))?;
    Ok(sink.as_slice().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode;

    fn tone(rate: u32) -> Pcm {
        let samples = (0..rate).map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5).collect();
        Pcm { samples, rate }
    }

    /// Each encoding decodes, through symphonia, back to the same audio:
    /// an independent decoder, not our own inverse.
    #[test]
    fn wav_and_flac_round_trip_through_an_independent_decoder() {
        let original = tone(24_000);
        for format in [Format::Wav, Format::Flac] {
            let bytes = encode(&original, format).unwrap();
            let back = decode(&bytes, None).unwrap();
            assert_eq!(back.rate, 24_000, "{format:?}");
            assert_eq!(back.samples.len(), original.samples.len(), "{format:?}");
            let worst = back.samples.iter().zip(&original.samples).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst < 1e-4, "{format:?}: max error {worst}");
        }
    }

    /// A rising chirp: aperiodic, so any misalignment shows as lost
    /// correlation rather than hiding a whole period away.
    fn chirp(rate: u32, secs: f32) -> Pcm {
        let n = (rate as f32 * secs) as usize;
        let samples = (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (std::f32::consts::TAU * (200.0 * t + 600.0 * t * t)).sin() * 0.4
            })
            .collect();
        Pcm { samples, rate }
    }

    fn correlation(a: &[f32], b: &[f32]) -> f32 {
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
    }

    /// Opus is lossy, so the check is structural: symphonia's Ogg demuxer
    /// reads our pages, the priming and padding are trimmed to the sample,
    /// and what remains lines up with the input in time.
    #[test]
    fn opus_decodes_to_the_same_length_and_timing() {
        for (rate, secs) in [(24_000, 1.3), (22_050, 0.7), (16_000, 1.0), (48_000, 0.02)] {
            let original = chirp(rate, secs);
            let bytes = encode(&original, Format::Opus).unwrap();
            assert_eq!(&bytes[..4], b"OggS");
            let back = decode(&bytes, None).unwrap();
            assert_eq!(back.rate, 48_000);
            let expected = crate::resample(original, 48_000).unwrap();
            let len = back.samples.len() as isize - expected.samples.len() as isize;
            assert!(len.abs() <= 2, "{rate} Hz: {} samples back for {}", back.samples.len(), expected.samples.len());
            if secs < 0.1 {
                continue; // one frame: too short to judge the codec's fidelity
            }
            let r = correlation(&back.samples, &expected.samples);
            // Either opus-rs bug we route around (#37: 24 kHz input; #38:
            // SILK/hybrid timing) still scores 0.9; exact CELT reaches 0.999.
            assert!(r > 0.99, "{rate} Hz: correlation {r}");
            // A pre-skip off by even a millisecond would fall well short.
            let shifted = correlation(&back.samples[48..], &expected.samples);
            assert!(r > shifted + 0.1, "{rate} Hz: aligned {r}, shifted by 1 ms {shifted}");
        }
    }

    #[test]
    fn opus_of_silence_is_small_and_valid() {
        let bytes = encode(&Pcm { samples: vec![0.0; 24_000 * 10], rate: 24_000 }, Format::Opus).unwrap();
        assert!(bytes.len() < 10_000, "{} bytes", bytes.len());
        assert_eq!(decode(&bytes, None).unwrap().samples.len(), 480_000);
    }

    #[test]
    fn wav_header_is_canonical() {
        let bytes = encode(&Pcm { samples: vec![0.0; 10], rate: 24_000 }, Format::Wav).unwrap();
        assert_eq!(bytes.len(), 44 + 20);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 24_000);
    }

    #[test]
    fn pcm_is_raw_little_endian_i16() {
        let bytes = encode(&Pcm { samples: vec![1.0, -1.0, 0.0], rate: 24_000 }, Format::Pcm).unwrap();
        assert_eq!(bytes, [0xff, 0x7f, 0x01, 0x80, 0, 0]);
    }
}
