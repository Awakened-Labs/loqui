//! Speaks text into a 16-bit 24 kHz WAV file, for listening tests and the
//! parity harness.
//!
//!     speak MODEL.onnx VOICES_DIR OUT.wav "text" [--voice af_heart] [--speed 1.0]
//!           [--phonemes] [--raw OUT.f32] [--threads N]
//!     speak MODEL.onnx VOICES_DIR OUT_DIR --lines FILE [--voice af_heart]
//!
//! `--voice` takes a pack id, an OpenAI name or a blend such as
//! `af_bella(2)+af_sky(1)`. `--phonemes` treats the text as a phoneme string
//! and skips G2P; `--raw` also writes the unprocessed model output as
//! little-endian f32. `--lines` speaks each line of FILE to
//! OUT_DIR/NNNN.wav with one model load.

use std::path::PathBuf;
use std::time::Instant;

use loqui_g2p::OovFallback;
use loqui_kokoro::{Blend, Device, Kokoro, KokoroModel, SAMPLE_RATE, Voice, post};

fn wav(samples: &[f32]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend(b"RIFF");
    out.extend((36 + data_len).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes()); // PCM
    out.extend(1u16.to_le_bytes()); // mono
    out.extend(SAMPLE_RATE.to_le_bytes());
    out.extend((SAMPLE_RATE * 2).to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend(data_len.to_le_bytes());
    for s in samples {
        out.extend(((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let positional: Vec<&String> = {
        let mut skip = false;
        args.iter()
            .filter(|a| {
                if skip {
                    skip = false;
                    return false;
                }
                if matches!(a.as_str(), "--voice" | "--speed" | "--raw" | "--threads" | "--lines") {
                    skip = true;
                    return false;
                }
                !a.starts_with("--")
            })
            .collect()
    };
    if let Some(lines) = flag("--lines") {
        let [model_path, voices, out_dir] = positional[..] else {
            return Err("usage: speak MODEL.onnx VOICES_DIR OUT_DIR --lines FILE".into());
        };
        let voice: Blend = flag("--voice").as_deref().unwrap_or("af_heart").parse()?;
        let model = KokoroModel::load(&PathBuf::from(model_path), Device::Cpu, None)?;
        let kokoro = Kokoro::new(model, PathBuf::from(voices), OovFallback::Neural);
        std::fs::create_dir_all(out_dir)?;
        let (started, mut audio_secs) = (Instant::now(), 0.0);
        for (i, line) in std::fs::read_to_string(lines)?.lines().enumerate() {
            let audio = kokoro.speak(line, &voice, 1.0)?;
            audio_secs += audio.len() as f32 / SAMPLE_RATE as f32;
            std::fs::write(PathBuf::from(out_dir).join(format!("{:04}.wav", i + 1)), wav(&audio))?;
        }
        let elapsed = started.elapsed().as_secs_f32();
        eprintln!("{audio_secs:.1}s of audio in {elapsed:.1}s (real-time factor {:.3})", elapsed / audio_secs.max(1e-6));
        return Ok(());
    }
    let [model_path, voices, out, text] = positional[..] else {
        return Err("usage: speak MODEL.onnx VOICES_DIR OUT.wav TEXT [--voice V] [--speed S] [--phonemes] [--raw F]".into());
    };
    let voice: Blend = flag("--voice").as_deref().unwrap_or("af_heart").parse()?;
    let speed: f32 = flag("--speed").map_or(Ok(1.0), |s| s.parse())?;
    let threads = flag("--threads").map(|t| t.parse()).transpose()?;

    let started = Instant::now();
    let model = KokoroModel::load(&PathBuf::from(model_path), Device::Cpu, threads)?;
    eprintln!("model loaded in {:.2}s", started.elapsed().as_secs_f32());

    let started = Instant::now();
    let raw = if args.iter().any(|a| a == "--phonemes") {
        // Mixed here from the public pieces, as `Kokoro` mixes them.
        let packs = voice.audible().map(|(id, w)| Ok((Voice::from_file(&PathBuf::from(voices).join(format!("{id}.bin")))?, w)));
        let packs = packs.collect::<Result<Vec<_>, loqui_kokoro::Error>>()?;
        let style = Voice::blend(&packs.iter().map(|(v, w)| (v, *w)).collect::<Vec<_>>())?;
        model.synthesize(text, &style, speed)?
    } else {
        let kokoro = Kokoro::new(model, PathBuf::from(voices), OovFallback::Neural);
        kokoro.speak_raw(text, &voice, speed)?
    };
    let elapsed = started.elapsed().as_secs_f32();
    let seconds = raw.len() as f32 / SAMPLE_RATE as f32;
    eprintln!("{seconds:.2}s of audio in {elapsed:.2}s (real-time factor {:.3})", elapsed / seconds.max(1e-6));

    if let Some(path) = flag("--raw") {
        std::fs::write(path, raw.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    let processed = post::normalize_peak(post::trim_silence(&raw, post::TRIM_THRESHOLD), post::PEAK);
    std::fs::write(out, wav(&processed))?;
    Ok(())
}
