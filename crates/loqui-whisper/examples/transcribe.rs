//! Transcribes audio files with a GGML Whisper model, for parity checks.
//!
//!     transcribe MODEL.bin FILE [FILE ...] [--language en] [--gpu N] [--dir DIR --out OUT.txt]
//!
//! With `--dir`, every `NNNN.wav` in DIR is transcribed in name order and
//! one line per file is written to OUT.txt, with one model load. `--gpu N`
//! needs the `cuda` or `vulkan` feature.

use std::path::{Path, PathBuf};
use std::time::Instant;

use loqui_whisper::{Device, Options, Whisper};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let model = args.first().ok_or("usage: transcribe MODEL.bin FILE...")?;
    let options = Options { language: flag("--language"), ..Options::default() };

    let started = Instant::now();
    let device = flag("--gpu").map(|n| n.parse()).transpose()?.map_or(Device::Cpu, Device::Gpu);
    let whisper = Whisper::load(Path::new(model), device)?;
    eprintln!("model loaded in {:.1}s (multilingual: {})", started.elapsed().as_secs_f32(), whisper.is_multilingual());

    let mut files: Vec<PathBuf> = match flag("--dir") {
        Some(dir) => {
            std::fs::read_dir(dir)?.filter_map(Result::ok).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wav")).collect()
        }
        None => args.iter().skip(1).filter(|a| !a.starts_with("--") && Path::new(a).is_file()).map(PathBuf::from).collect(),
    };
    files.sort();
    let (mut lines, mut audio_secs, started) = (Vec::new(), 0.0, Instant::now());
    for file in &files {
        let pcm = loqui_audio::decode_mono_16k(&std::fs::read(file)?, Some(1800))?;
        audio_secs += pcm.duration_secs();
        let result = whisper.transcribe(&pcm.samples, &options)?;
        if flag("--out").is_none() {
            println!("{}: [{}] {}", file.display(), result.language.as_deref().unwrap_or("?"), result.text);
        }
        lines.push(result.text);
    }
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!("{audio_secs:.1}s of audio in {elapsed:.1}s (real-time factor {:.3})", elapsed / audio_secs.max(1e-6));
    if let Some(out) = flag("--out") {
        std::fs::write(out, lines.join("\n") + "\n")?;
    }
    Ok(())
}
