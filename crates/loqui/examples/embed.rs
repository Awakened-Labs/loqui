//! loqui in process, the narrow way: no listener, no downloads at run time,
//! limits sized for this program, and the engine shared between threads.
//!
//!     loqui fetch                                # once, where the network is allowed
//!     cargo run -p loqui --features whisper --example embed

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let builder = loqui::Engine::builder()
        // Never reach the network: weights must already be in the cache.
        .downloads(loqui::Downloads::Deny)
        // Bound what one call may cost; the defaults are 4096 and 1800.
        .max_input_chars(1000)
        .max_audio_secs(120);
    // Speech-to-text is off unless asked for.
    #[cfg(feature = "whisper")]
    let builder = builder.stt(Some(loqui::SttConfig::default()));
    let engine = builder.build()?;

    // Clones share the loaded models, so each worker thread gets its own.
    let worker = engine.clone();
    let speech = std::thread::spawn(move || {
        let mut request = loqui::SpeakRequest::new("Speech, made on this machine.");
        request.format = loqui::Format::Opus;
        worker.speak(&request)
    })
    .join()
    .expect("speech thread panicked")?;
    std::fs::write("embed.opus", &speech.audio)?;
    println!("embed.opus: {:.1} s of speech", speech.duration_secs);

    #[cfg(feature = "whisper")]
    {
        let request = loqui::TranscribeRequest { audio: speech.audio, ..Default::default() };
        println!("heard: {}", engine.transcribe(&request)?.text);
    }
    Ok(())
}
