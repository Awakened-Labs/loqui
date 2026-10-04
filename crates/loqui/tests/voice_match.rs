//! Matching against real models: Kokoro and the pinned speaker model.
//!
//! Ignored by default: they need the weights (the speaker model, 26 MB, is
//! fetched on first run into the default cache) and minutes of CPU. Run them
//! with `cargo test -p loqui --test voice_match -- --ignored --test-threads 1`
//! and record the outcome in `tools/parity/README.md`'s results log. The
//! targets are Kokoro's own voices and blends, spoken on a sentence the search
//! never speaks, so what a match should find is known.

use std::ops::ControlFlow;

use loqui::{Blend, Engine, Format, MatchRequest, SpeakRequest, SpeakerConfig};

const HELD_OUT: &str = "The juice of lemons makes fine punch. The box was thrown beside the parked truck.";
const OTHER: &str = "It's easy to tell the depth of a well. These days a chicken leg is a rare dish.";

fn engine() -> Engine {
    Engine::builder().speaker(Some(SpeakerConfig::default())).build().expect("engine")
}

fn spoken(engine: &Engine, text: &str, voice: &str) -> Vec<u8> {
    let request = SpeakRequest { text: text.into(), voice: voice.into(), speed: 1.0, format: Format::Wav };
    engine.speak(&request).expect("speak").audio
}

fn weight(spec: &str, voice: &str) -> f32 {
    let blend: Blend = spec.parse().expect("a match's spec parses");
    blend.parts().iter().find(|(id, _)| id == voice).map_or(0.0, |(_, w)| *w)
}

fn quiet(_: &loqui::MatchProgress) -> ControlFlow<()> {
    ControlFlow::Continue(())
}

#[test]
#[ignore = "needs Kokoro and the speaker model; see the module docs"]
fn the_same_voice_scores_above_other_voices() {
    let engine = engine();
    let bella = engine.speaker_embedding(&spoken(&engine, HELD_OUT, "af_bella")).unwrap();
    let bella_again = engine.speaker_embedding(&spoken(&engine, OTHER, "af_bella")).unwrap();
    for other in ["af_sky", "bf_emma", "am_michael", "bm_george"] {
        let them = engine.speaker_embedding(&spoken(&engine, OTHER, other)).unwrap();
        let (same, different) = (bella.similarity(&bella_again), bella.similarity(&them));
        assert!(same > different + 0.1, "af_bella {same:.3} vs {other} {different:.3}");
    }
}

#[test]
#[ignore = "needs Kokoro and the speaker model; see the module docs"]
fn a_synthesized_blend_is_recovered() {
    let engine = engine();
    let target = spoken(&engine, HELD_OUT, "af_bella(70)+af_sky(30)");
    let found = engine.match_voice(&MatchRequest::new(target), &mut quiet).unwrap();
    eprintln!("{found:?}");
    let bella = weight(&found.spec, "af_bella");
    let sky = weight(&found.spec, "af_sky");
    assert!(bella + sky >= 0.8, "the parents carry the blend: {}", found.spec);
    assert!((bella - 0.7).abs() <= 0.15, "af_bella at {bella}: {}", found.spec);
    assert!(found.similarity >= 0.7, "{found:?}");
    assert!(found.similarity >= found.closest_similarity, "{found:?}");
    assert!(found.evaluations <= 36, "{found:?}");
}

#[test]
#[ignore = "needs Kokoro and the speaker model; see the module docs"]
fn a_single_voice_is_recovered_as_dominant() {
    let engine = engine();
    let target = spoken(&engine, HELD_OUT, "bm_george");
    let found = engine.match_voice(&MatchRequest::new(target), &mut quiet).unwrap();
    eprintln!("{found:?}");
    assert_eq!(found.closest_voice, "bm_george", "{found:?}");
    assert!(weight(&found.spec, "bm_george") >= 0.6, "{}", found.spec);
    assert!(found.spec.starts_with("bm_") || found.spec.starts_with("bf_"), "a British accent: {}", found.spec);
}

#[test]
#[ignore = "needs Kokoro and the speaker model; see the module docs"]
fn matching_is_deterministic() {
    let engine = engine();
    let target = spoken(&engine, HELD_OUT, "am_michael(60)+am_onyx(40)");
    let mut request = MatchRequest::new(target);
    request.evaluations = 12;
    let first = engine.match_voice(&request, &mut quiet).unwrap();
    let second = engine.match_voice(&request, &mut quiet).unwrap();
    assert_eq!(first.spec, second.spec);
    assert!((first.similarity - second.similarity).abs() < 1e-4, "{first:?} / {second:?}");
}
