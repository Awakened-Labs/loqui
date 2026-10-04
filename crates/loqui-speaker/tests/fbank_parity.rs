//! The filterbank against kaldi-native-fbank, the implementation sherpa-onnx
//! feeds these models with. The fixtures are `tools/parity/fbank_ref.py`'s: a
//! synthetic signal (tones, a chirp, noise) and the reference's log mel
//! energies for it, before mean subtraction.

use loqui_speaker::{Fbank, MEL_BINS};

fn read<T, const N: usize>(bytes: &[u8], from: fn([u8; N]) -> T) -> Vec<T> {
    bytes.as_chunks::<N>().0.iter().map(|c| from(*c)).collect()
}

#[test]
fn fbank_matches_kaldi_native_fbank() {
    let pcm = read(include_bytes!("data/fbank-input.i16"), i16::from_le_bytes);
    let expected = read(include_bytes!("data/fbank-expected.f32"), f32::from_le_bytes);
    let samples: Vec<f32> = pcm.iter().map(|&s| f32::from(s) / 32_768.0).collect();
    let features = Fbank::new().compute(&samples);
    assert_eq!(features.frames * MEL_BINS, expected.len(), "frame count");
    let worst = features.data.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    // Log energies of 10 to 30: 2e-3 is float rounding, not a different filter.
    assert!(worst < 2e-3, "max |delta| {worst}");
}

/// The whole encoder against Python's onnxruntime on the same model: the
/// filterbank, the mean subtraction, the tensor layout and the output name all
/// have to agree for the embeddings to. Needs the pinned model, so it is
/// ignored by default; run it with
/// `LOQUI_SPEAKER_MODEL=/path/wespeaker_en_voxceleb_resnet34_LM.onnx cargo test
/// -p loqui-speaker -- --ignored`.
#[test]
#[ignore = "needs the speaker model; set LOQUI_SPEAKER_MODEL (see tools/parity/README.md)"]
fn the_embedding_matches_onnxruntime_in_python() {
    let model = std::env::var("LOQUI_SPEAKER_MODEL").expect("LOQUI_SPEAKER_MODEL");
    let encoder = loqui_speaker::SpeakerEncoder::load(std::path::Path::new(&model), None).expect("load");
    let pcm = read(include_bytes!("data/fbank-input.i16"), i16::from_le_bytes);
    let samples: Vec<f32> = pcm.iter().map(|&s| f32::from(s) / 32_768.0).collect();
    let expected = read(include_bytes!("data/embedding-expected.f32"), f32::from_le_bytes);
    let embedding = encoder.embed(&samples).expect("embed");
    assert_eq!(embedding.len(), expected.len());
    let similarity = loqui_speaker::cosine(&embedding, &expected);
    assert!(similarity > 0.9999, "cosine {similarity}");
}
