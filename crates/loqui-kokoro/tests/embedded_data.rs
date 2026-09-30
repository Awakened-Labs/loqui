//! The phoneme vocabulary must be exactly the table from hexgrad/Kokoro-82M's
//! `config.json` (revision f3ff3571791e39611d31c381e3a41a3af07b4987): a
//! changed id silently garbles every utterance.

#[test]
fn vocab_matches_the_upstream_config() {
    use std::fmt::Write;
    let bytes = include_bytes!("../data/vocab.json");
    let table: std::collections::BTreeMap<String, i64> = serde_json::from_slice(bytes).unwrap();
    assert_eq!(table.len(), 114);
    // A canonical rendering, so the pin does not depend on JSON whitespace.
    let mut canonical = String::new();
    for (k, v) in &table {
        writeln!(canonical, "{k}\t{v}").unwrap();
    }
    assert_eq!(fnv1a(canonical.as_bytes()), 0x5a95_e8e2_a38e_c571, "vocabulary changed");
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}
