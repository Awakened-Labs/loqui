//! The embedded model and lexicons must be byte-for-byte the upstream
//! releases named in NOTICE. A silent swap would change every voice.

use sha2::{Digest, Sha256};

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn embedded_files_match_their_upstream_releases() {
    let pinned: [(&str, &[u8], &str); 6] = [
        ("g2p-en-us.safetensors", include_bytes!("../data/g2p-en-us.safetensors"),
         "dc4a02e62d4fcb4bb4097ecf00db89b8e1a12a549a52ab6adfbba220b80a55c5"),
        ("g2p-en-gb.safetensors", include_bytes!("../data/g2p-en-gb.safetensors"),
         "4994f474bb6f4584076a4e98189caaec11aa3f773a0d82d5bc263a03dd07e703"),
        ("misaki-us_gold.json", include_bytes!("../data/misaki-us_gold.json"), US_GOLD),
        ("misaki-us_silver.json", include_bytes!("../data/misaki-us_silver.json"), US_SILVER),
        ("misaki-gb_gold.json", include_bytes!("../data/misaki-gb_gold.json"), GB_GOLD),
        ("misaki-gb_silver.json", include_bytes!("../data/misaki-gb_silver.json"), GB_SILVER),
    ];
    for (name, bytes, expected) in pinned {
        assert_eq!(sha256(bytes), expected, "{name} differs from its pinned upstream release");
    }
}

// misaki 0.9.4 wheel (sha256 90e2eeb1…da15a), misaki/data/*.json.
const US_GOLD: &str = "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f";
const US_SILVER: &str = "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475";
const GB_GOLD: &str = "29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71";
const GB_SILVER: &str = "48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce";
