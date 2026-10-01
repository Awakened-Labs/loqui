//! Opus, the codec of Discord voice notes, WhatsApp and Telegram voice
//! messages, and browser recordings (Ogg or WebM). symphonia demuxes those
//! containers but ships no Opus codec, so packets are decoded, and speech
//! encoded, here by opus-rs, a pure-Rust port of libopus.

use opus_rs::{Application, OpusDecoder, OpusEncoder};

use crate::{Error, Pcm, ogg, resample};

/// Opus always decodes to 48 kHz.
pub(crate) const RATE: u32 = 48_000;

/// The longest Opus packet: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5_760;

pub(crate) struct Opus {
    decoder: OpusDecoder,
    channels: usize,
    /// Encoder priming still to drop, in samples per channel.
    skip: usize,
    out: Vec<f32>,
}

impl Opus {
    /// `head` is the stream's OpusHead (the codec's extra data in Ogg and in
    /// WebM alike), which carries the encoder priming to drop.
    pub(crate) fn new(channels: usize, head: Option<&[u8]>) -> Result<Self, Error> {
        if !cpu_supported() {
            return Err(Error::Decode(UNSUPPORTED_CPU.into()));
        }
        if !(1..=2).contains(&channels) {
            return Err(Error::Decode(format!("Opus with {channels} channels is not supported; mono and stereo are")));
        }
        let decoder = OpusDecoder::new(RATE as i32, channels).map_err(|e| Error::Decode(format!("Opus: {e}")))?;
        Ok(Self { decoder, channels, skip: head.map_or(0, pre_skip), out: vec![0.0; MAX_FRAME * channels] })
    }

    pub(crate) fn channels(&self) -> usize {
        self.channels
    }

    /// Decodes one packet to interleaved samples, less `trim_end` samples of
    /// end padding. `None` for a packet that cannot be decoded, which the
    /// caller skips as it skips any corrupt one.
    pub(crate) fn decode(&mut self, packet: &[u8], trim_end: usize) -> Option<&[f32]> {
        let frames = self.decoder.decode(packet, MAX_FRAME, &mut self.out).ok()?;
        let start = frames.min(self.skip);
        self.skip -= start;
        let end = frames.saturating_sub(trim_end).max(start);
        Some(&self.out[start * self.channels..end * self.channels])
    }
}

/// The encoder's lookahead, in 48 kHz samples: 2.5 ms of CELT overlap plus
/// 4 ms of delay compensation, as in libopus, and as measured on opus-rs;
/// written as the OpusHead's pre-skip so players drop it.
const PRE_SKIP: u16 = 312;

/// High for one voice, and chosen for opus-rs rather than for speech: below
/// 64 kbps its SILK and hybrid paths mangle the first ~200 ms and run up to
/// 17 samples behind the pre-skip (restsend/opus-rs#38); at 64 kbps it stays
/// in CELT, which is exact from the first sample. 8 KB a second.
const BITRATE: i32 = 64_000;

/// Encodes speech as Ogg Opus (RFC 7845): mono, 20 ms packets, the priming
/// and end padding marked so a player plays exactly `pcm`'s samples.
pub(crate) fn encode_ogg(pcm: &Pcm) -> Result<Vec<u8>, Error> {
    if !cpu_supported() {
        return Err(Error::Encode(UNSUPPORTED_CPU.into()));
    }
    // Opus accepts 8 to 48 kHz input, but opus-rs 0.1.34 encodes 24 kHz
    // (Kokoro's rate) into garbage in every mode (restsend/opus-rs#37),
    // while 48 kHz round-trips faithfully. So everything is encoded from 48.
    let input = resample(pcm.clone(), RATE)?;
    let frame = RATE as usize / 50;
    let mut encoder = OpusEncoder::new(RATE as i32, 1, Application::Audio).map_err(|e| Error::Encode(format!("Opus: {e}")))?;
    encoder.bitrate_bps = BITRATE;

    // Feed the lookahead's worth of silence past the end, so the last real
    // samples leave the encoder, then round up to whole frames.
    let len = input.samples.len();
    let mut samples: Vec<f32> = input.samples.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect();
    samples.resize((len + usize::from(PRE_SKIP)).div_ceil(frame) * frame, 0.0);
    let end = u64::from(PRE_SKIP) + len as u64;

    let mut ogg = ogg::Writer::new(u32::from_le_bytes(*b"loqi"));
    ogg.packet(&opus_head(pcm.rate), 0, true);
    ogg.packet(&opus_tags(), 0, true);
    let mut packet = [0u8; 4_000];
    for (i, chunk) in samples.chunks_exact(frame).enumerate() {
        let len = encoder.encode(chunk, frame, &mut packet).map_err(|e| Error::Encode(format!("Opus: {e}")))?;
        // A packet's granule is where it ends; the last one ends at the
        // real audio's end, which is how Ogg Opus trims the padding.
        let granule = (((i + 1) * frame) as u64).min(end);
        ogg.packet(&packet[..len], granule, false);
    }
    Ok(ogg.finish())
}

/// The identification header (RFC 7845 §5.1): mono, channel mapping 0.
fn opus_head(input_rate: u32) -> Vec<u8> {
    let mut head = b"OpusHead".to_vec();
    head.push(1); // version
    head.push(1); // channels
    head.extend(PRE_SKIP.to_le_bytes());
    head.extend(input_rate.to_le_bytes()); // informational only
    head.extend(0i16.to_le_bytes()); // output gain
    head.push(0); // mapping family
    head
}

/// The comment header (RFC 7845 §5.2): a vendor string and no comments.
fn opus_tags() -> Vec<u8> {
    let vendor = concat!("loqui ", env!("CARGO_PKG_VERSION"), " (opus-rs)");
    let mut tags = b"OpusTags".to_vec();
    tags.extend((vendor.len() as u32).to_le_bytes());
    tags.extend(vendor.as_bytes());
    tags.extend(0u32.to_le_bytes());
    tags
}

/// The pre-skip field of an OpusHead, in 48 kHz samples (RFC 7845 §5.1).
fn pre_skip(head: &[u8]) -> usize {
    match head {
        [b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', _version, _channels, lo, hi, ..] => usize::from(u16::from_le_bytes([*lo, *hi])),
        _ => 0,
    }
}

const UNSUPPORTED_CPU: &str = "Opus is unavailable on this CPU: it has AVX without FMA, which crashes opus-rs (restsend/opus-rs#30)";

/// opus-rs runs FMA instructions after checking only for AVX, so a CPU with
/// AVX and no FMA (Sandy/Ivy Bridge, Bulldozer, some VMs) dies of SIGILL
/// mid-codec. Refusing Opus there turns a crash into an error. Remove once
/// an opus-rs release gates on FMA (restsend/opus-rs#31).
fn cpu_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        !std::arch::is_x86_feature_detected!("avx") || std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}
