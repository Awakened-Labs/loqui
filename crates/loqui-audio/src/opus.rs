//! Opus, the codec of Discord voice notes, WhatsApp and Telegram voice
//! messages, and browser recordings (Ogg or WebM). symphonia demuxes those
//! containers but ships no Opus decoder, so packets are decoded here by
//! opus-rs, a pure-Rust port of libopus.

use opus_rs::OpusDecoder;

use crate::Error;

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
        if !cpu_can_decode() {
            return Err(Error::Decode(
                "Opus is not decodable on this CPU: it has AVX without FMA, which crashes opus-rs (restsend/opus-rs#30)".into(),
            ));
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

/// The pre-skip field of an OpusHead, in 48 kHz samples (RFC 7845 §5.1).
fn pre_skip(head: &[u8]) -> usize {
    match head {
        [b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', _version, _channels, lo, hi, ..] => usize::from(u16::from_le_bytes([*lo, *hi])),
        _ => 0,
    }
}

/// opus-rs runs FMA instructions after checking only for AVX, so a CPU with
/// AVX and no FMA (Sandy/Ivy Bridge, Bulldozer, some VMs) dies of SIGILL
/// mid-decode. Refusing Opus there turns a crash into an error. Remove once
/// an opus-rs release gates on FMA (restsend/opus-rs#31).
fn cpu_can_decode() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        !std::arch::is_x86_feature_detected!("avx") || std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}
