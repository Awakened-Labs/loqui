//! MP3, behind the `mp3` feature: LAME 3.100 through mp3lame-encoder, built
//! from bundled C source and linked statically. LAME is LGPL, so it is off
//! by default and loqui carries no copyleft code unless a build asks for it.

use mp3lame_encoder::{Bitrate, Builder, FlushGap, Mode, MonoPcm, Quality, max_required_buffer_size};

use crate::{Error, Pcm};

/// Speech-grade CBR: at Kokoro's 24 kHz (MPEG-2 Layer III) 64 kbps is
/// comfortably clean for one voice. 8 KB a second.
const BITRATE: Bitrate = Bitrate::Kbps64;

pub(crate) fn encode(pcm: &Pcm) -> Result<Vec<u8>, Error> {
    let build = |e: mp3lame_encoder::BuildError| Error::Encode(format!("MP3: {e}"));
    let mut builder = Builder::new().ok_or_else(|| Error::Encode("MP3: LAME could not allocate an encoder".into()))?;
    builder.set_num_channels(1).map_err(build)?;
    builder.set_sample_rate(pcm.rate).map_err(build)?;
    builder.set_mode(Mode::Mono).map_err(build)?;
    builder.set_brate(BITRATE).map_err(build)?;
    builder.set_quality(Quality::NearBest).map_err(build)?;
    // The LAME tag records the encoder delay and end padding, which decoders
    // (symphonia, ffmpeg, browsers) trim to play exactly `pcm`'s samples.
    builder.set_to_write_vbr_tag(true).map_err(build)?;
    let mut encoder = builder.build().map_err(build)?;

    let encode = |e: mp3lame_encoder::EncodeError| Error::Encode(format!("MP3: {e}"));
    let samples: Vec<f32> = pcm.samples.iter().map(|s| s.clamp(-1.0, 1.0)).collect();
    let mut out = Vec::with_capacity(max_required_buffer_size(samples.len()));
    encoder.encode_to_vec(MonoPcm(&samples[..]), &mut out).map_err(encode)?;
    out.reserve(max_required_buffer_size(0));
    encoder.flush_to_vec::<FlushGap>(&mut out).map_err(encode)?;

    // LAME leaves the first frame blank for the tag, which it can only fill
    // in once the stream is complete.
    let mut tag = Vec::with_capacity(encoder.lame_tag_size());
    encoder.lame_tag_encode_to_vec(&mut tag).ok_or_else(|| Error::Encode("MP3: LAME wrote no tag frame".into()))?;
    let start = encoder.id3v2_tag_size();
    out.get_mut(start..start + tag.len())
        .ok_or_else(|| Error::Encode("MP3: stream is shorter than its tag frame".into()))?
        .copy_from_slice(&tag);
    Ok(out)
}
