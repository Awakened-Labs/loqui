//! The out-of-vocabulary fallback: a tiny character-level BART that maps an
//! English spelling straight to misaki phonemes.
//!
//! The checkpoints are `PeterReid/graphemes_to_phonemes_en_us` and `_en_gb`
//! (Apache-2.0): a one-layer encoder and one-layer decoder with
//! `d_model = 128`, trained on the misaki gold and silver lexicons of each
//! dialect. It therefore emits Kokoro's phoneme
//! alphabet directly, with no IPA-to-misaki mapping step.
//!
//! The forward pass is written out by hand rather than run through an ONNX
//! runtime. The model is 3 MB and a few hundred thousand multiply-adds per
//! step, so this keeps `loqui-g2p` free of native dependencies. It follows
//! Hugging Face's `modeling_bart` exactly: learned positions offset by 2,
//! `layernorm_embedding`, post-layer-norm blocks, erf GELU, a tied LM head
//! plus `final_logits_bias`, and greedy decoding from `decoder_start = <s>`.

use std::collections::HashMap;

use crate::{Dialect, Error};

const D: usize = 128;
const FFN: usize = 1024;
const MAX_POSITIONS: usize = 64;
/// BART's learned positional embedding reserves the first two rows.
const POSITION_OFFSET: usize = 2;
const LN_EPS: f32 = 1e-5;

const PAD: usize = 0;
const BOS: usize = 1;
const EOS: usize = 2;
const SPECIALS: usize = 4;

/// One embedded checkpoint and the character vocabularies from its
/// `config.json`. Index `i` of each string is token id `i`; the four leading
/// underscores stand in for `<pad> <s> </s> <unk>`.
struct Checkpoint {
    weights: &'static [u8],
    graphemes: &'static str,
    phonemes: &'static str,
}

/// Revision a5631b285d18d59483c32c0c3379cb9fac924f4b. SHA-256 pinned by a test.
const EN_US: Checkpoint = Checkpoint {
    weights: include_bytes!("../data/g2p-en-us.safetensors"),
    graphemes: "____AIOWYbdfhijklmnpstuvwz'-.BCDEFGHJKLMNPQRSTUVXZacegoqrxy",
    phonemes: "____AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʔʤʧˈˌθᵊᵻ",
};

/// Revision d8357d5067fa26a5c34134d6bbcf4bbf000c0ac8. SHA-256 pinned by a test.
const EN_GB: Checkpoint = Checkpoint {
    weights: include_bytes!("../data/g2p-en-gb.safetensors"),
    graphemes: "____AIQWYabdfhijklmnpstuvwz'-.BCDEFGHJKLMNOPRSTUVXZcegoqrxy",
    phonemes: "____AIQWYabdfhijklmnpstuvwzðŋɑɒɔəɛɜɡɪɹʃʊʌʒʤʧˈˌːθᵊ",
};

impl Checkpoint {
    fn for_dialect(dialect: Dialect) -> &'static Checkpoint {
        match dialect {
            Dialect::American => &EN_US,
            Dialect::British => &EN_GB,
        }
    }
}

/// The longest spelling the model can take: its 64 positions hold the
/// word plus `<s>` and `</s>`.
pub const MAX_WORD_CHARS: usize = MAX_POSITIONS - 2;

struct Linear {
    /// Row-major `[out][in]`, as PyTorch stores it.
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        let mut out = self.bias.clone();
        for (o, row) in out.iter_mut().zip(self.weight.chunks_exact(self.in_dim)) {
            *o += dot(row, x);
        }
        debug_assert_eq!(out.len(), self.out_dim);
        out
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    fn forward_in_place(&self, x: &mut [f32]) {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for ((v, w), b) in x.iter_mut().zip(&self.weight).zip(&self.bias) {
            *v = (*v - mean) * inv * w + b;
        }
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

impl Attention {
    /// Single-head, unmasked attention of each query over all of `keys_values`.
    fn forward(&self, queries: &[Vec<f32>], keys_values: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let scale = (D as f32).powf(-0.5);
        let keys: Vec<Vec<f32>> = keys_values.iter().map(|x| self.k.forward(x)).collect();
        let values: Vec<Vec<f32>> = keys_values.iter().map(|x| self.v.forward(x)).collect();
        queries
            .iter()
            .map(|x| {
                let q: Vec<f32> = self.q.forward(x).iter().map(|v| v * scale).collect();
                let scores: Vec<f32> = keys.iter().map(|k| dot(&q, k)).collect();
                let weights = softmax(&scores);
                let mut mixed = vec![0.0; D];
                for (w, v) in weights.iter().zip(&values) {
                    for (m, x) in mixed.iter_mut().zip(v) {
                        *m += w * x;
                    }
                }
                self.out.forward(&mixed)
            })
            .collect()
    }
}

struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let hidden: Vec<f32> = self.fc1.forward(x).into_iter().map(gelu).collect();
        self.fc2.forward(&hidden)
    }
}

struct Embeddings {
    positions: Vec<f32>,
    norm: LayerNorm,
}

impl Embeddings {
    fn embed(&self, tokens: &[f32], ids: &[usize]) -> Vec<Vec<f32>> {
        ids.iter()
            .enumerate()
            .map(|(pos, &id)| {
                let token = &tokens[id * D..(id + 1) * D];
                let row = (pos + POSITION_OFFSET) * D;
                let position = &self.positions[row..row + D];
                let mut h: Vec<f32> = token.iter().zip(position).map(|(t, p)| t + p).collect();
                self.norm.forward_in_place(&mut h);
                h
            })
            .collect()
    }
}

/// The loaded model. Immutable after construction, so `Sync` and shareable.
pub struct NeuralG2p {
    shared: Vec<f32>,
    final_logits_bias: Vec<f32>,
    encoder_embeddings: Embeddings,
    encoder_attn: Attention,
    encoder_attn_norm: LayerNorm,
    encoder_ffn: FeedForward,
    encoder_ffn_norm: LayerNorm,
    decoder_embeddings: Embeddings,
    decoder_self_attn: Attention,
    decoder_self_attn_norm: LayerNorm,
    decoder_cross_attn: Attention,
    decoder_cross_attn_norm: LayerNorm,
    decoder_ffn: FeedForward,
    decoder_ffn_norm: LayerNorm,
    vocab_size: usize,
    grapheme_ids: HashMap<char, usize>,
    phonemes: Vec<char>,
}

impl std::fmt::Debug for NeuralG2p {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralG2p").field("vocab_size", &self.vocab_size).finish_non_exhaustive()
    }
}

impl NeuralG2p {
    /// Loads the embedded checkpoint for `dialect`.
    pub fn load(dialect: Dialect) -> Result<Self, Error> {
        let checkpoint = Checkpoint::for_dialect(dialect);
        let mut t = Tensors::parse(checkpoint.weights)?;
        let linear = |t: &mut Tensors, name: &str, in_dim: usize, out_dim: usize| -> Result<Linear, Error> {
            Ok(Linear {
                weight: t.take(&format!("{name}.weight"), in_dim * out_dim)?,
                bias: t.take(&format!("{name}.bias"), out_dim)?,
                in_dim,
                out_dim,
            })
        };
        let norm = |t: &mut Tensors, name: &str| -> Result<LayerNorm, Error> {
            Ok(LayerNorm { weight: t.take(&format!("{name}.weight"), D)?, bias: t.take(&format!("{name}.bias"), D)? })
        };
        let attention = |t: &mut Tensors, name: &str| -> Result<Attention, Error> {
            Ok(Attention {
                q: linear(t, &format!("{name}.q_proj"), D, D)?,
                k: linear(t, &format!("{name}.k_proj"), D, D)?,
                v: linear(t, &format!("{name}.v_proj"), D, D)?,
                out: linear(t, &format!("{name}.out_proj"), D, D)?,
            })
        };
        let ffn = |t: &mut Tensors, layer: &str| -> Result<FeedForward, Error> {
            Ok(FeedForward { fc1: linear(t, &format!("{layer}.fc1"), D, FFN)?, fc2: linear(t, &format!("{layer}.fc2"), FFN, D)? })
        };
        let embeddings = |t: &mut Tensors, stack: &str| -> Result<Embeddings, Error> {
            Ok(Embeddings {
                positions: t.take(&format!("model.{stack}.embed_positions.weight"), (MAX_POSITIONS + POSITION_OFFSET) * D)?,
                norm: norm(t, &format!("model.{stack}.layernorm_embedding"))?,
            })
        };

        let final_logits_bias = t.take_any("final_logits_bias")?;
        let vocab_size = final_logits_bias.len();
        let enc = "model.encoder.layers.0";
        let dec = "model.decoder.layers.0";
        let model = Self {
            shared: t.take("model.shared.weight", vocab_size * D)?,
            final_logits_bias,
            encoder_embeddings: embeddings(&mut t, "encoder")?,
            encoder_attn: attention(&mut t, &format!("{enc}.self_attn"))?,
            encoder_attn_norm: norm(&mut t, &format!("{enc}.self_attn_layer_norm"))?,
            encoder_ffn: ffn(&mut t, enc)?,
            encoder_ffn_norm: norm(&mut t, &format!("{enc}.final_layer_norm"))?,
            decoder_embeddings: embeddings(&mut t, "decoder")?,
            decoder_self_attn: attention(&mut t, &format!("{dec}.self_attn"))?,
            decoder_self_attn_norm: norm(&mut t, &format!("{dec}.self_attn_layer_norm"))?,
            decoder_cross_attn: attention(&mut t, &format!("{dec}.encoder_attn"))?,
            decoder_cross_attn_norm: norm(&mut t, &format!("{dec}.encoder_attn_layer_norm"))?,
            decoder_ffn: ffn(&mut t, dec)?,
            decoder_ffn_norm: norm(&mut t, &format!("{dec}.final_layer_norm"))?,
            vocab_size,
            grapheme_ids: checkpoint.graphemes.chars().enumerate().skip(SPECIALS).map(|(i, c)| (c, i)).collect(),
            phonemes: checkpoint.phonemes.chars().collect(),
        };
        if let Some(name) = t.remaining() {
            return Err(Error::Weights(format!("unexpected tensor {name}")));
        }
        Ok(model)
    }

    /// Phonemizes one word. The spelling is used as given, case included,
    /// because the model was trained on cased lexicon keys. Characters the
    /// model has never seen are folded to ASCII where possible and
    /// otherwise dropped. Returns `None` only when nothing encodable is
    /// left or the model emits nothing.
    pub fn phonemize(&self, word: &str) -> Option<String> {
        let ids: Vec<usize> = std::iter::once(BOS)
            .chain(fold(word).chars().filter_map(|c| self.grapheme_ids.get(&c).copied()).take(MAX_WORD_CHARS))
            .chain(std::iter::once(EOS))
            .collect();
        if ids.len() == 2 {
            return None;
        }
        let memory = self.encode(&ids);
        let decoded = self.decode_greedy(&memory);
        let out: String = decoded.into_iter().filter(|&id| id >= SPECIALS).filter_map(|id| self.phonemes.get(id)).collect();
        (!out.is_empty()).then_some(out)
    }

    fn encode(&self, ids: &[usize]) -> Vec<Vec<f32>> {
        let mut h = self.encoder_embeddings.embed(&self.shared, ids);
        let attended = self.encoder_attn.forward(&h, &h);
        residual_norm(&mut h, &attended, &self.encoder_attn_norm);
        let fed: Vec<Vec<f32>> = h.iter().map(|x| self.encoder_ffn.forward(x)).collect();
        residual_norm(&mut h, &fed, &self.encoder_ffn_norm);
        h
    }

    /// Greedy decoding. The decoder has a single layer, so only the newest
    /// position's output is ever needed, and it attends over the embeddings
    /// of the whole prefix: no causal mask and no KV cache are required.
    fn decode_greedy(&self, memory: &[Vec<f32>]) -> Vec<usize> {
        let mut ids = vec![BOS];
        while ids.len() < MAX_POSITIONS {
            let prefix = self.decoder_embeddings.embed(&self.shared, &ids);
            let mut h = vec![prefix.last().expect("decoder input is never empty").clone()];
            let attended = self.decoder_self_attn.forward(&h, &prefix);
            residual_norm(&mut h, &attended, &self.decoder_self_attn_norm);
            let crossed = self.decoder_cross_attn.forward(&h, memory);
            residual_norm(&mut h, &crossed, &self.decoder_cross_attn_norm);
            let fed = vec![self.decoder_ffn.forward(&h[0])];
            residual_norm(&mut h, &fed, &self.decoder_ffn_norm);

            let next = (0..self.vocab_size)
                .map(|id| (id, dot(&self.shared[id * D..(id + 1) * D], &h[0]) + self.final_logits_bias[id]))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(EOS, |(id, _)| id);
            if next == EOS || next == PAD {
                break;
            }
            ids.push(next);
        }
        ids.remove(0);
        ids
    }
}

/// `h = norm(h + delta)`, row by row: BART's post-layer-norm residual.
fn residual_norm(h: &mut [Vec<f32>], delta: &[Vec<f32>], norm: &LayerNorm) {
    for (row, d) in h.iter_mut().zip(delta) {
        for (v, x) in row.iter_mut().zip(d) {
            *v += x;
        }
        norm.forward_in_place(row);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}

/// Folds the Latin-1 letters an English text realistically carries
/// ("café", "naïve", "Zürich") onto the ASCII the model knows, and maps
/// typographic apostrophes and dashes onto `'` and `-`.
fn fold(word: &str) -> String {
    word.chars()
        .map(|c| match c {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
            'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => 'A',
            'è' | 'é' | 'ê' | 'ë' => 'e',
            'È' | 'É' | 'Ê' | 'Ë' => 'E',
            'ì' | 'í' | 'î' | 'ï' => 'i',
            'Ì' | 'Í' | 'Î' | 'Ï' => 'I',
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' => 'o',
            'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' => 'O',
            'ù' | 'ú' | 'û' | 'ü' => 'u',
            'Ù' | 'Ú' | 'Û' | 'Ü' => 'U',
            'ñ' => 'n',
            'Ñ' => 'N',
            'ç' => 'c',
            'Ç' => 'C',
            'ý' | 'ÿ' => 'y',
            '’' | '‘' | 'ʼ' => '\'',
            '‐' | '‑' | '–' => '-',
            other => other,
        })
        .collect()
}

/// A minimal safetensors reader: an 8-byte little-endian header length, a
/// JSON header naming each tensor's dtype, shape and byte range, then the
/// raw data. Only `F32` is accepted, which is all this checkpoint holds.
struct Tensors {
    tensors: HashMap<String, Vec<f32>>,
}

impl Tensors {
    fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let bad = |why: &str| Error::Weights(why.to_owned());
        let header_len = bytes.get(..8).ok_or_else(|| bad("truncated header length"))?;
        let header_len = u64::from_le_bytes(header_len.try_into().map_err(|_| bad("header length"))?) as usize;
        let header = bytes.get(8..8 + header_len).ok_or_else(|| bad("truncated header"))?;
        let data = &bytes[8 + header_len..];
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(header).map_err(|e| Error::Weights(e.to_string()))?;

        let mut tensors = HashMap::new();
        for (name, meta) in header {
            if name == "__metadata__" {
                continue;
            }
            if meta["dtype"] != "F32" {
                return Err(Error::Weights(format!("{name}: dtype {} is not F32", meta["dtype"])));
            }
            let offsets = meta["data_offsets"].as_array().ok_or_else(|| bad("data_offsets"))?;
            let (start, end) = match offsets.as_slice() {
                [s, e] => (s.as_u64().unwrap_or(u64::MAX) as usize, e.as_u64().unwrap_or(0) as usize),
                _ => return Err(bad("data_offsets")),
            };
            let raw = data.get(start..end).filter(|r| r.len() % 4 == 0).ok_or_else(|| bad("tensor range"))?;
            let values = raw.as_chunks::<4>().0.iter().copied().map(f32::from_le_bytes).collect();
            tensors.insert(name, values);
        }
        Ok(Self { tensors })
    }

    fn take(&mut self, name: &str, len: usize) -> Result<Vec<f32>, Error> {
        let values = self.take_any(name)?;
        if values.len() != len {
            return Err(Error::Weights(format!("{name}: {} values, expected {len}", values.len())));
        }
        Ok(values)
    }

    fn take_any(&mut self, name: &str) -> Result<Vec<f32>, Error> {
        self.tensors.remove(name).ok_or_else(|| Error::Weights(format!("missing tensor {name}")))
    }

    fn remaining(&self) -> Option<&str> {
        self.tensors.keys().next().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIALECTS: [Dialect; 2] = [Dialect::American, Dialect::British];

    #[test]
    fn vocabularies_match_each_checkpoint() {
        for dialect in DIALECTS {
            let checkpoint = Checkpoint::for_dialect(dialect);
            let model = NeuralG2p::load(dialect).unwrap();
            // The two alphabets share ids, and the British checkpoint pads its
            // vocabulary beyond both; every character must still have a row.
            let longest = checkpoint.graphemes.chars().count().max(checkpoint.phonemes.chars().count());
            assert!(longest <= model.vocab_size, "{dialect:?}: {longest} > {}", model.vocab_size);
        }
    }

    #[test]
    fn empty_and_unencodable_words_yield_none() {
        let model = NeuralG2p::load(Dialect::American).unwrap();
        assert_eq!(model.phonemize(""), None);
        assert_eq!(model.phonemize("日本"), None);
    }

    #[test]
    fn folding_keeps_accented_words_encodable() {
        let model = NeuralG2p::load(Dialect::American).unwrap();
        assert!(model.phonemize("Zürich").is_some());
    }

    #[test]
    fn overlong_words_are_truncated_not_rejected() {
        let model = NeuralG2p::load(Dialect::American).unwrap();
        assert!(model.phonemize(&"a".repeat(200)).is_some());
    }
}
