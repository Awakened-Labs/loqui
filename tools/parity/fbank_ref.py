#!/usr/bin/env python3
"""The reference loqui-speaker's filterbank is held to.

    fbank_ref.py OUT_DIR [SPEAKER.onnx]

Writes, for `crates/loqui-speaker/tests/fbank_parity.rs`:
- `fbank-input.i16`: 1.5 s of 16 kHz mono, little-endian i16 — two tones, a
  chirp and LCG noise, generated here so the fixture has no source to cite;
- `fbank-expected.f32`: kaldi-native-fbank's log mel energies for it,
  little-endian f32, frames x 80, BEFORE mean subtraction;
- with SPEAKER.onnx, `embedding-expected.f32`: that model's unit-length
  embedding of the same features after mean subtraction, run on Python's
  onnxruntime (also needs `onnxruntime`).

kaldi-native-fbank is what sherpa-onnx computes these models' features with,
set as WeSpeaker's `compute_fbank` sets torchaudio's Kaldi fbank: 80 bins,
25/10 ms, dither 0, Hamming window, edges snipped, no energy, samples on the
16-bit scale. Needs `numpy` and `kaldi-native-fbank`.
"""

import os
import sys

import numpy as np


def signal():
    rate, n = 16000, 24000
    t = np.arange(n) / rate
    x = 0.30 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 1870 * t)
    x += 0.20 * np.sin(2 * np.pi * (300 * t + 1500 * t * t))  # 300 Hz -> 4.8 kHz
    state, noise = 12345, np.empty(n)
    for i in range(n):
        state = (1103515245 * state + 12345) % 2**31
        noise[i] = state / 2**31 - 0.5
    x += 0.05 * noise
    return np.round(np.clip(x, -1, 1) * 32767).astype("<i2")


def main():
    import kaldi_native_fbank as knf

    out = sys.argv[1]
    pcm = signal()
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.snip_edges = True
    opts.frame_opts.window_type = "hamming"
    opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(16000, pcm.astype(np.float32).tolist())
    fb.input_finished()
    feats = np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], dtype="<f4")
    os.makedirs(out, exist_ok=True)
    pcm.tofile(os.path.join(out, "fbank-input.i16"))
    feats.tofile(os.path.join(out, "fbank-expected.f32"))
    print(f"{len(pcm)} samples, {feats.shape[0]} frames x {feats.shape[1]} bins")
    if len(sys.argv) > 2:
        import onnxruntime as ort

        sess = ort.InferenceSession(sys.argv[2], providers=["CPUExecutionProvider"])
        cmn = feats - feats.mean(axis=0, keepdims=True)
        emb = sess.run(["embs"], {"feats": cmn[None].astype(np.float32)})[0][0]
        (emb / np.linalg.norm(emb)).astype("<f4").tofile(os.path.join(out, "embedding-expected.f32"))
        print(f"embedding of {emb.shape[0]} values")


if __name__ == "__main__":
    main()
