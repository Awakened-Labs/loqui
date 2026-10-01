#!/usr/bin/env python3
"""Reference Kokoro audio for one phoneme string, run inside the container.

    kokoro_ref.py onnx  MODEL.onnx VOICE.bin "PHONEMES" OUT.f32 [SPEED]
    kokoro_ref.py torch VOICE_ID          "PHONEMES" OUT.f32 [SPEED]

`onnx` runs the same ONNX export loqui uses, on Python's onnxruntime, so the
two should agree to floating-point noise. `torch` runs hexgrad's PyTorch
KModel (what open-speech serves); its iSTFTNet source adds random noise, so
it is compared by correlation rather than sample by sample. Output is raw
little-endian float32 at 24 kHz.
"""

import json
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np


def vocab():
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download("hexgrad/Kokoro-82M", "config.json")) as f:
        return json.load(f)["vocab"]


def main():
    mode = sys.argv[1]
    if mode == "onnx":
        import onnxruntime as ort

        model, voice, ps, out = sys.argv[2:6]
        speed = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
        v = vocab()
        ids = [0] + [v[c] for c in ps if c in v] + [0]
        pack = np.fromfile(voice, dtype="<f4").reshape(510, 256)
        style = pack[len(ps) - 1][None, :]
        sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        audio = sess.run(["waveform"], {
            "input_ids": np.array([ids], dtype=np.int64),
            "style": style.astype(np.float32),
            "speed": np.array([speed], dtype=np.float32),
        })[0][0]
    else:
        import torch
        from kokoro import KModel, KPipeline

        voice_id, ps, out = sys.argv[2:5]
        speed = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
        pipeline = KPipeline(lang_code=voice_id[0], repo_id="hexgrad/Kokoro-82M", model=False)
        pack = pipeline.load_voice(voice_id)
        torch.manual_seed(0)
        with torch.no_grad():
            audio = model(ps, pack[len(ps) - 1], speed).numpy()
    np.asarray(audio, dtype="<f4").tofile(out)


if __name__ == "__main__":
    main()
