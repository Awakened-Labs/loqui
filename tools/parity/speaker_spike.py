#!/usr/bin/env python3
"""Can a speaker-verification model steer a search over Kokoro blends?

    speaker_spike.py synth KOKORO.onnx VOICES_DIR VOCAB.json PHONEMES_US PHONEMES_GB OUT_DIR
    speaker_spike.py score SPEAKER.onnx OUT_DIR [TARGET.wav ...]
    speaker_spike.py search SPEAKER.onnx KOKORO.onnx VOICES_DIR VOCAB.json PHONEMES_US PHONEMES_GB OUT_DIR K BUDGET

`synth` speaks seven Harvard sentences, given as phonemes one per line
(`phonemize` and `phonemize --gb` over the same lines), with Kokoro's ONNX
export on Python's onnxruntime, blending packs in numpy as loqui does:
- every stock voice, sentences 1-5 (the anchors);
- twelve voice pairs, each at 25/50/75% of its first voice, sentences 3-4
  (the targets);
- the same pairs at 0, 10, ..., 100%, on sentences 3-4 (same text as the
  targets) and on sentences 6-7 (different text).
A blend speaks in its first voice's dialect, even at weight 0. Audio is kept
as float32 `.npy` at 24 kHz in OUT_DIR, so `score` can run once per model.

`score` embeds everything with a speaker model exported for sherpa-onnx
(WeSpeaker's 80-dim Kaldi fbank at 16 kHz; `feats [B,T,80]` in,
`embs [B,D]` out), then reports:
- M1: leave-one-sentence-out nearest-centroid accuracy over the stock voices;
- M2: mean margin between the true voice and the best other voice;
- M3: whether a 50/50 blend's nearest anchor is one of its parents, and
  whether the similarity to each parent moves monotonically with the weight;
- M4: whether a grid over the weight finds the target's weight within 15,
  with candidates on the target's text and on other text (M6);
- linear: how well `normalize(w a + (1-w) b)` of the anchors predicts a
  blend's embedding (the search's starting point);
- timing: synthesis real-time factor and embedding cost.
Any TARGET.wav is ranked against the anchors (top five voices and cosines).

`search` runs the matcher's algorithm against each pair's 50% and 75% target,
synthesizing candidates on sentences 6-7 (text the target never said): rank
the stock voices against the target, start from the least-squares mix of the
top K anchors on a 5% lattice, then move weight between pairs of voices
(20, 10, then 5 points at a time) while the similarity improves, within BUDGET
syntheses. It reports what the search reached beside two references: the best
single voice, and the true blend spoken on the same text (the ceiling).
"""

import itertools
import json
import os
import sys
import time

import numpy as np

VOICES = (
    "af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky "
    "am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa "
    "bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis"
).split()
PAIRS = [
    ("af_bella", "af_sky"), ("af_heart", "af_nicole"), ("am_michael", "am_onyx"),
    ("am_adam", "am_puck"), ("bf_emma", "bf_isabella"), ("bm_george", "bm_lewis"),
    ("af_sarah", "bf_emma"), ("am_eric", "bm_daniel"), ("af_heart", "am_michael"),
    ("bf_alice", "bm_fable"), ("af_kore", "am_liam"), ("af_aoede", "bm_george"),
]
ANCHOR_LINES = [0, 1, 2, 3, 4]
TARGET_LINES = [2, 3]
PROBE_LINES = [5, 6]
TARGET_WEIGHTS = [25, 50, 75]
GRID = list(range(0, 101, 10))
RATE = 24000


class Kokoro:
    """Kokoro's ONNX export with packs mixed in numpy, as loqui mixes them."""

    def __init__(self, model, voices_dir, vocab_path, us_path, gb_path):
        import onnxruntime as ort

        self.vocab = json.load(open(vocab_path))["vocab"]
        self.phonemes = {"a": open(us_path).read().splitlines(), "b": open(gb_path).read().splitlines()}
        self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.packs = {
            v: np.fromfile(os.path.join(voices_dir, f"{v}.bin"), dtype="<f4").reshape(510, 256) for v in VOICES
        }
        self.spent, self.audio_secs = 0.0, 0.0

    def speak(self, parts, line):
        """`parts` is [(voice, weight)], the first setting the dialect."""
        total = sum(w for _, w in parts)
        pack = sum((w / total) * self.packs[v] for v, w in parts if w > 0)
        ps = self.phonemes[parts[0][0][0]][line]
        ids = [0] + [self.vocab[c] for c in ps if c in self.vocab] + [0]
        started = time.perf_counter()
        wave = self.sess.run(["waveform"], {
            "input_ids": np.array([ids], dtype=np.int64),
            "style": pack[len(ps) - 1][None, :].astype(np.float32),
            "speed": np.array([1.0], dtype=np.float32),
        })[0][0]
        self.spent += time.perf_counter() - started
        self.audio_secs += len(wave) / RATE
        return wave.astype(np.float32)


def synth(model, voices_dir, vocab_path, us_path, gb_path, out):
    kokoro = Kokoro(model, voices_dir, vocab_path, us_path, gb_path)
    os.makedirs(out, exist_ok=True)

    def speak(name, parts, line):
        path = os.path.join(out, f"{name}.{line}.npy")
        if not os.path.exists(path):
            np.save(path, kokoro.speak(parts, line))

    for v in VOICES:
        for line in ANCHOR_LINES:
            speak(v, [(v, 1)], line)
    for a, b in PAIRS:
        for w in TARGET_WEIGHTS:
            for line in TARGET_LINES:
                speak(f"target-{a}-{b}-{w}", [(a, w), (b, 100 - w)], line)
        for w in GRID:
            for line in TARGET_LINES + PROBE_LINES:
                speak(f"grid-{a}-{b}-{w}", [(a, w), (b, 100 - w)], line)
    if kokoro.audio_secs:
        print(f"synthesized {kokoro.audio_secs:.0f}s of audio in {kokoro.spent:.0f}s "
              f"(real-time factor {kokoro.spent / kokoro.audio_secs:.3f})")


class Embedder:
    def __init__(self, model):
        import kaldi_native_fbank as knf
        import onnxruntime as ort

        self.knf = knf
        self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        assert meta.get("sample_rate") == "16000", meta
        self.scale = 1.0 if meta.get("normalize_samples", "1") == "1" else 32768.0
        self.spent, self.audio_secs = 0.0, 0.0

    def fbank(self, samples16k):
        opts = self.knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.snip_edges = True
        opts.frame_opts.window_type = "hamming"
        opts.mel_opts.num_bins = 80
        fb = self.knf.OnlineFbank(opts)
        fb.accept_waveform(16000, (samples16k * self.scale).tolist())
        fb.input_finished()
        feats = np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], dtype=np.float32)
        return feats - feats.mean(axis=0, keepdims=True)

    def embed(self, wave, rate=RATE):
        from scipy.signal import resample_poly

        started = time.perf_counter()
        wave = trim(wave)
        samples = resample_poly(wave, 16000 // np.gcd(16000, rate), rate // np.gcd(16000, rate)).astype(np.float32)
        feats = self.fbank(samples)
        emb = self.sess.run(["embs"], {"feats": feats[None]})[0][0]
        self.spent += time.perf_counter() - started
        self.audio_secs += len(wave) / rate
        return emb / np.linalg.norm(emb)


def trim(wave, threshold=0.01, frame=480):
    """Drops leading and trailing frames quieter than 1% of the peak."""
    peak = np.abs(wave).max() or 1.0
    frames = [np.abs(wave[i:i + frame]).max() for i in range(0, len(wave), frame)]
    loud = [i for i, f in enumerate(frames) if f > threshold * peak]
    if not loud:
        return wave
    return wave[loud[0] * frame:(loud[-1] + 1) * frame]


def unit(v):
    return v / np.linalg.norm(v)


def score(model, out, targets):
    emb = Embedder(model)
    cache = {}

    def e(name, lines):
        """Mean embedding over the given lines, renormalized."""
        for line in lines:
            if (name, line) not in cache:
                cache[name, line] = emb.embed(np.load(os.path.join(out, f"{name}.{line}.npy")))
        return unit(sum(cache[name, line] for line in lines))

    print(f"model {os.path.basename(model)}")
    # M1 / M2: leave one sentence out.
    hits, margins = 0, []
    for v in VOICES:
        for held in ANCHOR_LINES:
            rest = [l for l in ANCHOR_LINES if l != held]
            x = e(v, [held])
            sims = {u: float(x @ e(u, rest)) for u in VOICES}
            best_other = max(s for u, s in sims.items() if u != v)
            hits += max(sims, key=sims.get) == v
            margins.append(sims[v] - best_other)
    n = len(VOICES) * len(ANCHOR_LINES)
    print(f"M1 nearest-centroid accuracy: {hits}/{n} = {hits / n:.1%}")
    print(f"M2 margin: mean {np.mean(margins):.3f}, min {np.min(margins):.3f}")
    anchors = {v: e(v, ANCHOR_LINES) for v in VOICES}
    same_voice = np.mean([float(e(v, [l]) @ anchors[v]) for v in VOICES for l in ANCHOR_LINES])
    print(f"   mean same-voice cosine {same_voice:.3f}")

    # M3: betweenness.
    mid_ok, mono_ok = 0, 0
    for a, b in PAIRS:
        x = e(f"target-{a}-{b}-50", TARGET_LINES)
        nearest = max(VOICES, key=lambda u: float(x @ anchors[u]))
        mid_ok += nearest in (a, b)
        to_a = [float(e(f"grid-{a}-{b}-{w}", TARGET_LINES) @ anchors[a]) for w in GRID]
        to_b = [float(e(f"grid-{a}-{b}-{w}", TARGET_LINES) @ anchors[b]) for w in GRID]
        mono_ok += all(np.diff(to_a) > -0.02) and all(np.diff(to_b) < 0.02)
    print(f"M3 50/50 nearest anchor is a parent: {mid_ok}/{len(PAIRS)}; monotone (tolerance 0.02): {mono_ok}/{len(PAIRS)}")

    # M4 / M6: recover the weight by grid search.
    for label, lines in (("same text", TARGET_LINES), ("other text", PROBE_LINES)):
        ok, errs, gaps = 0, [], []
        for a, b in PAIRS:
            for w in TARGET_WEIGHTS:
                t = e(f"target-{a}-{b}-{w}", TARGET_LINES)
                sims = {g: float(t @ e(f"grid-{a}-{b}-{g}", lines)) for g in GRID}
                found = max(sims, key=sims.get)
                ok += abs(found - w) <= 15
                errs.append(abs(found - w))
                gaps.append(sims[found] - min(sims[0], sims[100]))
        m = len(PAIRS) * len(TARGET_WEIGHTS)
        print(f"M4 ({label}): weight within 15: {ok}/{m} = {ok / m:.1%}; mean error {np.mean(errs):.1f}; "
              f"mean gain over the worse pure parent {np.mean(gaps):.3f}")

    # The linear model the search starts from.
    lin = []
    for a, b in PAIRS:
        for w in TARGET_WEIGHTS:
            pred = unit(w / 100 * anchors[a] + (1 - w / 100) * anchors[b])
            lin.append(float(pred @ e(f"target-{a}-{b}-{w}", TARGET_LINES)))
    print(f"linear prediction cosine: mean {np.mean(lin):.3f}, min {np.min(lin):.3f}")
    print(f"embedding cost: {emb.spent / emb.audio_secs * 1000:.1f} ms per second of audio")

    for path in targets:
        import soundfile as sf

        wave, rate = sf.read(path, dtype="float32", always_2d=True)
        x = emb.embed(wave.mean(axis=1), rate)
        ranked = sorted(VOICES, key=lambda u: -float(x @ anchors[u]))[:5]
        print(f"{path}: " + ", ".join(f"{u} {float(x @ anchors[u]):.3f}" for u in ranked))


def simplex_project(v):
    """Euclidean projection onto {w >= 0, sum w = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1), 0)


def lattice(weights, step=5):
    """Rounds weights summing to one onto integer percents in `step`s, by
    largest remainder, so they sum to exactly 100."""
    units = 100 // step
    raw = np.asarray(weights) * units
    out = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - out))[: units - out.sum()]:
        out[i] += 1
    return [int(x) * step for x in out]


def search(model, kokoro_model, voices_dir, vocab, us, gb, out, k, budget):
    emb = Embedder(model)
    kokoro = Kokoro(kokoro_model, voices_dir, vocab, us, gb)
    k, budget = int(k), int(budget)
    cache = {}

    def e(name, lines):
        for line in lines:
            if (name, line) not in cache:
                cache[name, line] = emb.embed(np.load(os.path.join(out, f"{name}.{line}.npy")))
        return unit(sum(cache[name, line] for line in lines))

    anchors = {v: e(v, ANCHOR_LINES) for v in VOICES}
    rows = []
    started = time.perf_counter()
    for a, b in PAIRS:
        for w in (50, 75):
            t = e(f"target-{a}-{b}-{w}", TARGET_LINES)
            ranked = sorted(VOICES, key=lambda u: -float(t @ anchors[u]))
            top = ranked[:k]
            # Least squares on the simplex: projected gradient, fixed steps.
            A = np.stack([anchors[v] for v in top], axis=1)
            x = np.full(k, 1 / k)
            lr = 1 / (np.linalg.norm(A, 2) ** 2)
            for _ in range(200):
                x = simplex_project(x - lr * A.T @ (A @ x - t))
            seen = {}

            def score_of(weights):
                key = tuple(weights)
                if key not in seen:
                    parts = [(v, wt) for v, wt in zip(top, weights)]
                    lead = top[int(np.argmax(weights))]
                    parts.sort(key=lambda p: (p[0] != lead, -p[1]))
                    wave = np.concatenate([kokoro.speak(parts, line) for line in PROBE_LINES])
                    seen[key] = float(t @ emb.embed(wave))
                return seen[key]

            single = [100] + [0] * (k - 1)
            best = lattice(x)
            best_s = score_of(best)
            single_s = score_of(single)
            if single_s > best_s:
                best, best_s = single, single_s
            for step in (20, 10, 5):
                improved = True
                while improved and len(seen) < budget:
                    improved = False
                    for i in range(k):
                        for j in range(k):
                            if i == j or best[j] == 0 or len(seen) >= budget:
                                continue
                            cand = list(best)
                            moved = min(step, cand[j])
                            cand[i] += moved
                            cand[j] -= moved
                            if tuple(cand) in seen:
                                continue
                            s = score_of(cand)
                            if s > best_s + 0.001:
                                best, best_s, improved = cand, s, True
                                break
                        if improved:
                            break
            truth = float(t @ emb.embed(np.concatenate(
                [kokoro.speak([(a, w), (b, 100 - w)], line) for line in PROBE_LINES])))
            spec = "+".join(f"{v}({wt})" for v, wt in sorted(zip(top, best), key=lambda p: -p[1]) if wt)
            rows.append((best_s, single_s, truth, len(seen)))
            print(f"{a}({w})+{b}({100 - w}): found {best_s:.3f} [{spec}] in {len(seen)}; "
                  f"single {single_s:.3f}; true blend {truth:.3f}")
    found, single, truth, evals = map(np.array, zip(*rows))
    print(f"K={k} budget={budget}: mean found {found.mean():.3f}, single {single.mean():.3f}, "
          f"ceiling {truth.mean():.3f}; found >= ceiling - 0.02 in {(found >= truth - 0.02).sum()}/{len(rows)}; "
          f"mean evaluations {evals.mean():.1f}")
    print(f"kokoro real-time factor {kokoro.spent / kokoro.audio_secs:.3f}; "
          f"{(time.perf_counter() - started) / len(rows):.1f}s per match")


def main():
    if sys.argv[1] == "synth":
        synth(*sys.argv[2:8])
    elif sys.argv[1] == "search":
        search(*sys.argv[2:11])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4:])


if __name__ == "__main__":
    main()
