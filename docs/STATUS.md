# Status: 2026-10-02

## Milestones

| Milestone | State | Evidence |
|---|---|---|
| M0 G2P spike | passed | Harvard 0.01% PER, assistant 0.72%, Gutenberg 0.78% vs Python misaki; OOV fallback beats eSpeak vs CMUdict (16.8% vs 17.5% PER) |
| M0 synthesis spike | passed | E2E WER 2.42% vs the open-speech container's 2.28% (gate +0.5); ONNX output correlates 0.997 with Python ORT, 0.986 with PyTorch |
| M1 skeleton | done | workspace, lints, LICENSE/NOTICE/README; CI (fmt, clippy, test, TTS-only build, MSRV 1.92, `cargo deny`) |
| M2 loqui-audio | done | out: WAV, FLAC, PCM, Ogg Opus, and MP3 behind the off-by-default `mp3` feature (LAME, LGPL). ffmpeg decodes both compressed formats to the exact sample count; Whisper transcribes Kokoro speech from each word for word; in: WAV, FLAC, MP3, AAC, Ogg Vorbis, and Opus in Ogg or WebM (opus-rs; sample counts match libopus). |
| M3 g2p + kokoro + models | done | pinned revisions + SHA-256 for every model file |
| M4 whisper + facade | done | `Engine`, lazy/TTL `Slot`, offline mode |
| M5 server + CLI | done | 16 security integration tests; live-tested with an existing OpenAI-compatible client, unchanged |
| M6 parity + cut-over | done | STT WER 2.19% vs faster-whisper 2.28% on the same 250 clips; the `cuda` image serves both models on the GPU (speech at real-time factor 0.09, transcription 0.16) in 2.7 GB of GPU memory |
| M7 client integration | done over HTTP | an existing OpenAI-compatible speech client, unchanged, speaks and transcribes through `loqui serve` |
| M8 publish | done | 0.1.0 on crates.io; public repository; SECURITY.md (private reporting); docs/embedding.md |
| Custom voices | done, unreleased (0.2.0) | named voices from `voices.toml`; `GET /v1/audio/voices` and `loqui voices`; blends with open-speech's grammar and a cache bounded by the voice packs (see `tools/parity/README.md` for blend parity) |

Run `./scripts/gates.sh` before pushing; CI runs the same script, one gate
per job. `--all-features` needs the CUDA and Vulkan toolkits; the `cuda`
build lives in `docker/cuda.Dockerfile`.

GPU builds need **CUDA 13**: the ONNX Runtime that `ort` rc.13 downloads
(1.28) ships its CUDA provider for 13 only. A CUDA 12 toolkit builds
Whisper but cannot run Kokoro on the GPU. The image is 3.3 GB (open-speech:
9.1 GB).

## Known gaps and deliberate differences

- Dependencies: MPL-2.0 is allowed for symphonia (per-file copyleft,
  unmodified). One advisory is ignored in `deny.toml`: `number_prefix`
  (unmaintained) arrives through hf-hub 0.4's progress bar and goes away
  with hf-hub 1.x, an API migration not yet done.
- Opus input is refused on CPUs with AVX but no FMA (Sandy/Ivy Bridge):
  opus-rs crashes there (restsend/opus-rs#30). Drop the guard in
  `loqui-audio/src/opus.rs` once a release carries #31.
- Opus output is 64 kbps CELT, encoded from 48 kHz, to route around two
  opus-rs 0.1.34 encoder bugs: 24 kHz input comes out as garbage in every
  mode (restsend/opus-rs#37), and below 64 kbps the SILK/hybrid paths mangle
  the first ~200 ms and run up to 17 samples behind the pre-skip
  (restsend/opus-rs#38). The encode test asserts correlation > 0.99, which
  either bug fails. Once both are fixed, encode from 24 kHz at 24-32 kbps,
  which is plenty for speech.
- WebM/Opus keeps up to 13.5 ms of trailing padding: symphonia 0.6 parses
  `DiscardPadding` but does not pass it on. Ogg is trimmed exactly.
- CI does not build the `cuda` feature (no toolkit on hosted runners); the
  image build is the check for that path.
- Time expressions: "3:00" reads "three zero zero", as upstream misaki does.
  Reading clock times properly would be a deliberate improvement.
- "St." reads as letters, not "Saint" (eSpeak's abbreviation table).
- loqui does not reproduce two upstream bugs: dropped negative numbers
  ("-13 degrees"), garbled euro amounts.
- `/v1/audio/speech` defaults to MP3 when `response_format` is omitted, as
  OpenAI does, but only in builds with `mp3`; others default to WAV. The
  `cuda` image is built without `mp3`.
- MP3 is behind a feature because LAME is LGPL (the bindings LGPL-3.0, LAME
  itself LGPL-2.0-or-later), statically linked. Running `loqui serve` as a
  separate process keeps it away from its clients; a proprietary program
  linking loqui in process should leave `mp3` off and use Opus or WAV (see
  `docs/embedding.md`). `cargo deny` allows the LGPL for those two crates
  only, and the `mp3` CI gate builds and tests it.
- Upstream bug to report: flacenc 0.5.1 writes the final block into
  STREAMINFO's minimum block size (worked around in `loqui-audio`).
