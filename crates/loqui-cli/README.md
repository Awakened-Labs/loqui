# loqui-cli

The `loqui` command: serve the OpenAI audio API, speak and transcribe in
process, manage tokens, and check what an exposure would allow.

    cargo install loqui-cli       # needs cmake + a C++ compiler for Whisper
    loqui serve                   # Unix socket, this user only
    loqui serve --listen loopback:8100
    loqui doctor --listen all:8100
    loqui speak "Hello." -o hello.wav
    loqui transcribe hello.wav

`--no-default-features` builds text-to-speech only, with no C++ toolchain.
`--features cuda` runs on an NVIDIA GPU; `--features mp3` adds MP3 output
through LAME (LGPL).

Part of [loqui](https://github.com/Awakened-Labs/loqui). MIT; see `NOTICE`
for third-party attributions.
