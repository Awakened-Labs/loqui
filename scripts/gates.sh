#!/usr/bin/env bash
# The checks a change must pass, in the order CI runs them. CI calls this
# script one gate at a time, so the commands never differ. The toolchain can:
# CI uses the latest stable, whose clippy may know lints an older local one
# does not, and `deny` sees crates yanked since the local index was fetched.
#
#     ./scripts/gates.sh            every gate
#     ./scripts/gates.sh test       one gate: fmt, clippy, test, tts-only, mp3, deny
#
# The workspace build includes Whisper and TLS, because loqui-cli enables
# both by default; that needs cmake and a C++ compiler. `tts-only` is the
# build that promises to need neither. `mp3` is the off-by-default LAME
# build (a C compiler and make), which the workspace gates never reach.
set -euo pipefail
cd "$(dirname "$0")/.."

fmt() { cargo fmt --all --check; }
clippy() { cargo clippy --workspace --all-targets --locked -- -D warnings; }
test() { cargo test --workspace --locked; }
tts-only() { cargo clippy -p loqui-cli --no-default-features --locked -- -D warnings; }
mp3() {
    cargo clippy -p loqui-cli --features mp3 --all-targets --locked -- -D warnings
    cargo test -p loqui-audio --features mp3 --locked
}
deny() { cargo deny check; }

gates=(fmt clippy test tts-only mp3 deny)
if [ $# -gt 0 ]; then
    case " ${gates[*]} " in
        *" $1 "*) gates=("$1") ;;
        *) echo "unknown gate: $1 (one of: ${gates[*]})" >&2; exit 2 ;;
    esac
fi
for gate in "${gates[@]}"; do
    echo "== $gate"
    "$gate"
done
