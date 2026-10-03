#!/usr/bin/env bash
# The checks a change must pass, in the order CI runs them. CI calls this
# script one gate at a time, so the commands never differ. The toolchain can:
# CI uses the latest stable, whose clippy may know lints an older local one
# does not, and `deny` sees crates yanked since the local index was fetched.
#
#     ./scripts/gates.sh            every gate
#     ./scripts/gates.sh test       one gate: fmt, clippy, test, tts-only, mp3, deny, package
#
# The workspace build includes Whisper and TLS, because loqui-cli enables
# both by default; that needs cmake and a C++ compiler. `tts-only` is the
# build that promises to need neither. `mp3` is the off-by-default LAME
# build (a C compiler and make), which the workspace gates never reach.
# `package` builds every crate from what crates.io would receive, and keeps
# each under its 10 MiB limit (loqui-g2p embeds ~9 MiB of lexicons and weights).
set -euo pipefail
cd "$(dirname "$0")/.."

fmt() { cargo fmt --all --check; }
clippy() {
    cargo clippy --workspace --all-targets --locked -- -D warnings
    # Nothing in the workspace enables `unguarded-opus`; lint what it selects.
    cargo clippy -p loqui-audio --features unguarded-opus --all-targets --locked -- -D warnings
}
test() { cargo test --workspace --locked; }
tts-only() { cargo clippy -p loqui-cli --no-default-features --locked -- -D warnings; }
mp3() {
    cargo clippy -p loqui-cli --features mp3 --all-targets --locked -- -D warnings
    cargo test -p loqui-audio --features mp3 --locked
}
deny() { cargo deny check; }
package() {
    cargo package --workspace --locked
    local limit=$((10 * 1024 * 1024)) crate size
    for crate in target/package/*.crate; do
        size=$(stat -c %s "$crate")
        echo "$(basename "$crate"): $size bytes"
        if [ "$size" -ge "$limit" ]; then
            echo "$crate is over crates.io's 10 MiB limit" >&2
            return 1
        fi
    done
}

gates=(fmt clippy test tts-only mp3 deny package)
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
