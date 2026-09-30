#!/usr/bin/env bash
# The checks a change must pass, in the order CI runs them. CI calls this
# script one gate at a time, so what passes here passes there.
#
#     ./scripts/gates.sh            every gate
#     ./scripts/gates.sh test       one gate: fmt, clippy, test, tts-only, deny
#
# The workspace build includes Whisper and TLS, because loqui-cli enables
# both by default; that needs cmake and a C++ compiler. `tts-only` is the
# build that promises to need neither.
set -euo pipefail
cd "$(dirname "$0")/.."

fmt() { cargo fmt --all --check; }
clippy() { cargo clippy --workspace --all-targets --locked -- -D warnings; }
test() { cargo test --workspace --locked; }
tts-only() { cargo clippy -p loqui-cli --no-default-features --locked -- -D warnings; }
deny() { cargo deny check; }

gates=(fmt clippy test tts-only deny)
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
