#!/usr/bin/env bash
# Runs loqui-audio's tests on an emulated CPU that has AVX but not FMA: the
# machine CI does not have, since every hosted runner has FMA.
#
# Through 0.1.34, opus-rs ran its `avx,fma` kernels after checking for AVX
# alone, so the first `vfmadd` trapped on Sandy and Ivy Bridge, Bulldozer, or a
# VM that masks FMA: Opus died of SIGILL mid-codec (restsend/opus-rs#30).
# loqui refused Opus on those CPUs until opus-rs 0.1.36 checked for FMA itself
# (restsend/opus-rs#31, issue #7). This gate keeps an upgrade from bringing the
# crash back. It runs the whole test binary, both Opus directions included,
# under `qemu-x86_64 -cpu SandyBridge`, so it covers every dependency that
# picks its SIMD at run time, not only opus-rs.
#
# A probe compiled on the spot checks the emulated CPU first. It must have AVX,
# or opus-rs never enters its AVX paths and the run proves nothing, and must
# not have FMA, or it is not the CPU the bug needs. Otherwise a qemu that
# changed its SandyBridge model would turn this gate green without testing.
#
# Needs qemu-x86_64 (Debian and Ubuntu: qemu-user; 7.2 or later emulates AVX),
# jq, and a libc built for plain x86-64: on a host built with -march=native,
# even the probe traps. Without qemu or such a libc it skips with a notice,
# since a workstation may lack either, except under CI, where a skip would be
# a gate that quietly stopped.
set -euo pipefail
cd "$(dirname "$0")/.."

# Less the two features qemu's TCG cannot emulate, which would otherwise earn
# a warning per thread. Neither matters in user mode.
cpu=SandyBridge,-x2apic,-tsc-deadline

if ! command -v qemu-x86_64 >/dev/null; then
    if [ -n "${CI:-}" ]; then
        echo "test-without-fma.sh: qemu-x86_64 is not on PATH; CI installs qemu-user" >&2
        exit 1
    fi
    echo "test-without-fma.sh: skipped, qemu-x86_64 is not on PATH (install qemu-user)"
    exit 0
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cat >"$work/probe.rs" <<'EOF'
fn main() {
    let (avx, fma) = (std::arch::is_x86_feature_detected!("avx"), std::arch::is_x86_feature_detected!("fma"));
    println!("avx={avx} fma={fma}");
    std::process::exit(if avx && !fma { 0 } else { 1 });
}
EOF
rustc --edition 2024 -O -o "$work/probe" "$work/probe.rs"
status=0
probe=$(qemu-x86_64 -cpu "$cpu" "$work/probe") || status=$?
if [ "$status" -gt 128 ]; then
    # The probe never reached main: the libc it links was built for a newer
    # CPU (Gentoo's -march=native, say), so nothing built here runs on this
    # one. CI's libc targets plain x86-64.
    echo "test-without-fma.sh: the probe died of signal $((status - 128)) under $cpu; this host's libc needs a newer CPU" >&2
    if [ -n "${CI:-}" ]; then
        exit 1
    fi
    echo "test-without-fma.sh: skipped, run it in CI or a container with a generic libc"
    exit 0
elif [ "$status" -ne 0 ]; then
    echo "test-without-fma.sh: qemu's $cpu reports $probe; this gate needs AVX without FMA" >&2
    exit 1
fi
echo "qemu -cpu $cpu: $probe"

# The lib test binary, from cargo's own report of what it built.
exe=$(cargo test -p loqui-audio --locked --no-run --message-format=json |
    jq -r 'select(.reason == "compiler-artifact" and .target.name == "loqui_audio"
                  and (.target.kind | index("lib")) and .profile.test) | .executable')
if [ ! -x "$exe" ]; then
    echo "test-without-fma.sh: cargo reported no loqui-audio lib test binary" >&2
    exit 1
fi

# From the crate's directory, as `cargo test` runs it.
echo "qemu-x86_64 -cpu $cpu ${exe#"$PWD"/}"
(cd crates/loqui-audio && qemu-x86_64 -cpu "$cpu" "$exe") | tee "$work/out"

# A binary that runs no tests passes too, so require that some did.
passed=$(sed -n 's/^test result: ok\. \([0-9]*\) passed.*/\1/p' "$work/out")
if [ -z "$passed" ] || [ "$passed" -eq 0 ]; then
    echo "test-without-fma.sh: no test ran under $cpu" >&2
    exit 1
fi
