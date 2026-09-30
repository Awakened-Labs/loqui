# syntax=docker/dockerfile:1
#
# loqui with CUDA: Kokoro through ONNX Runtime's CUDA provider, Whisper
# through whisper.cpp's CUDA backend.
#
#     docker build -f docker/cuda.Dockerfile -t loqui:cuda .
#     docker run --gpus all --network host -v ~/.cache/loqui:/root/.cache/loqui \
#         loqui:cuda serve --listen loopback:8100 --gpu cuda
#
# Host networking keeps the listener on the host's own loopback, so loqui's
# exposure rules mean what they say; publishing a port would need all:PORT.
#
# The `toolchain` stage doubles as a CUDA dev shell, for hosts whose own
# toolkit is missing or is not CUDA 13:
#
#     docker build -f docker/cuda.Dockerfile --target toolchain -t loqui:cuda-dev .
#     docker run --rm --gpus all -v "$PWD":/src:ro -v loqui-cuda13-target:/target \
#         -v loqui-cuda-registry:/usr/local/cargo/registry loqui:cuda-dev \
#         cargo build --release --locked -p loqui-whisper --features cuda --example transcribe
#
# CUDA 13, because the ONNX Runtime that `ort` downloads (1.28) ships its
# CUDA provider for 13 with cuDNN 9 only. CUDA_ARCHS picks the GPU
# generations whisper.cpp compiles kernels for; the default covers Turing
# (RTX 20xx) through Ada (RTX 40xx), and CUDA 13 supports nothing older.

ARG CUDA=13.0.2
ARG UBUNTU=ubuntu24.04

FROM nvidia/cuda:${CUDA}-cudnn-devel-${UBUNTU} AS toolchain
ARG RUST=1.92
ARG CUDA_ARCHS="75;80;86;89"
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential cmake clang libclang-dev pkg-config git \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --component clippy,rustfmt \
        --default-toolchain ${RUST} --no-modify-path
ENV CUDAARCHS=${CUDA_ARCHS} CARGO_TARGET_DIR=/target
WORKDIR /src

FROM toolchain AS build
ARG CUDA
COPY . .
# ONNX Runtime's CUDA provider libraries are symlinked into the target
# directory from ort's download cache, hence the cache mount and `cp -L`.
# The target cache is keyed by CUDA version: cargo cannot see the toolkit
# change, and whisper.cpp objects built against one do not link with another.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/target,id=loqui-target-cuda-${CUDA} \
    cargo build --release --locked -p loqui-cli --features cuda \
    && mkdir -p /out \
    && cp /target/release/loqui /out/ \
    && cp -L /target/release/libonnxruntime_providers_*.so /out/

FROM nvidia/cuda:${CUDA}-cudnn-runtime-${UBUNTU} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /opt/loqui/
ENV PATH=/opt/loqui:$PATH
ENTRYPOINT ["/opt/loqui/loqui"]
CMD ["--help"]
