# Rust toolchain comes from the canonical rust image; the agent builds a Cargo project.
FROM public.ecr.aws/docker/library/rust:1.85-slim@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36 AS rust

# Harbor's harness needs tmux and asciinema, so the final image is the canonical Python
# base with the Rust toolchain copied in.
FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tmux \
        asciinema \
        gcc \
        g++ \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup

# The layer export the kernel consumes, and the Cargo project that consumes it.
COPY data /app/data
COPY kernel /app/kernel

# The crate has no dependencies — the numerics are the point, so there is nothing to
# fetch and no linear-algebra library to fall back on. Compiling the stub now means the
# toolchain is warm and a build failure in the sandbox is the agent's own code.
RUN cd /app/kernel \
    && cargo build --release --offline \
    && rm -rf /app/kernel/target/release/int8_kernel /app/out

WORKDIR /app
