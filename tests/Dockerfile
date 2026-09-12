# Verifier image. Runs in its own container: it never sees the agent's filesystem, only
# the paths declared in `artifacts`. It rebuilds the agent's crate from source and grades
# what comes out, so it needs the Rust toolchain as well as pytest.

FROM public.ecr.aws/docker/library/rust:1.85-slim@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36 AS rust

FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup

# Verifier tooling, installed from a hash-pinned lockfile so the image cannot pick up a
# different build of any package, transitive ones included. Nothing is installed at trial
# time. numpy lives here and only here — the agent image has no Python numerics stack,
# which is the point of the task.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# test.sh rebuilds and runs the agent's crate, so agent-authored code executes in this
# image — and this image holds the answer key. That code runs as `runner`, never root:
# runner owns the crate and the cargo cache and nothing else.
RUN useradd --create-home --uid 10001 --shell /bin/bash runner

# The crate skeleton mirrors environment/kernel's manifest so the toolchain is warmed now,
# while the build still has a network. The agent's own src/, Cargo.toml and Cargo.lock
# arrive later as artifacts and overwrite these. Keep this in sync with
# environment/kernel/Cargo.{toml,lock}.
COPY skeleton /app/kernel
# The prebuilt skeleton binary must NOT survive into the image. `rm -rf src` alone leaves
# target/release/{pkg} and its fingerprint behind, and cargo decides freshness by mtime:
# when Harbor uploads the agent's sources with an mtime older than this layer, cargo skips
# the rebuild ("Finished" with no "Compiling"), runs the stale `fn main() {}` binary, and
# nothing is written — the graded output stops coming from the agent's code at all. That is
# a silent failure: it looks like the agent produced nothing. `cargo clean -p` drops just
# this crate's artifacts and keeps the compiled dependency tree, which is the whole point
# of the skeleton.
RUN cd /app/kernel \
    && cargo build --release --offline \
    && cargo clean --release -p int8_kernel \
    && rm -rf /app/kernel/src

# Artifact landing directories must exist before Harbor uploads the declared artifacts.
RUN mkdir -p /app/kernel/src /app/data /app/out

# The crate tree and the output directory are runner's; /app itself stays root's, so the build
# user has nowhere directly under it to leave a file for a later run to find. CARGO_HOME stays
# root-owned and is never handed over, so a run cannot write into it at all; the crate tree
# itself is restored from a root-only pristine copy before each of the six runs, which is what
# stops one run leaving state in target/ for a later one to read back.
RUN chown -R runner:runner /app/kernel /app/out \
    && chown root:root /app \
    && chmod 755 /app

# Where the frozen calibration parameters are handed back for a serving run. Root owns it and
# the file inside it is 0444: the serving run reads what calibration committed and cannot
# rewrite it.
RUN mkdir -p /app/scheme && chmod 555 /app/scheme

# The second export of the layer, weights and calibration capture. test.sh swaps this in for
# /app/data and puts the agent's kernel through the same two steps, which is what proves the
# scheme is worked out at run time rather than pasted into the source. /tests is a runtime
# mount — permissions set on it here are both ineffective and a documented cause of verifier
# failure — so nothing withheld can be protected there; these root-only copies are what the
# verifier actually reads.
COPY holdout /opt/holdout
RUN chmod 700 /opt/holdout && chmod 600 /opt/holdout/*

# The serving captures and their double-precision reference products, for both exports. These
# never reach any container the agent can see, and they do not exist as far as the calibration
# step is concerned: the weight side is committed first and only then is the kernel handed one
# of these. That is what stops the graded quantity from being computable while the decisions
# that drive it are still open.
COPY apply /opt/apply
RUN chmod 700 /opt/apply && chmod -R go-rwx /opt/apply

# The shipped export as issued. test_shipped_export_unmodified compares the copy that comes back
# from the agent against these bytes and the shipped calibration run is fed from here, so no
# digest constant exists to fall out of step with the file the agent is given.
COPY shipped /opt/shipped
RUN chmod 700 /opt/shipped && chmod 600 /opt/shipped/*

COPY . /tests/

WORKDIR /app
