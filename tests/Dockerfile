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

# Verifier tooling, pinned. Nothing is installed at trial time. numpy lives here and only
# here — the agent image has no Python numerics stack, which is the point of the task.
RUN pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5 numpy==2.3.2

# test.sh rebuilds and runs the agent's crate, so agent-authored code executes in this
# image — and this image holds the answer key. That code runs as `runner`, never root:
# runner owns the crate and the cargo cache and nothing else.
RUN useradd --create-home --uid 10001 --shell /bin/bash runner

# The crate skeleton mirrors environment/kernel's manifest so the toolchain is warmed now,
# while the build still has a network. The agent's own src/, Cargo.toml and Cargo.lock
# arrive later as artifacts and overwrite these. Keep this in sync with
# environment/kernel/Cargo.{toml,lock}.
COPY skeleton /app/kernel
RUN cd /app/kernel \
    && cargo build --release --offline \
    && rm -rf /app/kernel/src

# Artifact landing directories must exist before Harbor uploads the declared artifacts.
RUN mkdir -p /app/kernel/src /app/data /app/out

# Building as `runner` means cargo writes target/ and the CARGO_HOME package-cache lock;
# neither holds anything secret. /app is runner-owned so the crate can create its output,
# but sticky so runner cannot rename or delete the root-owned export it sits beside.
RUN chown -R runner:runner /app /usr/local/cargo \
    && chmod 1755 /app

# The graded answer key, at a path Harbor does not reserve, readable only by root. /tests is
# a runtime mount — permissions set on it here are both ineffective and a documented cause of
# verifier failure — so the labels cannot be protected there. test_outputs.py reads this copy.
COPY golden /opt/golden
RUN chmod 700 /opt/golden && chmod 600 /opt/golden/*

COPY . /tests/

WORKDIR /app
