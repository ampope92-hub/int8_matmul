#!/bin/bash

# Verifier entrypoint. This runs in the verifier container, not the agent's — it sees
# only the paths declared in `artifacts`. All tooling is baked into tests/Dockerfile;
# nothing is installed or fetched here.

set -uo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

# The graded output must come out of the agent's own crate, so discard whatever is on disk
# and rebuild from the collected sources. A build or run failure is not fatal here — the
# tests report the missing files.
#
# That rebuild executes code the agent wrote, so it runs as the unprivileged `runner`. As
# root it could read the answer key and echo it back, or rewrite test_outputs.py before
# pytest reads it below. Artifacts land owned by root, so hand the crate over first and pin
# the shipped export read-only. Nothing here touches /tests or /logs — Harbor mounts both at
# runtime and modifying them breaks the verifier; the reference is read from the root-only
# copy baked into this image instead (see tests/Dockerfile).
chown -R runner:runner /app/kernel /app/out
chmod -R a-w /app/data

# The shipped tests/ directory also lands here at runtime, so a plaintext copy of the
# reference sits at /tests/golden alongside the graded one. Close it to `runner` — best
# effort, because Harbor owns this mount and may present it read-only. Grading never depends
# on this succeeding (the reference is read from /opt/golden), so a failure here must not
# fail the run; it is logged instead, because it is the difference between a closed and an
# open verifier.
chmod 600 /tests/golden/reference.bin 2>/dev/null || true
if runuser -u runner -- test -r /tests/golden/reference.bin 2>/dev/null; then
  echo "WARNING: /tests/golden/reference.bin is still readable by the unprivileged user"
else
  echo "reference closed to the unprivileged user"
fi

rm -f /app/out/*.bin
runuser -u runner -- bash -c 'cd /app/kernel && cargo run --release --offline'
echo "kernel exit status: $?"

python -m pytest -o cache_dir=/tmp/pytest_cache \
  --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

# Harbor grades from reward.txt, never from this script's exit status.
exit 0
