#!/bin/bash

# Verifier entrypoint. This runs in the verifier container, not the agent's — it sees only the
# paths declared in `artifacts`. All tooling is baked into tests/Dockerfile; nothing is
# installed or fetched here.

set -uo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

# Each run below is started under `setsid`, so it becomes a session leader and its process
# group id is its own pid. That is the id used to tear the group down. Reading the group out of
# /proc instead is a race: there is a window after the fork and before setsid(2) lands in which
# the child still reports THIS shell's group, and signalling that group kills the verifier
# itself -- the log stops mid-script, pytest never runs, no ctrf.json is written and the reward
# stays 0 on a correct solution. Two trials were scored 0 that way. The pid is known without
# reading anything, so nothing below reads /proc.

chown -R runner:runner /app/kernel /app/out 2>/dev/null || true
# /app itself is root's. The crate tree and the output directory under it are the only places
# the build user can write, and both are replaced before every run; a run that could create a
# file directly under /app would have somewhere to leave the weights for a later run to find.
chown root:root /app 2>/dev/null || true
chmod 755 /app 2>/dev/null || true

# Everything that touches agent-authored code runs through this helper: the build user, with
# supplementary groups cleared and no-new-privileges set, so an inherited group or a setuid
# binary cannot widen what that code can reach.
as_runner() {
  setpriv --clear-groups --no-new-privs --reuid runner --regid runner "$@"
}

# The shipped tests/ directory lands here at runtime, so the verifier's own code and the
# reference products sit under /tests just as agent-authored code is about to run. Close the
# whole directory to the build user before anything untrusted starts: root still reads it,
# `runner` cannot even traverse it. If it is still reachable the run is abandoned with reward 0
# rather than graded -- a reachable /tests is one the agent can read the references out of, or
# whose test file it can rewrite before pytest reads it. The graded references are read from the
# root-only copies at /opt/apply either way.
chmod 700 /tests 2>/dev/null || true
# /logs/verifier is writable by the account that runs agent-authored code unless it is closed
# here too. Root's final write of reward.txt does win, so this was a hardening gap rather than a
# live exploit, but a run that can create files beside the reward has no business doing so.
chmod 700 /logs/verifier 2>/dev/null || true
if as_runner bash -c 'test -r /tests/test_outputs.py || test -r /tests/apply/shipped/batch0_ref.bin' \
     2>/dev/null; then
  echo "FATAL: /tests is still reachable by the unprivileged build user; the verifier's code"
  echo "and reference products are exposed to agent-authored code, so this run cannot be graded."
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi
echo "verifier files closed to the unprivileged build user"

# Keep the export exactly as the agent left it. It is put back at the end so
# test_shipped_export_unmodified still checks what the agent actually did. The runs themselves
# are fed from the root-only copy of the export as issued at /opt/shipped, so what is graded is
# the export we shipped, and an altered copy fails the byte comparison rather than steering the
# calibration.
rm -rf /opt/agent_data
cp -a /app/data /opt/agent_data
chown -R root:root /opt/agent_data
chmod -R go-rwx /opt/agent_data

# A pristine, root-owned copy of the crate as collected. Every run below is given a fresh copy
# of this rather than whatever the previous run left behind. Six runs share this container, and
# without the reset a build script could stash the calibration capture's numbers in the target
# directory and a later serving run could read them back -- which would put the weights back
# within reach of a run that is supposed to be working without them.
rm -rf /opt/kernel_pristine
cp -a /app/kernel /opt/kernel_pristine
chown -R root:root /opt/kernel_pristine
# Root-owned is not enough on its own: cp -a keeps the modes the sources arrived with, and a
# world-writable main.rs would let a calibration run edit the tree every later run is rebuilt
# from, which is a channel for the weights. Nobody but root writes here.
chmod -R u=rwX,go=rX /opt/kernel_pristine

reset_crate() {
  # Nothing a run wrote survives into the next one, wherever it put it. The crate tree and
  # /app/out are replaced below; everything else the build user owns anywhere on the box is
  # removed here and the world-writable scratch directories are emptied, so a calibration run
  # cannot leave the weights in /tmp or its home directory for a serving run to read back.
  # Every mount is swept, not just the root one: a separate tmpfs such as /run would otherwise
  # be skipped, and /run/lock is world-writable on this base.
  find / \( -path /proc -o -path /sys -o -path /app/kernel -o -path /app/out \) \
       -prune -o -user runner -print0 2>/dev/null | xargs -0r rm -rf --
  for d in /tmp /var/tmp /dev/shm /run/lock /var/lock; do
    [ -d "$d" ] && find "$d" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + 2>/dev/null
  done
  mkdir -p /home/runner && chown runner:runner /home/runner && chmod 700 /home/runner
  rm -rf /app/kernel
  cp -a /opt/kernel_pristine /app/kernel
  chown -R runner:runner /app/kernel
  # Harbor uploads the agent's sources with whatever mtime they carried in its container, which
  # can predate this image's own layers. Cargo decides freshness by mtime, so an older source
  # silently counts as "already built" -- stamp them to now so the rebuild is unconditional.
  find /app/kernel/src -type f -exec touch {} + 2>/dev/null || true
  rm -rf /app/out
  mkdir -p /app/out
  chown runner:runner /app/out
}

# $1 = calibrate | apply
# Every compiler and interpreter on the box, by real path. They are open to the build user while
# cargo builds the crate, since linking needs cc, and closed to it while the built binary runs,
# so the arithmetic that produces the graded files cannot be handed to gcc, python or perl at run
# time. Root is unaffected, so pytest and Harbor's own tooling still run. bash is deliberately not
# on this list: Harbor collects the logs with it.
# The shells are on the list as well, so a kernel cannot hand the work to a script it carries.
# They are closed only for the seconds the binary runs and reopened straight after, and the
# trap reopens them if this script dies in between, so log collection is never left without a
# shell. This script itself is already running and root besides.
# The toolchain binaries under rustup are on the list as well, not just the cargo and rustc
# shims on PATH, so a kernel cannot compile at run time by calling them by their real paths.
TOOLS="$( { for t in python3 python3.13 pip gcc cc g++ ld as perl awk mawk rustc cargo bash sh dash; do
             p="$(command -v "$t" 2>/dev/null)" && readlink -f "$p"; done
           find /usr/local/rustup/toolchains -type f -perm /111 -path '*/bin/*' 2>/dev/null; } | sort -u)"
close_tools() { for f in $TOOLS; do chmod 750 "$f" 2>/dev/null || true; done; }
open_tools()  { for f in $TOOLS; do chmod 755 "$f" 2>/dev/null || true; done; }
trap open_tools EXIT
open_tools
echo "tools closed to the build user while the kernel runs: $(echo $TOOLS | tr '\n' ' ')"

run_step() {
  reset_crate
  setsid \
    setpriv --clear-groups --no-new-privs --reuid runner --regid runner \
    bash -c "cd /app/kernel && cargo build --release --offline" &
  local bpid=$!
  wait "$bpid"
  echo "build exit status ($1): $?"
  kill -TERM -- "-$bpid" 2>/dev/null || true
  kill -KILL -- "-$bpid" 2>/dev/null || true
  as_runner bash -c 'kill -9 -1' 2>/dev/null || true
  # The binary is whatever cargo built into target/release, found by name of the package in the
  # manifest, so a renamed package still runs; the instruction's entry point is cargo run, and
  # this is the same binary that command would start.
  local bin
  bin="$(python3 -c 'import tomllib,sys; m=tomllib.load(open("/app/kernel/Cargo.toml","rb")); print(m["package"]["name"].replace("-", "_"))' 2>/dev/null)"
  [ -x "/app/kernel/target/release/$bin" ] || bin="$(find /app/kernel/target/release -maxdepth 1 -type f -perm /111 -printf '%f\n' 2>/dev/null | head -1)"
  echo "kernel binary: target/release/$bin"
  close_tools
  # Started directly, not through a shell, since the shells are closed too, and from /app/kernel,
  # the directory the instruction's cargo run command is given, so a kernel that resolves its
  # files relative to that directory behaves as it does under that command.
  setsid \
    setpriv --clear-groups --no-new-privs --reuid runner --regid runner \
    env -C /app/kernel "/app/kernel/target/release/$bin" "$1" &
  local pid=$!
  wait "$pid"
  echo "kernel exit status ($1): $?"
  open_tools
  # Reap before anything else starts. A double-forked child left alive from one run would
  # otherwise still be around during the next, where it could write into /app/out and pass off
  # one run's numbers as another's.
  kill -TERM -- "-$pid" 2>/dev/null || true
  kill -KILL -- "-$pid" 2>/dev/null || true
  as_runner bash -c 'kill -9 -1' 2>/dev/null || true
}

# $1 = destination. Everything a run produced, moved out of runner's reach before the next one.
# Only regular files come out. A symlink left at params.bin would be followed by the plain cp
# in install_scheme and whatever it pointed at, the weight matrix included, would land in
# /app/scheme as an ordinary file for the serving runs to read. Anything that is not a regular
# file is left behind and named in the log.
capture() {
  rm -rf "$1"
  mkdir -p "$1"
  find /app/out -mindepth 1 -maxdepth 1 ! -type f -printf 'not staged, not a regular file: %p\n' 2>/dev/null
  find /app/out -mindepth 1 -maxdepth 1 -type f -exec cp -p {} "$1"/ \; 2>/dev/null
  chown -R root:root "$1"
  chmod -R go-rwx "$1"
}

# $1 = weights file or "" for none, $2 = activations file.
# The serving runs get no weights at all: that is the whole point of the split, so the weight
# side cannot be re-derived against the capture being graded.
set_data() {
  rm -rf /app/data
  mkdir -p /app/data
  if [ -n "$1" ]; then
    cp -f "$1" /app/data/weights.bin
  fi
  cp -f "$2" /app/data/activations.bin
  chown -R root:root /app/data
  chmod 444 /app/data/*.bin
  chmod 555 /app/data
}

# $1 = the calibration run's captured output. Only params.bin crosses over, and it crosses
# read-only: a serving run reads what calibration decided and cannot revise it.
#
# The weight side does NOT cross. Not the float weights and not the int8 codes either. A
# serving run that could read weights_q.bin could score its own output against the very
# matrix it is about to be graded on, and refine its activation codes per capture until the
# error came down -- which reaches the bar without ever migrating anything, and leaves
# params.bin empty. Withholding the weight side removes that route by construction: with no
# weights on the box there is nothing to measure an output error against, so the only thing
# that can protect the peaky channels is the factor calibration already committed.
install_scheme() {
  rm -rf /app/scheme
  mkdir -p /app/scheme
  for f in params.bin; do
    cp -f "$1/$f" /app/scheme/ 2>/dev/null || true
  done
  chown -R root:root /app/scheme
  chmod 444 /app/scheme/* 2>/dev/null || true
  chmod 555 /app/scheme
  echo "scheme handed to the serving runs: $(ls /app/scheme 2>/dev/null | tr '\n' ' ')"
}

# ---------------------------------------------------------------------------------------
# Each export gets one calibration run, which is handed both tensors, and then two serving
# runs, which are handed a capture that was not on the box during calibration and no weights.
#
# The second export is a different draw of both tensors, baked into this image and never
# present in any container the agent can reach, so a kernel that writes numbers worked out
# once into its source produces correctly shaped files holding the wrong values.
for export in shipped holdout; do
  if [ "$export" = "shipped" ]; then
    W=/opt/shipped/weights.bin
    A=/opt/shipped/activations.bin
  else
    W=/opt/holdout/weights.bin
    A=/opt/holdout/activations.bin
  fi

  echo "=== ${export}: calibration run ==="
  set_data "$W" "$A"
  run_step calibrate
  capture "/opt/cal_${export}"
  install_scheme "/opt/cal_${export}"

  for j in 0 1; do
    echo "=== ${export}: serving run ${j} ==="
    set_data "" "/opt/apply/${export}/batch${j}_acts.bin"
    run_step apply
    capture "/opt/run_${export}_${j}"
  done
done

# Restore the agent's export exactly as collected, so the sha256 check tests what the agent
# left rather than what this script has been swapping in and out.
rm -rf /app/data
mkdir -p /app/data
cp -a /opt/agent_data/. /app/data/
chmod -R a-w /app/data

# -P keeps the working directory (/app, per the Dockerfile's WORKDIR) off sys.path. The crate
# runs as `runner` and writes into /app/out, so without this it can drop an /app/pytest.py
# holding sys.exit(0); `python -m pytest` would import that no-op, exit 0, and bypass the whole
# suite for reward 1. The test file and cache are addressed absolutely either way.
python -P -m pytest -o cache_dir=/tmp/pytest_cache \
  --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

# Harbor grades from reward.txt, never from this script's exit status.
exit 0
