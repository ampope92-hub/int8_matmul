"""Verifier for the int8 dense-layer kernel.

test.sh deletes whatever was in /app/out, rebuilds the crate at /app/kernel from the
agent's collected sources and runs it, so the files graded here are always the output of a
clean release build of the agent's own code.

Two things decide the reward. The consistency check recomputes the product from the
agent's *own* declared int8 tensors and scales and requires output.bin to match it, so a
float32 matmul written straight to output.bin fails even though it would be numerically
perfect against the reference. The accuracy check scores each output column against the
double-precision reference in /opt/golden (root-only, never in the agent environment) and
requires the worst column to reach the stated SQNR bar.

numpy is available in the verifier image only; the agent environment has no Python
numerics stack.
"""

import hashlib
import pathlib

import numpy as np
import pytest

M, K, N = 96, 512, 192

OUT = pathlib.Path("/app/out")
OUTPUT = OUT / "output.bin"
WQ = OUT / "weights_q.bin"
WS = OUT / "w_scales.bin"
AQ = OUT / "acts_q.bin"
AS = OUT / "a_scales.bin"

DATA = pathlib.Path("/app/data")
REFERENCE = pathlib.Path("/opt/golden/reference.bin")

SQNR_BAR_DB = 32.0
CONSISTENCY_RTOL = 1e-4

# frozen at task-build time; the shipped export must not be altered
ACTIVATIONS_SHA = "f4bd04653352ebcefd8d8a9047f39f2f5d6a63f5d77d8ec51fae883ec7e6a270"
WEIGHTS_SHA = "ac4cbff09209f0e86c073773e2888190dbb4775ca2b52e997e034bfa49531769"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f32(path, count):
    raw = path.read_bytes()
    assert len(raw) == count * 4, (
        f"{path} should be {count * 4} bytes of little-endian f32, got {len(raw)}"
    )
    return np.frombuffer(raw, dtype="<f4")


def _i8(path, count):
    raw = path.read_bytes()
    assert len(raw) == count, f"{path} should be {count} signed bytes, got {len(raw)}"
    return np.frombuffer(raw, dtype=np.int8)


@pytest.fixture(scope="session")
def declared():
    """The agent's five output files, shaped, with sizes already validated."""
    for p in (OUTPUT, WQ, WS, AQ, AS):
        assert p.exists(), f"{p} was not written — `cargo run --release` in /app/kernel must produce it"
    return {
        "out": _f32(OUTPUT, M * N).reshape(M, N),
        "wq": _i8(WQ, K * N).reshape(K, N),
        "ws": _f32(WS, N),
        "aq": _i8(AQ, M * K).reshape(M, K),
        "as": _f32(AS, M),
    }


@pytest.fixture(scope="session")
def reference():
    """The withheld double-precision product, as f32."""
    return _f32(REFERENCE, M * N).reshape(M, N).astype(np.float64)


def test_shipped_export_unmodified():
    """The shipped weights and activations are byte-for-byte as issued."""
    assert _sha(DATA / "activations.bin") == ACTIVATIONS_SHA, "activations.bin was altered"
    assert _sha(DATA / "weights.bin") == WEIGHTS_SHA, "weights.bin was altered"


def test_scales_are_positive_and_finite(declared):
    """Every declared scale is finite and strictly positive."""
    for name in ("ws", "as"):
        s = declared[name].astype(np.float64)
        assert np.isfinite(s).all(), f"{name} contains a non-finite scale"
        assert (s > 0).all(), f"{name} contains a non-positive scale"


def test_output_is_finite_and_not_degenerate(declared):
    """output.bin is finite and carries real variation rather than a constant fill."""
    out = declared["out"].astype(np.float64)
    assert np.isfinite(out).all(), "output.bin contains NaN or inf"
    assert out.std() > 1e-6, "output.bin is essentially constant — nothing was computed"
    assert np.count_nonzero(out) > out.size // 2, "output.bin is mostly zeros"


def test_quantized_tensors_use_the_signed_range(declared):
    """The int8 tensors are not degenerate — they span more than a couple of levels."""
    for name, expected in (("wq", "weights_q.bin"), ("aq", "acts_q.bin")):
        q = declared[name].astype(np.int32)
        assert np.abs(q).max() > 1, f"{expected} is all zeros or +/-1 — no quantization happened"


def test_output_matches_the_declared_quantization(declared):
    """output.bin equals the integer product of the agent's own declared tensors.

    This is what stops a float32 matmul being written straight to output.bin: such a file
    is accurate against the reference but inconsistent with the int8 tensors beside it.
    """
    acc = declared["aq"].astype(np.int64) @ declared["wq"].astype(np.int64)
    expected = (
        acc.astype(np.float64)
        * declared["as"].astype(np.float64)[:, None]
        * declared["ws"].astype(np.float64)[None, :]
    )
    out = declared["out"].astype(np.float64)
    scale = max(float(np.abs(expected).max()), 1e-12)
    worst = float(np.abs(out - expected).max()) / scale
    assert worst <= CONSISTENCY_RTOL, (
        f"output.bin disagrees with the product of the declared int8 tensors and scales by "
        f"a relative {worst:.2e} (limit {CONSISTENCY_RTOL:.0e}) — the reported numbers are "
        f"not the ones those tensors produce"
    )


def test_worst_column_sqnr_meets_bar(declared, reference):
    """The lowest per-column SQNR against the reference reaches the required 32 dB."""
    out = declared["out"].astype(np.float64)
    signal = (reference ** 2).sum(axis=0)
    noise = ((out - reference) ** 2).sum(axis=0)
    sqnr = 10.0 * np.log10(signal / np.maximum(noise, 1e-300))
    worst = float(sqnr.min())
    worst_col = int(sqnr.argmin())
    assert worst >= SQNR_BAR_DB, (
        f"worst-column SQNR is {worst:.2f} dB at column {worst_col}, below the required "
        f"{SQNR_BAR_DB:.0f} dB — that column is dominated by quantization noise"
    )
