"""Verifier for the int8 dense-layer kernel.

test.sh rebuilds the crate at /app/kernel from the agent's collected sources before every
run, so everything graded here is the output of a clean release build of the agent's own
code. Each export gets one calibration run, which is handed both tensors and commits the
weight side, and then two serving runs, which are handed a capture the calibration run never
saw and no weights at all.

Accuracy is graded on the serving runs, and the product is formed here rather than read from
the kernel: the frozen weights_q/w_scales from the calibration run are put beside the
acts_q/a_scales from the serving run. So a kernel cannot aim its integer codes at anything
except a faithful quantization of the capture it was handed, because at that point it does
not have the weights to aim at.

numpy is available in the verifier image only; the agent environment has no Python numerics
stack.
"""

import pathlib

import numpy as np
import pytest
import tomllib

M, K, N = 96, 512, 192

EXPORTS = ("shipped", "holdout")
BATCHES = (0, 1)

DATA = pathlib.Path("/app/data")
# The withheld serving captures and their double-precision reference products, baked into
# the verifier image at build time and never present in any agent-reachable container.
APPLY = pathlib.Path("/opt/apply")

# Calibrated on the captures that ship, with the weight side frozen and graded on the two
# serving batches of each export. The routes that fail are the ones an implementer reaches
# for first: quantizing each row of the capture as it arrives lands at 23.4-23.8 dB on the
# shipped export and 24.6-25.0 on the holdout, because a row's scale is set by an outlier
# channel; error-feedback rounding of the weight codes against the calibration capture lands
# in the same place, 23.2-23.9 and 24.4-24.8, because the error it is minimising is not the
# one that survives to a different batch; a single tensor-wide weight scale scores under
# 3 dB. Moving all of each channel's magnitude into the weights fails from the other side at
# 23.6-25.7. Every balanced split of that magnitude reaches at least 32.5 dB on both exports.
# The bars sit in that gap, about 3 dB above the best route that fails and 4 dB below the
# worst route that works.
SQNR_BAR_DB = {"shipped": 29.0, "holdout": 28.0}
# Floor for the calibration run's own output against the exact product of the tensors it was
# handed. Every real quantization clears it by a wide margin (the plain per-row route sits near
# 24 dB on its own capture), so it grades nothing about the scheme. It exists because the
# consistency identity alone is satisfied by all-zero codes and an all-zero product, and that is
# not a quantization of the capture at all.
CAL_FLOOR_DB = 20.0
CONSISTENCY_RTOL = 1e-4

# params.bin is the only thing that crosses from calibration into serving, and the cap is
# what makes it a decision rather than a smuggling channel: 512 f32 cannot carry the 512x192
# weight matrix, so a serving run cannot reconstruct the weights from it.
PARAMS_MAX_FLOATS = 512

# The shipped export as issued, a root-only copy baked into this image by tests/Dockerfile. The
# copy that comes back from the agent is compared against these bytes rather than against a
# digest constant, so there is no constant to fall out of step with the file the agent is given.
SHIPPED = pathlib.Path("/opt/shipped")


def _f32(path, count):
    assert path.exists(), f"{path.name} was not written by the {path.parent.name} run"
    raw = path.read_bytes()
    assert len(raw) == count * 4, (
        f"{path.name} should be {count * 4} bytes of little-endian f32, got {len(raw)}"
    )
    return np.frombuffer(raw, dtype="<f4")


def _i8(path, count):
    assert path.exists(), f"{path.name} was not written by the {path.parent.name} run"
    raw = path.read_bytes()
    assert len(raw) == count, f"{path.name} should be {count} signed bytes, got {len(raw)}"
    return np.frombuffer(raw, dtype=np.int8)


def _cal_dir(export):
    return pathlib.Path(f"/opt/cal_{export}")


def _export_dir(export):
    """The export as issued, root-only copies baked into this image."""
    return SHIPPED if export == "shipped" else pathlib.Path("/opt/holdout")


def _run_dir(export, batch):
    return pathlib.Path(f"/opt/run_{export}_{batch}")


@pytest.fixture(scope="session")
def scheme():
    """The weight side each calibration run committed, per export."""
    got = {}
    for export in EXPORTS:
        d = _cal_dir(export)
        got[export] = {
            "wq": _i8(d / "weights_q.bin", K * N).reshape(K, N),
            "ws": _f32(d / "w_scales.bin", N),
        }
    return got


def _sqnr(out, ref):
    signal = (ref ** 2).sum(axis=0)
    noise = ((out - ref) ** 2).sum(axis=0)
    return 10.0 * np.log10(signal / np.maximum(noise, 1e-300))


KERNEL = pathlib.Path("/opt/kernel_pristine")


def test_kernel_is_plain_rust():
    """The crate is Rust and nothing else, as the instruction says: the manifest declares no
    build script and no build dependencies, its dependency list is empty, and src/ holds only
    .rs files. Together with test.sh closing every compiler and interpreter to the build user
    while the binary runs, that is what keeps the arithmetic in the agent's own Rust. Nothing
    here reads the source text itself, since a name in a comment is not a use of it."""
    with (KERNEL / "Cargo.toml").open("rb") as fh:
        manifest = tomllib.load(fh)
    assert "build" not in manifest.get("package", {}), "Cargo.toml declares a build script"

    # Every dependency table of every kind, at any depth: [dependencies], [dev-dependencies],
    # [build-dependencies] and the [target.'cfg(...)'.*dependencies] forms all have to be empty.
    def dependency_tables(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("dependencies"):
                    yield path + key, value
                else:
                    yield from dependency_tables(value, path + key + ".")
    listed = [(where, list(val)) for where, val in dependency_tables(manifest) if isinstance(val, dict) and val]
    assert not listed, f"Cargo.toml lists dependencies; every such table has to stay empty: {listed[:3]}"
    files = sorted(p for p in (KERNEL / "src").rglob("*") if p.is_file())
    stray = [str(p.relative_to(KERNEL)) for p in files if p.suffix != ".rs"]
    assert not stray, f"src/ holds files that are not Rust: {stray[:5]}"
    # A .rs name on a binary is not a source file: every file under src has to be UTF-8 text.
    for p in files:
        raw = p.read_bytes()
        try:
            raw.decode("utf-8")
            text = b"\x00" not in raw
        except UnicodeDecodeError:
            text = False
        assert text, f"{p.relative_to(KERNEL)} is not a text file"


def test_shipped_export_unmodified():
    """The shipped weights and activations come back byte-for-byte as issued."""
    for name in ("activations.bin", "weights.bin"):
        assert (DATA / name).read_bytes() == (SHIPPED / name).read_bytes(), f"{name} was altered"


@pytest.mark.parametrize("export", EXPORTS)
def test_calibration_writes_the_committed_scheme(export):
    """The calibration run writes a weight side of the declared shapes and a params.bin.

    params.bin may be empty, but it has to be a whole number of little-endian f32 and no
    more than the stated cap.
    """
    d = _cal_dir(export)
    _i8(d / "weights_q.bin", K * N)
    _f32(d / "w_scales.bin", N)
    params = d / "params.bin"
    assert params.exists(), "params.bin was not written by the calibration run"
    raw = params.read_bytes()
    assert len(raw) % 4 == 0, (
        f"params.bin is {len(raw)} bytes, which is not a whole number of little-endian f32"
    )
    assert len(raw) // 4 <= PARAMS_MAX_FLOATS, (
        f"params.bin holds {len(raw) // 4} f32, above the {PARAMS_MAX_FLOATS} allowed"
    )


@pytest.mark.parametrize("export", EXPORTS)
def test_calibration_output_matches_the_declared_quantization(export):
    """output.bin equals the integer product of the tensors that calibration handed back.

    This is what stops a float32 matmul being written straight to output.bin: such a file is
    accurate against the layer but inconsistent with the int8 tensors beside it.
    """
    d = _cal_dir(export)
    aq = _i8(d / "acts_q.bin", M * K).reshape(M, K)
    a_s = _f32(d / "a_scales.bin", M)
    wq = _i8(d / "weights_q.bin", K * N).reshape(K, N)
    w_s = _f32(d / "w_scales.bin", N)
    out = _f32(d / "output.bin", M * N).reshape(M, N).astype(np.float64)

    acc = aq.astype(np.int64) @ wq.astype(np.int64)
    expected = (
        acc.astype(np.float64)
        * a_s.astype(np.float64)[:, None]
        * w_s.astype(np.float64)[None, :]
    )
    scale = max(float(np.abs(expected).max()), 1e-12)
    worst = float(np.abs(out - expected).max()) / scale
    assert worst <= CONSISTENCY_RTOL, (
        f"on the {export} export output.bin disagrees with the product of the declared int8 "
        f"tensors and scales by a relative {worst:.2e} (limit {CONSISTENCY_RTOL:.0e}) — the "
        f"reported numbers are not the ones those tensors produce"
    )


@pytest.mark.parametrize("export", EXPORTS)
def test_calibration_output_is_a_product_of_the_capture(export):
    """output.bin is a real int8 product of the capture calibration was given.

    The identity above is satisfied by zeros: all-zero codes give an all-zero product that agrees
    with an all-zero output.bin. So the output is also held against the exact product of the two
    tensors, and the worst column has to come within CAL_FLOOR_DB of it.
    """
    d = _cal_dir(export)
    src = _export_dir(export)
    acts = _f32(src / "activations.bin", M * K).reshape(M, K).astype(np.float64)
    weights = _f32(src / "weights.bin", K * N).reshape(K, N).astype(np.float64)
    out = _f32(d / "output.bin", M * N).reshape(M, N).astype(np.float64)
    worst = float(_sqnr(out, acts @ weights).min())
    assert worst >= CAL_FLOOR_DB, (
        f"on the {export} export the calibration run's output.bin is {worst:.1f} dB from the exact "
        f"product of the capture and the weights in its worst column, under the {CAL_FLOOR_DB:.0f} "
        f"dB floor. The codes it handed back are not a quantization of the capture it was given"
    )


@pytest.mark.parametrize("export", EXPORTS)
@pytest.mark.parametrize("batch", BATCHES)
def test_serving_run_worst_column_sqnr_meets_bar(scheme, export, batch):
    """The frozen weight side, put beside a serving run's codes, holds up on a fresh capture."""
    bar = SQNR_BAR_DB[export]
    d = _run_dir(export, batch)
    aq = _i8(d / "acts_q.bin", M * K).reshape(M, K)
    a_s = _f32(d / "a_scales.bin", M)
    wq = scheme[export]["wq"]
    w_s = scheme[export]["ws"]

    acc = aq.astype(np.int64) @ wq.astype(np.int64)
    out = (
        acc.astype(np.float64)
        * a_s.astype(np.float64)[:, None]
        * w_s.astype(np.float64)[None, :]
    )
    ref = (
        np.frombuffer((APPLY / export / f"batch{batch}_ref.bin").read_bytes(), dtype="<f4")
        .reshape(M, N)
        .astype(np.float64)
    )
    sqnr = _sqnr(out, ref)
    worst = float(sqnr.min())
    worst_col = int(sqnr.argmin())
    assert worst >= bar, (
        f"on the {export} export, serving run {batch}: worst-column SQNR is {worst:.2f} dB "
        f"at column {worst_col}, below the required {bar:.0f} dB — that column is dominated "
        f"by quantization noise, and the capture was not on the box when the weight side was "
        f"decided"
    )
