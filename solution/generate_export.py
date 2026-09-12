"""Generates both tensor exports this task ships, from one process, deterministically.

    python solution/generate_export.py shipped   -> environment/data/{weights,activations}.bin
    python solution/generate_export.py holdout   -> tests/holdout/{weights,activations}.bin

Both targets also write tests/apply/<target>/, the serving captures the frozen scheme is
graded on, with the double-precision product of each one.

The two exports come from the same distribution but different draws, so a kernel that bakes
in numbers worked out from the shipped tensors writes correctly shaped files holding the
wrong values on the second run.

Structure, and why each part is there:

  weights (512x192)     per output column n, entries ~ N(0, sigma_n), log(sigma_n) uniform
                        over log[0.02, 1.55]. The ~90x spread of column peaks is what makes
                        a single tensor-wide weight scale hopeless: the quantization step
                        sized for the largest column erases the smallest ones.

  activations (96x512)  iid standard normal, except that OUTLIER_CHANNELS of the 512 input
                        channels are drawn 45x larger. This is the ordinary shape of
                        transformer activations: a small number of feature channels run
                        orders of magnitude hotter than the rest, consistently across every
                        row of the batch. It is a property of the channel, not of the row.

The interaction between those two is the point. A per-row activation scale is set by that
row's peak, which lands on an outlier channel, so every ordinary channel in the row is
quantized with a step 45x coarser than it needs -- and a per-row scale cannot fix it, because
the hot channels are the same ones in every row. The magnitude can instead be migrated into
the weights, which costs nothing: dividing activation channel k by s_k and multiplying weight
row k by s_k leaves the product exactly unchanged. How far to migrate is the whole question.
Migrating none of it is the reflex and fails; migrating all of it hands the entire 45x spread
to the weight columns and fails just as badly, because a weight column's scale then has to
cover its own hot rows. A balanced split clears the bar with room to spare.
"""
import hashlib
import pathlib
import sys

import numpy as np

M, K, N = 96, 512, 192
SIG_LO, SIG_HI = 0.02, 1.55
OUTLIER_CHANNELS = 12
OUTLIER_BOOST = 45.0

TARGETS = {"shipped": 5, "holdout": 23}


def build(seed):
    rng = np.random.default_rng(seed)
    col_sigma = np.exp(rng.uniform(np.log(SIG_LO), np.log(SIG_HI), N))
    W = (rng.standard_normal((K, N)) * col_sigma).astype(np.float32)
    A = rng.standard_normal((M, K)).astype(np.float32)
    hot = np.sort(rng.choice(K, OUTLIER_CHANNELS, replace=False))
    A[:, hot] *= OUTLIER_BOOST
    return W, A.astype(np.float32), hot


APPLY_BATCHES = 2


def apply_batches(seed, hot):
    """Fresh activation captures from the same trace, for the apply runs.

    Same hot channels and same boost as the calibration capture, because they are a property
    of the layer rather than of the batch, but an independent draw. Drawn from their own
    generator so that adding them cannot disturb the export above: `build` consumes its rng in
    the same order it always did, and the shipped tensors stay byte-identical.
    """
    rng = np.random.default_rng(seed * 1000 + 7)
    for j in range(APPLY_BATCHES):
        A = rng.standard_normal((M, K)).astype(np.float32)
        A[:, hot] *= OUTLIER_BOOST
        yield j, A.astype(np.float32)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "shipped"
    if target not in TARGETS:
        sys.exit(f"target must be one of {sorted(TARGETS)}")
    seed = TARGETS[target]
    W, A, hot = build(seed)

    root = pathlib.Path(__file__).resolve().parent.parent
    # Nothing is graded against the calibration capture's own product, so no reference for it
    # is written or shipped: accuracy is scored on the serving captures below, which do not
    # exist as far as the calibration step is concerned.
    if target == "shipped":
        out = root / "environment" / "data"
    else:
        out = root / "tests" / "holdout"
    files = {"weights.bin": W, "activations.bin": A}
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in files.items():
        arr.astype("<f4").tofile(out / name)

    # The two draws must not collide after quantization: a per-column scale divides each
    # column by its own peak, so a shared standard-normal draw would survive as identical
    # int8 weights even with different sigmas. Different seeds make them independent, and
    # test_weight_side_is_not_a_constant keeps that honest.
    cmax = np.abs(W).max(axis=0)
    amax = np.abs(A).max(axis=0)
    print(f"{target} export (seed {seed}) -> {out}")
    print(f"  weight per-column peak spread {cmax.max() / cmax.min():.1f}x")
    print(f"  hot activation channels {list(hot)}")
    print(f"  activation channel peak spread {amax.max() / np.median(amax):.1f}x median")
    for name in files:
        b = (out / name).read_bytes()
        print(f"  {name:16s} {len(b):>7d} bytes  sha256 {hashlib.sha256(b).hexdigest()}")

    # The batches the frozen scheme is graded on. These never appear in the agent's container:
    # the weight side is committed from the capture above, and only then does the verifier
    # hand the kernel a batch it has not seen. That is what stops the graded objective from
    # being computable at solve time, and it is the whole point of the split.
    apply_dir = root / "tests" / "apply" / target
    apply_dir.mkdir(parents=True, exist_ok=True)
    for j, A_j in apply_batches(seed, hot):
        ref_j = (A_j.astype(np.float64) @ W.astype(np.float64)).astype(np.float32)
        A_j.astype("<f4").tofile(apply_dir / f"batch{j}_acts.bin")
        ref_j.astype("<f4").tofile(apply_dir / f"batch{j}_ref.bin")
        corr = np.corrcoef(np.abs(A).max(0), np.abs(A_j).max(0))[0, 1]
        print(f"  apply batch {j}: channel-peak correlation with the capture {corr:.3f}")


if __name__ == "__main__":
    main()
