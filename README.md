# Int8 Dense-Layer Quantization Kernel

A dense layer (512 x 192 weights) is moving onto an int8 serving path, and the agent writes the kernel by hand in Rust, offline, with no BLAS and an empty dependency list. The binary has two modes:

- **`calibrate`** sees the f32 weights and one activation capture (96 x 512). It writes int8 weights with one scale per output column, int8 activations with one scale per row, the dequantized product, and up to 512 f32 of extra parameters (`params.bin`).
- **`apply`** is the serving step and is the one that is scored. It sees only `params.bin` and a fresh activation capture it has never seen. No weights are on the box in any form. It must emit int8 activation codes and per-row scales.

The server combines those codes with the weight side committed at calibration. Accuracy is the **worst-column** SQNR against the exact double-precision product, and it must reach 29 dB.

## What makes it hard

Writing the matmul is routine. The real problem is a scheme decision that must be committed before the graded data exists.

Twelve of the 512 input channels carry values about 45x larger than the rest, and they are the same channels in every row. A per-row activation scale is always set by an outlier, so every ordinary channel in the row is quantized with a step size that is far too coarse. No per-row scale can avoid this.

The fix is to migrate the magnitude: divide activation channel k by s_k and multiply weight row k by s_k, which leaves the product unchanged. This must happen at calibration, because only that step sees both tensors, and the 512 factors are what `params.bin` is for. The split must also be balanced: s_k = a_peak^alpha / w_peak^(1-alpha).

Measured routes on the shipped export's serving captures:

| Approach | Worst-column SQNR |
|---|---|
| Quantize each capture as it arrives | 23.4 to 23.8 dB |
| Error-feedback rounding of weight codes against the calibration capture | 23.2 to 23.9 dB (does not transfer to a new batch) |
| Migrate all magnitude into the weights (alpha = 1) | 25.1 to 25.7 dB |
| Factors from weight peaks alone | 23.0 to 23.6 dB |
| Any balanced split, alpha 0.3 to 0.7 | at least 33.4 dB |
| Reference (alpha = 0.5) | about 36.9 dB |

The 29 dB bar sits 3.3 dB above the best failing route and 3.9 dB below the worst working one. The serving step never receives weights, so it cannot tune its activation codes against the graded product. The failure is diagnosable locally: re-scoring on the calibration capture shows 24.4 dB for the naive scheme and 36.6 dB for the balanced split.

The data README describes the export as a production capture. That framing is deliberate, so that the agent measures the tensors rather than reverse-engineering a generator. Both exports are actually synthetic, reproducible from seeds with `solution/generate_export.py`.

## How it's graded

The verifier runs in a separate offline container and rebuilds the collected crate from clean. Each of two exports gets one calibration run and then two serving runs on captures baked into the verifier image. The second export has a different draw of both tensors, a different set of hot channels, and a 28 dB bar. The weight side from calibration is frozen root-only, and `weights.bin` is removed before serving.

Twelve checks run:

- The crate is plain Rust: no build script, all dependency tables empty, only `.rs` UTF-8 files under `src`.
- The shipped export is byte-identical to what was issued.
- Calibration outputs have the declared shapes, and `params.bin` stays within its cap.
- `output.bin` matches the run's own integer product to a relative 1e-4.
- `output.bin`'s worst column is within 20 dB of the exact product, which rules out all-zero codes.
- Each of the four serving runs meets its worst-column SQNR bar.

Anti-cheat measures:

- Every run starts on a fresh box. Files the build user left anywhere, including `/tmp` and `/dev/shm`, are wiped, so calibration cannot stash the weights for serving.
- `params.bin` is capped at 512 floats, so it cannot carry the weight matrix.
- Only regular files are staged out of `/app/out`, which blocks symlink tricks.
- The binary runs unprivileged with every compiler, interpreter and shell closed to it.
- Hard-coded constants fail on the second export.

## Repository layout

- `instruction.md`: the task prompt given to the agent.
- `task.toml`: metadata, difficulty, solution and verification notes (including all measured routes), and resource limits.
- `environment/`: the agent's container (`Dockerfile`), the shipped export with its data README (`data/`) and the empty Rust crate (`kernel/`).
- `solution/`: `solve.sh`, the reference kernel (`main.rs`, with `SMOOTH_ALPHA` as the migration exponent) and the seeded export generator (`generate_export.py`).
- `tests/`: verifier image and entrypoint (`Dockerfile`, `test.sh`), pytest checks (`test_outputs.py`), the pristine crate skeleton (`skeleton/`), the shipped and held-back exports (`shipped/`, `holdout/`) and the withheld serving captures with reference products (`apply/`).

## Running it

Requires Docker and Harbor (`uv tool install harbor` or `pip install harbor`). From the repo root:

```
harbor run -p . -a oracle
```

This runs the reference solution through the verifier and should score reward 1. To try an agent instead:

```
harbor run -p . -a <agent> -m <model>
```

For example, `-a claude-code`.

---

Task design, data generators, reference solution and verifier by Audrey Pope.
