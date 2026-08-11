# Dense layer export — 2026-07-30

Pulled from the checkpoint we're preparing for the int8 release. Both files are raw
little-endian f32, row-major, no header — the exporter writes tensors straight out of
memory, so shape lives here rather than in the file.

| file | shape | contents |
|---|---|---|
| `weights.bin` | 512 x 192 | the layer's weight matrix, input dim by output dim |
| `activations.bin` | 96 x 512 | a batch of real activations captured at that layer |

The activation batch is the calibration capture from a production trace, not synthetic
noise — it was taken at the same layer over a run of ordinary requests, so the row
statistics are representative of what the kernel sees in serving.

Nothing here has been rescaled, clipped or reordered on the way out. The weight columns
are in the checkpoint's own order, which is also the order the next layer expects.
