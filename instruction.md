We're putting one of our linear layers onto an int8 path and the kernel has to be written here, in Rust, by hand — the box is offline, there's no BLAS, and `/app/kernel/Cargo.toml` has an empty dependency list on purpose — it and `/app/kernel/Cargo.lock` come back with the sources when we collect your work, so leave them buildable.

`/app/data/weights.bin` is the layer's weight matrix, 512 rows by 192 columns of little-endian f32, row-major, no header. `/app/data/activations.bin` is a batch of activations, 96 rows by 512 columns, same encoding. `/app/data/README.md` covers where they came from. Fill in `/app/kernel/src/main.rs` so that `cargo run --release` from `/app/kernel` computes the activations times the weights through an int8 quantized path and writes five files into `/app/out`:

- `output.bin` — the dequantized product, 96 by 192, little-endian f32, row-major
- `weights_q.bin` — your quantized weights, 512 by 192, one signed byte per entry, row-major
- `w_scales.bin` — 192 little-endian f32, one per output column
- `acts_q.bin` — your quantized activations, 96 by 512, one signed byte per entry, row-major
- `a_scales.bin` — 96 little-endian f32, one per activation row

On grading: we delete `/app/out`, rebuild the crate from your sources and run it ourselves, so it has to work from a clean build. Two things get checked. First, `output.bin` has to agree with the integer product of the tensors you declared — `a_scales[m] * w_scales[n] * sum over k of acts_q[m,k] * weights_q[k,n]` — to a relative 1e-4, so the numbers you hand back have to be the ones your own quantized tensors produce. Second, accuracy is scored column by column against the same product computed in double precision: for column `n`, `SQNR_n = 10 * log10( sum over m of ref[m,n]^2 / sum over m of (out[m,n] - ref[m,n])^2 )`. The **lowest** `SQNR_n` across all 192 columns has to reach 32 dB. We grade the worst column because that's the one that shows up downstream — a good average with one starved column is the failure mode we keep hitting.
