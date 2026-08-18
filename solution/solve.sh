#!/usr/bin/env bash
set -euo pipefail

# Reference int8 path for the dense layer.
#
# The thing that decides this one is where the weight scale comes from. The columns of
# this checkpoint are not on the same footing at all - the largest column peaks near 4.9
# and the smallest near 0.055, so a single scale taken over the whole matrix is set by the
# loudest column and leaves the quiet ones with a couple of usable levels. Their output
# then arrives as mostly quantization noise, and since the bar is on the *worst* column, a
# good average does not save it. One scale per output column fixes it: every column gets
# the full signed range regardless of how quiet it is.
#
# Clipping the scale below the column max is a trap here rather than a win. These weights
# are not activation outliers - the large entries carry real signal into the product, so
# trading them away for a finer step loses more than it gains. Measured on this export:
# column max scaling lands around 36 dB, clipping at the 99.9th percentile drops to 26.
#
# Activations go per row. The rows are within about 20% of each other so one scale over the
# whole batch would also have cleared the bar, but per row costs nothing and is what the
# serving path does anyway.
#
# Rounding is round-to-nearest, not truncation: truncation biases every entry toward zero
# and costs about 6 dB, which is the difference between passing and not.

# The model itself is solution/main.rs, next to this script — Rust is not a shell
# language and 102 lines of it inside a heredoc is neither reviewable nor
# compilable on its own. Drop it in and let cargo build it.
install -m 644 "$(dirname "$0")/main.rs" /app/kernel/src/main.rs

cd /app/kernel
cargo run --release --offline
