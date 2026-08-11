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

cat > /app/kernel/src/main.rs <<'RUST'
use std::fs;
use std::io::Write;

const M: usize = 96;
const K: usize = 512;
const N: usize = 192;

fn read_f32(path: &str, n: usize) -> Vec<f32> {
    let bytes = fs::read(path).expect("input tensor missing");
    assert_eq!(bytes.len(), n * 4, "{} has an unexpected length", path);
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn write_f32(path: &str, values: &[f32]) {
    let mut buf = Vec::with_capacity(values.len() * 4);
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fs::File::create(path).unwrap().write_all(&buf).unwrap();
}

fn write_i8(path: &str, values: &[i8]) {
    let bytes: Vec<u8> = values.iter().map(|v| *v as u8).collect();
    fs::File::create(path).unwrap().write_all(&bytes).unwrap();
}

// Symmetric quantize to the signed range, round-to-nearest, clamped.
fn quantize(x: f32, scale: f32) -> i8 {
    let q = (x as f64 / scale as f64).round();
    q.clamp(-127.0, 127.0) as i8
}

fn main() {
    let acts = read_f32("/app/data/activations.bin", M * K);
    let weights = read_f32("/app/data/weights.bin", K * N);

    // One scale per output column, from that column's own peak.
    let mut w_scales = vec![0f32; N];
    for n in 0..N {
        let mut peak = 0f32;
        for k in 0..K {
            let a = weights[k * N + n].abs();
            if a > peak {
                peak = a;
            }
        }
        w_scales[n] = (peak.max(1e-12) as f64 / 127.0) as f32;
    }

    // One scale per activation row.
    let mut a_scales = vec![0f32; M];
    for m in 0..M {
        let mut peak = 0f32;
        for k in 0..K {
            let a = acts[m * K + k].abs();
            if a > peak {
                peak = a;
            }
        }
        a_scales[m] = (peak.max(1e-12) as f64 / 127.0) as f32;
    }

    let mut wq = vec![0i8; K * N];
    for n in 0..N {
        for k in 0..K {
            wq[k * N + n] = quantize(weights[k * N + n], w_scales[n]);
        }
    }

    let mut aq = vec![0i8; M * K];
    for m in 0..M {
        for k in 0..K {
            aq[m * K + k] = quantize(acts[m * K + k], a_scales[m]);
        }
    }

    // Integer accumulation. K is 512 and each product reaches 127*127, so the running sum
    // needs more than 16 bits - i32 is the natural width and cannot overflow here.
    let mut out = vec![0f32; M * N];
    for m in 0..M {
        let arow = &aq[m * K..(m + 1) * K];
        for n in 0..N {
            let mut acc: i32 = 0;
            for k in 0..K {
                acc += arow[k] as i32 * wq[k * N + n] as i32;
            }
            out[m * N + n] = (acc as f64 * a_scales[m] as f64 * w_scales[n] as f64) as f32;
        }
    }

    fs::create_dir_all("/app/out").unwrap();
    write_f32("/app/out/output.bin", &out);
    write_i8("/app/out/weights_q.bin", &wq);
    write_f32("/app/out/w_scales.bin", &w_scales);
    write_i8("/app/out/acts_q.bin", &aq);
    write_f32("/app/out/a_scales.bin", &a_scales);
    println!("wrote /app/out ({} by {})", M, N);
}
RUST

cd /app/kernel
cargo run --release --offline
