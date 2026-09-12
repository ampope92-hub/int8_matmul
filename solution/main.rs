use std::fs;
use std::io::Write;

const K: usize = 512;
const N: usize = 192;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{} unreadable: {}", path, e));
    assert!(bytes.len() % 4 == 0, "{} is not a whole number of f32", path);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i8(path: &str) -> Vec<i8> {
    fs::read(path)
        .unwrap_or_else(|e| panic!("{} unreadable: {}", path, e))
        .into_iter()
        .map(|b| b as i8)
        .collect()
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

// Symmetric quantize to the signed range, round-to-nearest, clamped. Truncating instead
// biases every entry toward zero and costs about 6 dB, which is the whole margin.
fn quantize(x: f64, scale: f64) -> i8 {
    (x / scale).round().clamp(-127.0, 127.0) as i8
}

// How much of each input channel's magnitude is moved out of the activations and into the
// weights. 0.0 leaves the activations alone, which is the reflex; 1.0 flattens them
// completely and hands the whole channel spread to the weight columns. Both ends fail. The
// calibration behind the bar is reproducible from what ships: rebuild this file with the
// constant at 0.0 and at 1.0 for the two failing ends, and across 0.3 to 0.7 for the balanced
// ones, over exports drawn from solution/generate_export.py under other seeds.
const SMOOTH_ALPHA: f64 = 0.5;

/// Quantize one capture against a frozen set of channel factors: divide each channel by its
/// factor, then take one symmetric scale per row from that row's own peak.
///
/// The per-row scale is the only freedom a serving run has, and it is only worth anything
/// because the channel magnitudes came out first. Left in, a row's scale is set by whichever
/// hot channel that row happens to peak on, and every ordinary channel in the row is
/// quantized with a step sized for a value it has nothing to do with.
fn quantize_capture(acts: &[f32], factors: &[f64]) -> (Vec<i8>, Vec<f32>) {
    let rows = acts.len() / K;
    let scaled: Vec<f64> = (0..rows * K)
        .map(|i| acts[i] as f64 / factors[i % K].max(1e-12))
        .collect();

    let mut a_scales = vec![0f32; rows];
    for row in 0..rows {
        let mut peak = 0f64;
        for k in 0..K {
            peak = peak.max(scaled[row * K + k].abs());
        }
        a_scales[row] = (peak.max(1e-12) / 127.0) as f32;
    }

    let aq: Vec<i8> = (0..rows * K)
        .map(|i| quantize(scaled[i], a_scales[i / K] as f64))
        .collect();
    (aq, a_scales)
}

/// The integer product, dequantized. K is 512 and each term reaches 127*127, so the running
/// sum needs more than 16 bits: i32 is the natural width and cannot overflow here. Release
/// mode wraps silently rather than panicking, so a 16-bit accumulator fails quietly.
fn product(aq: &[i8], a_scales: &[f32], wq: &[i8], w_scales: &[f32]) -> Vec<f32> {
    let rows = a_scales.len();
    let mut out = vec![0f32; rows * N];
    for row in 0..rows {
        let arow = &aq[row * K..(row + 1) * K];
        for n in 0..N {
            let mut acc: i32 = 0;
            for k in 0..K {
                acc += arow[k] as i32 * wq[k * N + n] as i32;
            }
            out[row * N + n] = (acc as f64 * a_scales[row] as f64 * w_scales[n] as f64) as f32;
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// calibrate: both tensors are on the box, so this is where the weight side gets decided.
//
// A dozen of the 512 input channels carry activations tens of times larger than the rest, in
// every row of the capture. Because the hot channels are the same ones in every row, no choice
// of per-row activation scale escapes them: the problem belongs to the column, not the row.
//
// Dividing channel k of the activations by s_k and multiplying row k of the weights by the
// same s_k leaves the product algebraically identical, so the magnitude can be moved rather
// than lived with. It cannot simply be dumped on the weights: a column's scale then has to
// cover its own hot rows and the error comes straight back on the other side. Splitting it as
// s_k = a_peak^alpha / w_peak^(1-alpha) balances the two.
//
// s is the one thing a serving run cannot work out for itself, because it needs the activation
// peaks measured against the weights and it is never handed the weights. That is what
// params.bin is for.
fn calibrate() {
    let acts = read_f32("/app/data/activations.bin");
    let weights = read_f32("/app/data/weights.bin");
    assert_eq!(weights.len(), K * N, "weights.bin is the wrong length");

    let rows = acts.len() / K;
    let mut s = vec![1f64; K];
    for k in 0..K {
        let mut a_peak = 0f64;
        for row in 0..rows {
            a_peak = a_peak.max(acts[row * K + k].abs() as f64);
        }
        let mut w_peak = 0f64;
        for n in 0..N {
            w_peak = w_peak.max(weights[k * N + n].abs() as f64);
        }
        s[k] = a_peak.max(1e-12).powf(SMOOTH_ALPHA) / w_peak.max(1e-12).powf(1.0 - SMOOTH_ALPHA);
    }

    // One symmetric scale per output column, from that column's own peak. A single scale over
    // the whole matrix is set by the loudest column and leaves the quiet ones two or three
    // usable levels, and the bar is on the worst column rather than on an average.
    let migrated: Vec<f64> = (0..K * N).map(|i| weights[i] as f64 * s[i / N]).collect();
    let mut w_scales = vec![0f32; N];
    for n in 0..N {
        let mut peak = 0f64;
        for k in 0..K {
            peak = peak.max(migrated[k * N + n].abs());
        }
        w_scales[n] = (peak.max(1e-12) / 127.0) as f32;
    }
    let wq: Vec<i8> = (0..K * N)
        .map(|i| quantize(migrated[i], w_scales[i % N] as f64))
        .collect();

    // The capture that was used for calibration goes through the same serving path, so this
    // run hands back a complete, self-consistent set for the batch it was given.
    let (aq, a_scales) = quantize_capture(&acts, &s);
    let out = product(&aq, &a_scales, &wq, &w_scales); // the capture calibration was given

    fs::create_dir_all("/app/out").unwrap();
    write_i8("/app/out/weights_q.bin", &wq);
    write_f32("/app/out/w_scales.bin", &w_scales);
    write_f32(
        "/app/out/params.bin",
        &s.iter().map(|v| *v as f32).collect::<Vec<f32>>(),
    );
    write_i8("/app/out/acts_q.bin", &aq);
    write_f32("/app/out/a_scales.bin", &a_scales);
    write_f32("/app/out/output.bin", &out);
    println!("calibrated on {} rows: wrote six files to /app/out", rows);
}

// ---------------------------------------------------------------------------------------
// apply: a capture this run has never seen, and no weights at all -- not the float
// matrix and not the int8 codes either.
//
// Everything on the weight side was settled by calibrate and is read back out of /app/scheme
// unchanged. The only work left is quantizing the capture against the frozen factors.
fn apply() {
    let acts = read_f32("/app/data/activations.bin");
    let params = read_f32("/app/scheme/params.bin");
    assert_eq!(params.len(), K, "params.bin is the wrong length");

    // No weights, in either form. The server holds the committed weight side and does the
    // product itself; this step's whole job is to turn the capture in front of it into codes
    // using the factors calibration already fixed. There is nothing here to tune against.
    let factors: Vec<f64> = params.iter().map(|v| *v as f64).collect();
    let (aq, a_scales) = quantize_capture(&acts, &factors);

    fs::create_dir_all("/app/out").unwrap();
    write_i8("/app/out/acts_q.bin", &aq);
    write_f32("/app/out/a_scales.bin", &a_scales);
    println!("served {} rows: wrote two files to /app/out", a_scales.len());
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("calibrate") => calibrate(),
        Some("apply") => apply(),
        other => {
            eprintln!("expected `calibrate` or `apply`, got {:?}", other);
            std::process::exit(2);
        }
    }
}
