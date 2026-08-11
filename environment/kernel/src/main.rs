// int8 path for the dense layer.
//
// Reads /app/data/activations.bin and /app/data/weights.bin and is supposed to write the
// five files described in the task into /app/out. Not implemented yet.

const M: usize = 96; // activation rows
const K: usize = 512; // inner dimension
const N: usize = 192; // output columns

fn main() {
    eprintln!("int8 kernel not implemented ({}x{} by {}x{})", M, K, K, N);
    std::process::exit(1);
}
