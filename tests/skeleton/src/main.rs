// int8 path for the dense layer.
//
// Two steps, selected by the first argument.
//
//   calibrate  reads /app/data/weights.bin and /app/data/activations.bin and is supposed to
//              write the six files described in the task into /app/out.
//   apply      reads /app/data/activations.bin and the committed factors in
//              /app/scheme/params.bin, and is supposed to write acts_q.bin and a_scales.bin
//              into /app/out. There are no weights on the box during this step: the server
//              holds the committed weight side and does the product itself.
//
// Neither is implemented yet.

const K: usize = 512; // inner dimension
const N: usize = 192; // output columns

fn calibrate() {
    eprintln!("calibrate not implemented ({} by {})", K, N);
    std::process::exit(1);
}

fn apply() {
    eprintln!("apply not implemented ({} by {})", K, N);
    std::process::exit(1);
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
