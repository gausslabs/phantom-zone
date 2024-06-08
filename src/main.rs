use std::os::unix::thread;

use bin_rs::{Ntt, NttBackendU64, NttInit};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rand_distr::Uniform;

// fn decomposer(mut value: u64, q: u64, d: usize, logb: u64) -> Vec<u64> {
//     let b = 1u64 << logb;
//     let full_mask = b - 1u64;
//     let bby2 = b >> 1;

//     if value >= (q >> 1) {
//         value = !(q - value) + 1;
//     }

//     // let mut carry = 0;
//     // let mut out = Vec::with_capacity(d);
//     // for _ in 0..d {
//     //     let k_i = carry + (value & full_mask);
//     //     value = (value) >> logb;
//     //     if k_i > bby2 {
//     //         // if (k_i == bby2 && ((value & 1) == 1)) {
//     //         //     println!("AA");
//     //         // }
//     //         out.push(q - (b - k_i));
//     //         carry = 1;
//     //     } else {
//     //         // if (k_i == bby2) {
//     //         //     println!("BB");
//     //         // }
//     //         out.push(k_i);
//     //         carry = 0;
//     //     }
//     // }
//     // return out;

//     let mut out = Vec::with_capacity(d);
//     for _ in 0..d {
//         let k_i = value & full_mask;
//         value = (value - k_i) >> logb;

//         if k_i > bby2 || (k_i == bby2 && ((value & 1) == 1)) {
//             // if (k_i == bby2 && ((value & 1) == 1)) {
//             //     println!("AA");
//             // }
//             out.push(q - (b - k_i));
//             value += 1;
//         } else {
//             // if (k_i == bby2) {
//             //     println!("BB");
//             // }
//             out.push(k_i);
//         }
//     }

//     return out;
// }

// fn recompose(limbs: &[u64], q: u64, logb: u64) -> u64 {
//     let mut out = 0;
//     limbs.iter().enumerate().for_each(|(i, l)| {
//         let a = 1u128 << (logb * (i as u64));
//         let a = ((a * (*l as u128)) % (q as u128)) as u64;
//         out = (out + a) % q;
//     });
//     out % q
// }

pub(crate) fn forward_matrix(matrix_a: &mut [Vec<u64>], ntt_op: &NttBackendU64) {
    matrix_a.iter_mut().for_each(|r| {
        ntt_op.forward(r.as_mut_slice());
    });
}

fn main() {
    // let mut v = Vec::with_capacity(10);
    // v[0] = 1;
    // println!("Hello, world!");

    let prime = 36028797018820609u64;
    let ring_size = 1 << 11;
    let mut rng = thread_rng();
    let distr = Uniform::new(0, prime);

    let ntt = NttBackendU64::new(&prime, ring_size);
    let iterations = 1000;

    // Single Loop
    let now = std::time::Instant::now();
    let mut a = (&mut rng).sample_iter(distr).take(ring_size).collect_vec();
    for _ in 0..iterations {
        ntt.forward(a.as_mut_slice());
    }
    println!("Single Loop: {:?}", now.elapsed());

    // Multiple Loop
    // let now = std::time::Instant::now();
    // let d = 2;
    // let mut a_mat = (0..d)
    //     .into_iter()
    //     .map(|_| (&mut rng).sample_iter(distr).take(ring_size).collect_vec())
    //     .collect_vec();
    // for _ in 0..iterations {
    //     forward_matrix(&mut a_mat, &ntt);
    // }
    // println!("Multiple Loop: {:?}", now.elapsed());
}
