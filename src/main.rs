use std::os::unix::thread;

use bin_rs::Stats;
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rand_distr::Uniform;

fn decomposer(mut value: u64, q: u64, d: usize, logb: u64) -> Vec<u64> {
    let b = 1u64 << logb;
    let full_mask = b - 1u64;
    let bby2 = b >> 1;

    // if value >= (q >> 1) {
    //     value = !(q - value) + 1;
    // }

    // let mut carry = 0;
    // let mut out = Vec::with_capacity(d);
    // for _ in 0..d {
    //     let k_i = carry + (value & full_mask);
    //     value = (value) >> logb;
    //     let go = thread_rng().gen_bool(1.0 / 2.0);
    //     if k_i > bby2 || (k_i == bby2 && ((value & 1) == 1)) {
    //         // if (k_i == bby2 && ((value & 1) == 1)) {
    //         //     println!("AA");
    //         // }
    //         out.push(q - (b - k_i));
    //         carry = 1;
    //     } else {
    //         // if (k_i == bby2) {
    //         //     println!("BB");
    //         // }
    //         out.push(k_i);
    //         carry = 0;
    //     }
    // }
    // println!("Last carry {carry}");
    // return out;

    // ------------------ Signed decomposition
    // let mut out = Vec::with_capacity(d);

    // if value >= (q >> 1) {
    //     value = !(q - value) + 1;
    // }
    // for i in 0..d {
    //     let k_i = value & full_mask;
    //     value = (value - k_i) >> logb;

    //     if k_i >= bby2 {
    //         out.push(q - (b - k_i));
    //         value += 1;
    //     } else {
    //         out.push(k_i);
    //     }
    // }

    // ------------------ METHOD WE USR CURRENTLY (Signed Balanced decomposition)
    // let mut out = Vec::with_capacity(d);

    // if value >= (q >> 1) {
    //     value = !(q - value) + 1;
    // }
    // for i in 0..d {
    //     let k_i = value & full_mask;
    //     value = (value - k_i) >> logb;

    //     if k_i > bby2 || ((k_i == bby2) && ((value & 1) == 1)) {
    //         out.push(q - (b - k_i));
    //         value += 1;
    //     } else {
    //         out.push(k_i);
    //     }
    // }

    // ------------------ BRIAN's Method --------------------
    // let mut is_neg = false;
    // if value >= (q >> 1) {
    //     value = !(q - value) + 1;
    //     is_neg = true;
    // }

    // let mut out = vec![];

    // for i in 0..d {
    //     let k_i = value & full_mask;
    //     value = (value - k_i) >> logb;

    //     if (k_i > bby2 && i == d - 1) || (i == d - 1 && is_neg) {
    //         out.push(q - (b - k_i));
    //         value += 1;
    //     } else {
    //         out.push(k_i);
    //     }
    // }

    return out;
}

fn recompose(limbs: &[u64], q: u64, logb: u64) -> u64 {
    let mut out = 0;
    limbs.iter().enumerate().for_each(|(i, l)| {
        let a = 1u128 << (logb * (i as u64));
        let a = ((a * (*l as u128)) % (q as u128)) as u64;
        out = (out + a) % q;
    });
    out % q
}

fn main() {
    // let mut v = Vec::with_capacity(10);
    // v[0] = 1;
    // println!("Hello, world!");

    let mut rng = thread_rng();

    let mut stats = Stats::new();

    for _ in 0..1000 {
        let q = rng.sample(Uniform::new((1 << 54) + 1, 1 << 55));

        let logb = 11;
        let d = 5;

        for j in 0..100000 {
            let value = rng.gen_range(0..q);
            let limbs = decomposer(value, q, d, logb);
            // println!("{:?}", &limbs);
            let value_back = recompose(&limbs, q, logb);
            assert_eq!(value, value_back);

            stats.add_more(
                &limbs
                    .iter()
                    .map(|v| {
                        if *v > (q >> 1) {
                            -((q - v) as i64)
                        } else {
                            *v as i64
                        }
                    })
                    .collect_vec(),
            );
        }
    }

    println!("Mean: {}", stats.mean());
    println!("Std : {}", stats.std_dev().abs().log2());
}
