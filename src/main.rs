use std::os::unix::thread;

use rand::{thread_rng, Rng};

fn decomposer(mut value: u64, q: u64, d: usize, logb: u64) -> Vec<u64> {
    let b = 1u64 << logb;
    let full_mask = b - 1u64;
    let bby2 = b >> 1;

    if value >= (q >> 1) {
        value = !(q - value) + 1;
    }

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

    let mut out = Vec::with_capacity(d);
    for _ in 0..d {
        let k_i = value & full_mask;
        value = (value - k_i) >> logb;

        if k_i > bby2 || (k_i == bby2 && ((value & 1) == 1)) {
            // if (k_i == bby2 && ((value & 1) == 1)) {
            //     println!("AA");
            // }
            out.push(q - (b - k_i));
            value += 1;
        } else {
            // if (k_i == bby2) {
            //     println!("BB");
            // }
            out.push(k_i);
        }
    }

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

    let q = 36028797018820609u64;
    let logb = 11;
    let d = 5;

    for _ in 0..100000 {
        let value = rng.gen_range(0..q);
        let limbs = decomposer(value, q, d, logb);
        // println!("{:?}", &limbs);
        let value_back = recompose(&limbs, q, logb);
        assert_eq!(value, value_back)
    }
}
