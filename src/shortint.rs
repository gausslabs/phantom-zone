use itertools::izip;

use crate::Matrix;

struct FheUint8<M: Matrix> {
    data: M,
}

fn add<M: Matrix>(a: FheUint8<M>, b: FheUint8<M>) {
    // CALL THE EVALUATOR
    izip!(a.data.iter_rows(), b.data.iter_rows()).for_each(|(a_bit, b_bit)| {
        // A ^ B
    });
}
