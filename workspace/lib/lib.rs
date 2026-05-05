use std::ops::Range;

pub mod mipt;

/// Return the range of qubits in the reverse light cone of a target measurement
/// on qubit index `x` (out of `n` total qubits) occurring `dt` layers from
/// layer `t`.
pub fn rev_cone_range(n: usize, x: usize, t: usize, dt: usize) -> Range<usize> {
    if (t + x).is_multiple_of(2) {
        let start = x.saturating_sub(2 * (dt / 2));
        let end = (x + 2 * dt.div_ceil(2)).min(n);
        start .. end
    } else {
        let start = x.saturating_sub(2 * dt.div_ceil(2)) + 1;
        let end = (x + 2 * (dt / 2) + 1).min(n);
        start .. end
    }
}

