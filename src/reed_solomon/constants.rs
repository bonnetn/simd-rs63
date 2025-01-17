use crate::gf8::GF8;

/// Number of data blocks per stripe.
pub const RS_N: usize = 9;

/// Number of data blocks per stripe.
pub const RS_K: usize = 6;

/// Number of parity blocks per stripe.
pub const RS_M: usize = RS_N - RS_K;

const H_MAT: [[GF8; RS_N]; RS_M] = compute_h_matrix();
const H_P_MAT: [[GF8; RS_M]; RS_M] = compute_h_p_matrix();
const H_M_MAT: [[GF8; RS_K]; RS_M] = compute_h_m_matrix();
const P_MAT: [[GF8; RS_K]; RS_M] = matmul(invert_matrix(H_P_MAT), H_M_MAT);
const G_MAT: [[GF8; RS_K]; RS_N] = compute_g();
const COMBINATIONS: [[usize; RS_K]; 84] = compute_combinations();
const FIX_MATRICES: [[[u8; RS_K]; RS_M]; 84] = compute_fix_matrices();

const BITMASK_TO_IDX: [u8; 512] = compute_bitmask_to_idx();

pub fn get_fix_matrix(survivors: &[usize; RS_K]) -> &'static [[u8; RS_K]; RS_M] {
    let mut mask = 0usize;
    for &pos in survivors {
        mask |= 1 << pos;
    }
    let idx = BITMASK_TO_IDX[mask] as usize;
    &FIX_MATRICES[idx]
}

const fn compute_h_matrix() -> [[GF8; RS_N]; RS_M] {
    let mut result = [[GF8::new(0); RS_N]; RS_M];
    let mut i = 0;
    while i < RS_M {
        let mut j = 0;
        while j < RS_N {
            result[i][j] = GF8::new(1);
            let mut k = 0;
            while k < ((i + 1) * j) {
                result[i][j] = result[i][j].mul(GF8::new(2));
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    result
}


const fn compute_h_m_matrix() -> [[GF8; RS_K]; RS_M] {
    let mut result = [[GF8::new(0); RS_K]; RS_M];
    let mut i = 0;
    while i < RS_M {
        let mut j = 0;
        while j < RS_K {
            result[i][j] = H_MAT[i][j];
            j += 1;
        }
        i += 1;
    }
    result
}


const fn compute_h_p_matrix() -> [[GF8; RS_M]; RS_M] {
    let mut result = [[GF8::new(0); RS_M]; RS_M];
    let mut i = 0;
    while i < RS_M {
        let mut j = 0;
        while j < RS_M {
            result[i][j] = H_MAT[i][j + RS_K];
            j += 1;
        }
        i += 1;
    }
    result
}


const fn invert_matrix<const N: usize>(mut a: [[GF8; N]; N]) -> [[GF8; N]; N] {
    let mut inv = [[GF8::new(0); N]; N];
    let mut i = 0;
    while i < N {
        inv[i][i] = GF8::new(1);
        i += 1;
    }
    let mut col = 0;
    while col < N {
        let mut pivot = col;
        while pivot < N && a[pivot][col].value() == 0 {
            pivot += 1;
        }
        if pivot == N {
            panic!("matrix is not invertible");
        }
        if pivot != col {
            let tmp = a[col]; a[col] = a[pivot]; a[pivot] = tmp;
            let tmp = inv[col]; inv[col] = inv[pivot]; inv[pivot] = tmp;
        }
        let pivot_inv = a[col][col].inv();
        let mut j = 0;
        while j < N {
            a[col][j] = a[col][j].mul(pivot_inv);
            inv[col][j] = inv[col][j].mul(pivot_inv);
            j += 1;
        }
        let mut row = 0;
        while row < N {
            if row != col {
                let factor = a[row][col];
                if factor.value() != 0 {
                    let mut j = 0;
                    while j < N {
                        a[row][j] = a[row][j].add(factor.mul(a[col][j]));
                        inv[row][j] = inv[row][j].add(factor.mul(inv[col][j]));
                        j += 1;
                    }
                }
            }
            row += 1;
        }
        col += 1;
    }
    inv
}

const fn matmul<const M: usize, const N: usize, const P: usize>(
    a: [[GF8; N]; M],
    b: [[GF8; P]; N],
) -> [[GF8; P]; M] {
    let mut out = [[GF8::new(0); P]; M];
    let mut i = 0;
    while i < M {
        let mut j = 0;
        while j < P {
            let mut acc = GF8::new(0);
            let mut k = 0;
            while k < N {
                acc = acc.add(a[i][k].mul(b[k][j]));
                k += 1;
            }
            out[i][j] = acc;
            j += 1;
        }
        i += 1;
    }
    out
}


const fn compute_g() -> [[GF8; RS_K]; RS_N] {
    let mut result = [[GF8::new(0); RS_K]; RS_N];
    let mut i = 0;
    while i < RS_N {
        let mut j = 0;
        while j < RS_K {
            result[i][j] = if i < RS_K {
                GF8::new(if i == j { 1 } else { 0 })
            } else {
                P_MAT[i - RS_K][j]
            };
            j += 1;
        }
        i += 1;
    }
    result
}


const fn select_rows<const N: usize, const M: usize, const K: usize>(
    matrix: [[GF8; M]; N],
    rows: [usize; K],
) -> [[GF8; M]; K] {
    let mut out = [[GF8::new(0); M]; K];
    let mut i = 0;
    while i < K {
        let row = rows[i];
        if row >= N {
            panic!("row index out of bounds");
        }
        out[i] = matrix[row];
        i += 1;
    }
    out
}

const fn compute_fix_matrix(indices: [usize; RS_K]) -> [[GF8; RS_K]; RS_N] {
    matmul(G_MAT, invert_matrix(select_rows(G_MAT, indices)))
}

const fn compute_combinations() -> [[usize; RS_K]; 84] {
    let mut out = [[0_usize; RS_K]; 84];
    let mut idx = 0;
    let mut a = 0;
    while a <= 3 {
        let mut b = a + 1;
        while b <= 4 {
            let mut c = b + 1;
            while c <= 5 {
                let mut d = c + 1;
                while d <= 6 {
                    let mut e = d + 1;
                    while e <= 7 {
                        let mut f = e + 1;
                        while f <= 8 {
                            out[idx] = [a, b, c, d, e, f];
                            idx += 1;
                            f += 1;
                        }
                        e += 1;
                    }
                    d += 1;
                }
                c += 1;
            }
            b += 1;
        }
        a += 1;
    }
    out
}


const fn compute_fix_matrices() -> [[[u8; RS_K]; RS_M]; 84] {
    let mut result = [[[0u8; RS_K]; RS_M]; 84];
    let mut i = 0;
    while i < COMBINATIONS.len() {
        let combo = COMBINATIONS[i];
        let mut mask = [true; RS_N];
        let mut j = 0;
        while j < RS_K {
            mask[combo[j]] = false;
            j += 1;
        }
        let mut error_indices = [0usize; RS_M];
        let mut count = 0;
        let mut j = 0;
        while j < RS_N {
            if mask[j] {
                error_indices[count] = j;
                count += 1;
            }
            j += 1;
        }
        let full = compute_fix_matrix(combo);
        let mut r = 0;
        while r < RS_M {
            let mut c = 0;
            while c < RS_K {
                result[i][r][c] = full[error_indices[r]][c].value();
                c += 1;
            }
            r += 1;
        }
        i += 1;
    }
    result
}


const fn compute_bitmask_to_idx() -> [u8; 512] {
    let mut result = [u8::MAX; 512];
    let mut i = 0;
    while i < COMBINATIONS.len() {
        let mut mask = 0usize;
        let mut j = 0;
        while j < RS_K {
            mask |= 1 << COMBINATIONS[i][j];
            j += 1;
        }
        result[mask] = i as u8;
        i += 1;
    }
    result
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_inv_and_mul() {
        let result = matmul(H_P_MAT, invert_matrix(H_P_MAT));
        let expected = [
            [GF8::new(1), GF8::new(0), GF8::new(0)],
            [GF8::new(0), GF8::new(1), GF8::new(0)],
            [GF8::new(0), GF8::new(0), GF8::new(1)],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_select_first_rows_of_g() {
        let mat = select_rows(G_MAT, [0, 1, 2, 3, 4, 5]);
        let expected = [
            [GF8::new(1), GF8::new(0), GF8::new(0), GF8::new(0), GF8::new(0), GF8::new(0)],
            [GF8::new(0), GF8::new(1), GF8::new(0), GF8::new(0), GF8::new(0), GF8::new(0)],
            [GF8::new(0), GF8::new(0), GF8::new(1), GF8::new(0), GF8::new(0), GF8::new(0)],
            [GF8::new(0), GF8::new(0), GF8::new(0), GF8::new(1), GF8::new(0), GF8::new(0)],
            [GF8::new(0), GF8::new(0), GF8::new(0), GF8::new(0), GF8::new(1), GF8::new(0)],
            [GF8::new(0), GF8::new(0), GF8::new(0), GF8::new(0), GF8::new(0), GF8::new(1)],
        ];
        assert_eq!(mat, expected);
    }

    #[test]
    fn test_select_last_rows_of_g() {
        let mat = select_rows(G_MAT, [6, 7, 8]);
        assert_eq!(mat, P_MAT);
    }

    const fn col<const N: usize>(values: [u8; N]) -> [[GF8; 1]; N] {
        let mut result = [[GF8::new(0); 1]; N];
        let mut i = 0;
        while i < N {
            result[i][0] = GF8::new(values[i]);
            i += 1;
        }
        result
    }

    #[test]
    fn test_compute_fix_matrix() {
        let m = col([0, 1, 2, 3, 4, 5]);
        let c = matmul(G_MAT, m);
        for i in 0..6 {
            assert_eq!(c[i], m[i]);
        }
        let mut survivors = [[GF8::new(0)]; RS_K];
        let indices = [3, 4, 5, 6, 7, 8];
        for (idx, i) in indices.iter().enumerate() {
            survivors[idx][0] = c[*i][0];
        }
        let fix_m = compute_fix_matrix([3, 4, 5, 6, 7, 8]);
        let fixed_c = matmul(fix_m, survivors);
        assert_eq!(fixed_c, c);
    }

    #[test]
    fn test_combinations() {
        for v in COMBINATIONS {
            for (prev, next) in v.iter().zip(v.iter().skip(1)) {
                assert!(*prev < *next);
                assert!(*prev < 9);
                assert!(*next < 9);
            }
        }
        let h = HashSet::from(COMBINATIONS);
        assert_eq!(h.len(), COMBINATIONS.len());
        let mut sorted = COMBINATIONS;
        sorted.sort();
        assert_eq!(COMBINATIONS, sorted);
    }

    #[test]
    fn test_get_fix_matrix() {
        let got = get_fix_matrix(&[2, 4, 5, 6, 7, 8]);
        assert_eq!(got, &[[222, 29, 63, 197, 80, 19], [4, 160, 147, 163, 77, 152], [93, 10, 138, 210, 78, 238]]);
    }
}
