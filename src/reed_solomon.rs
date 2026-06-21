mod constants;
mod lanes;

use std::simd::Simd;
use crate::{gf8::{gf8_simd_mul, gf8_simd_mul_xor}, reed_solomon::constants::get_fix_matrix};

pub(crate) use crate::reed_solomon::lanes::LANES;

use constants::{RS_K, RS_M, RS_N};


pub fn fix_errors<'a, 'b, const N: usize>(
    mut survivors: [(usize, &'a [u8]); RS_K],
    mut to_fix: [(usize, &'b mut [u8]); N],
) {
    if to_fix.is_empty() {
        return;
    }

    survivors.sort_by_key(|(position, _)| *position);
    to_fix.sort_by_key(|(position, _)| *position);

    let survivor_indices = get_indices(&survivors);
    let to_fix_indices = get_indices(&to_fix);

    let fix_matrix = get_fix_matrix(&survivor_indices);

    let fix_matrix_indices = fix_matrix_indices(&survivor_indices, &to_fix_indices);
    let fix_matrix: &[[u8; RS_K]; N] = &std::array::from_fn(|i| fix_matrix[fix_matrix_indices[i]]);

    let n_chunks = survivors[0].1.len() / LANES;
    let s_chunks: [&[[u8; LANES]]; RS_K] =
        std::array::from_fn(|i| survivors[i].1.as_chunks::<LANES>().0);
    let f_chunks: [&mut [[u8; LANES]]; N] =
        to_fix.map(|(_, s)| s.as_chunks_mut::<LANES>().0);

    for i in 1..RS_K {
        debug_assert_eq!(s_chunks[i].len(), n_chunks);
    }
    for j in 0..N {
        debug_assert_eq!(f_chunks[j].len(), n_chunks);
    }

    for chunk_idx in 0..n_chunks {
        let survivor_simd: [Simd<u8, LANES>; RS_K] =
            std::array::from_fn(|i| Simd::from_array(s_chunks[i][chunk_idx]));

        for j in 0..N {
            let mut result = gf8_simd_mul(survivor_simd[0], fix_matrix[j][0]);
            for k in 1..RS_K {
                gf8_simd_mul_xor(&mut result, survivor_simd[k], fix_matrix[j][k]);
            }
            f_chunks[j][chunk_idx] = result.to_array();
        }
    }
}

fn get_indices<T, const N: usize>(v: &[(usize, T); N]) -> [usize; N] {
    std::array::from_fn(|i| v[i].0)
}

fn fix_matrix_indices<const N: usize>(survivor_pos: &[usize; RS_K], to_fix_pos: &[usize; N]) -> [usize; N] {
    let mut pos_to_idx = [0_usize; RS_N];
    for pos in survivor_pos {
        pos_to_idx[*pos] = usize::MAX;
    }

    let mut count = 0;
    for v in &mut pos_to_idx {
        if *v == usize::MAX {
            continue;
        }
        *v = count;
        count += 1;
    }

    let mut result = [0; N];
    for (i, p) in to_fix_pos.iter().enumerate() {
        result[i] = pos_to_idx[*p];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_errors() {
        const SIZE: usize = 1024;
        let a = vec![0_u8; SIZE];
        let b = vec![1_u8; SIZE];
        let c = vec![2_u8; SIZE];
        let d = vec![3_u8; SIZE];
        let e = vec![4_u8; SIZE];
        let f = vec![5_u8; SIZE];
        let mut g = vec![0_u8; SIZE];
        let mut h = vec![0_u8; SIZE];
        let mut i = vec![0_u8; SIZE];
        fix_errors::<3>(
            [(0, &a), (1, &b), (2, &c), (3, &d), (4, &e), (5, &f)],
            [(6, &mut g), (7, &mut h), (8, &mut i)],
        );

        {
            let mut d_fixed = vec![0_u8; SIZE];
            let mut e_fixed = vec![0_u8; SIZE];
            let mut f_fixed = vec![0_u8; SIZE];
            fix_errors::<3>(
                [(0, &a), (1, &b), (2, &c), (6, &g), (7, &h), (8, &i)],
                [(3, &mut d_fixed), (4, &mut e_fixed), (5, &mut f_fixed)],
            );

            assert_eq!(d, d_fixed);
            assert_eq!(e, e_fixed);
            assert_eq!(f, f_fixed);
        }

        {
            let mut d_fixed = vec![0_u8; SIZE];
            let mut f_fixed = vec![0_u8; SIZE];
            fix_errors::<2>(
                [(0, &a), (1, &b), (2, &c), (6, &g), (7, &h), (8, &i)],
                [(3, &mut d_fixed), (5, &mut f_fixed)],
            );

            assert_eq!(d, d_fixed);
            assert_eq!(f, f_fixed);
        }

        {
            let mut d_fixed = vec![0_u8; SIZE];
            fix_errors::<1>(
                [(0, &a), (1, &b), (2, &c), (6, &g), (7, &h), (8, &i)],
                [(3, &mut d_fixed)],
            );

            assert_eq!(d, d_fixed);
        }
    }
}
