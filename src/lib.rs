#![feature(portable_simd)]
#![deny(missing_docs)]

//! Reed-Solomon erasure coding over GF(2⁸).
//!
//! This crate implements a systematic RS(9, 6) code: stripes of [`N`] = 9 equal-sized blocks,
//! of which [`K`] = 6 carry data and [`M`] = 3 are parity. Any combination of up to [`M`]
//! lost blocks can be recovered from the remaining [`K`] blocks.
//!
//! # Quick start
//!
//! ```
//! use reed_solomon::{encode, recover, K, M, BLOCK_ALIGNMENT};
//!
//! let block_size = 4 * BLOCK_ALIGNMENT;
//!
//! // Build K data blocks.
//! let data: Vec<Vec<u8>> = (0..K).map(|i| vec![i as u8; block_size]).collect();
//!
//! // Encode: compute M parity blocks.
//! let mut parity: Vec<Vec<u8>> = vec![vec![0u8; block_size]; M];
//! encode(
//!     std::array::from_fn(|i| data[i].as_slice()),
//!     std::array::from_fn(|i| parity[i].as_mut_slice()),
//! ).unwrap();
//!
//! // Simulate losing data blocks 3, 4 and 5.
//! let mut r3 = vec![0u8; block_size];
//! let mut r4 = vec![0u8; block_size];
//! let mut r5 = vec![0u8; block_size];
//!
//! // Recover them from the surviving blocks.
//! recover(
//!     [(0, data[0].as_slice()), (1, data[1].as_slice()), (2, data[2].as_slice()),
//!      (6, parity[0].as_slice()), (7, parity[1].as_slice()), (8, parity[2].as_slice())],
//!     [(3, &mut r3), (4, &mut r4), (5, &mut r5)],
//! ).unwrap();
//!
//! assert_eq!(r3, data[3]);
//! assert_eq!(r4, data[4]);
//! assert_eq!(r5, data[5]);
//! ```
//!
//! # Block sizes
//!
//! All blocks in a call must be the same size, and that size must be a positive multiple of
//! [`BLOCK_ALIGNMENT`]. On this platform [`BLOCK_ALIGNMENT`] is chosen to match the widest
//! SIMD shuffle instruction available, so it varies by CPU (16 on NEON/SSSE3, 32 on AVX2,
//! 64 on AVX-512 VBMI).

mod gf8;
mod reed_solomon;
mod error;

use reed_solomon::LANES;

/// Total number of blocks per stripe (data + parity).
pub use crate::reed_solomon::{RS_N as N};

/// Number of data blocks per stripe.
pub use crate::reed_solomon::{RS_K as K};

/// Number of parity blocks per stripe.
pub use crate::reed_solomon::{RS_M as M};

/// Required alignment for block sizes.
///
/// Every block passed to [`encode`] or [`recover`] must have a length that is a positive
/// multiple of this value. On this platform it equals the width of the widest SIMD shuffle
/// instruction available (16, 32, or 64 bytes).
pub const BLOCK_ALIGNMENT: usize = LANES;

pub use error::Error;

/// Computes the [`M`] parity blocks from [`K`] data blocks.
///
/// Data blocks are assigned indices `0..K` and parity blocks indices `K..N`. All blocks
/// must have the same length, which must be a positive multiple of [`BLOCK_ALIGNMENT`].
///
/// # Errors
///
/// - [`Error::InvalidBlockSize`] — block length is zero or not a multiple of
///   [`BLOCK_ALIGNMENT`].
/// - [`Error::BlockSizeMismatch`] — not all blocks have the same length.
///
/// # Example
///
/// ```
/// use reed_solomon::{encode, BLOCK_ALIGNMENT};
///
/// let block_size = 4 * BLOCK_ALIGNMENT;
///
/// let data = [
///     vec![0; block_size],
///     vec![1; block_size],
///     vec![2; block_size],
///     vec![3; block_size],
///     vec![4; block_size],
///     vec![5; block_size],
/// ];
///
/// let mut parity = [
///     vec![0; block_size],
///     vec![0; block_size],
///     vec![0; block_size],
/// ];
///
/// let data_shards = [
///     data[0].as_slice(),
///     data[1].as_slice(),
///     data[2].as_slice(),
///     data[3].as_slice(),
///     data[4].as_slice(),
///     data[5].as_slice(),
/// ];
///
/// let parity_shards = [
///     parity[0].as_mut_slice(),
///     parity[1].as_mut_slice(),
///     parity[2].as_mut_slice(),
/// ];
///
/// encode(data_shards, parity_shards).unwrap();
/// ```
pub fn encode(data: [&[u8]; K], parity: [&mut [u8]; M]) -> Result<(), Error> {
    let block_size = data[0].len();
    validate_block_size(block_size)?;
    for slice in data.iter().skip(1) {
        if slice.len() != block_size {
            return Err(Error::BlockSizeMismatch { expected: block_size, got: slice.len() });
        }
    }
    for slice in &parity {
        if slice.len() != block_size {
            return Err(Error::BlockSizeMismatch { expected: block_size, got: slice.len() });
        }
    }

    let survivors: [(usize, &[u8]); K] = std::array::from_fn(|i| (i, data[i]));
    let mut j = K;
    let to_fix: [(usize, &mut [u8]); M] = parity.map(|s| {
        let idx = j;
        j += 1;
        (idx, s)
    });

    reed_solomon::fix_errors(survivors, to_fix);
    Ok(())
}

/// Recovers up to [`M`] missing blocks from any [`K`] known blocks.
///
/// `known` contains exactly [`K`] blocks, each paired with its stripe index in `0..N`.
/// `missing` pairs output buffers with the indices of the blocks to recover. All blocks
/// must have the same length, which must be a positive multiple of [`BLOCK_ALIGNMENT`].
/// Indices across `known` and `missing` must be distinct.
///
/// The number of missing blocks `MISSING` must be at most [`M`]; this is enforced at
/// compile time.
///
/// # Errors
///
/// - [`Error::InvalidBlockSize`] — block length is zero or not a multiple of
///   [`BLOCK_ALIGNMENT`].
/// - [`Error::BlockSizeMismatch`] — not all blocks have the same length.
/// - [`Error::IndexOutOfRange`] — an index is ≥ [`N`].
/// - [`Error::DuplicateIndex`] — an index appears more than once.
///
/// # Example
///
/// ```
/// use reed_solomon::{encode, recover, K, M, BLOCK_ALIGNMENT};
///
/// let block_size = 4 * BLOCK_ALIGNMENT;
///
/// let data: Vec<Vec<u8>> = (0..K)
///     .map(|i| vec![i as u8; block_size])
///     .collect();
///
/// let mut parity: Vec<Vec<u8>> = vec![vec![0; block_size]; M];
///
/// let data_refs = [
///     data[0].as_slice(),
///     data[1].as_slice(),
///     data[2].as_slice(),
///     data[3].as_slice(),
///     data[4].as_slice(),
///     data[5].as_slice(),
/// ];
///
/// let parity_refs = [
///     parity[0].as_mut_slice(),
///     parity[1].as_mut_slice(),
///     parity[2].as_mut_slice(),
/// ];
///
/// encode(data_refs, parity_refs).unwrap();
///
/// // Recover data shard 0 using data shards 1..=5 and parity shard 0.
/// // Shard indexes 0..K are data shards, and K..K+M are parity shards.
/// let available = [
///     (1, data[1].as_slice()),
///     (2, data[2].as_slice()),
///     (3, data[3].as_slice()),
///     (4, data[4].as_slice()),
///     (5, data[5].as_slice()),
///     (K, parity[0].as_slice()),
/// ];
///
/// let mut recovered = vec![0; block_size];
/// let missing = [(0, recovered.as_mut_slice())];
///
/// recover(available, missing).unwrap();
///
/// assert_eq!(recovered, data[0]);
/// ```
pub fn recover<const MISSING: usize>(
    known: [(usize, &[u8]); K],
    missing: [(usize, &mut [u8]); MISSING],
) -> Result<(), Error> {
    const { assert!(MISSING <= M, "cannot recover more than M blocks at once") };

    if MISSING == 0 {
        return Ok(());
    }

    let block_size = known[0].1.len();
    validate_block_size(block_size)?;

    let mut seen = 0u16;
    for &(idx, slice) in &known {
        if idx >= N {
            return Err(Error::IndexOutOfRange(idx));
        }
        let bit = 1u16 << idx;
        if seen & bit != 0 {
            return Err(Error::DuplicateIndex(idx));
        }
        seen |= bit;
        if slice.len() != block_size {
            return Err(Error::BlockSizeMismatch { expected: block_size, got: slice.len() });
        }
    }
    for (idx, slice) in &missing {
        if *idx >= N {
            return Err(Error::IndexOutOfRange(*idx));
        }
        let bit = 1u16 << *idx;
        if seen & bit != 0 {
            return Err(Error::DuplicateIndex(*idx));
        }
        seen |= bit;
        if slice.len() != block_size {
            return Err(Error::BlockSizeMismatch { expected: block_size, got: slice.len() });
        }
    }

    reed_solomon::fix_errors(known, missing);
    Ok(())
}

fn validate_block_size(size: usize) -> Result<(), Error> {
    if size == 0 || size % BLOCK_ALIGNMENT != 0 {
        return Err(Error::InvalidBlockSize(size));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: usize = 4 * BLOCK_ALIGNMENT;

    fn make_data() -> [Vec<u8>; K] {
        std::array::from_fn(|i| vec![i as u8; SIZE])
    }

    fn encode_stripe(data: &[Vec<u8>; K]) -> [Vec<u8>; M] {
        let mut parity: [Vec<u8>; M] = std::array::from_fn(|_| vec![0u8; SIZE]);
        let [p0, p1, p2] = &mut parity;
        encode(
            std::array::from_fn(|i| data[i].as_slice()),
            [p0.as_mut_slice(), p1.as_mut_slice(), p2.as_mut_slice()],
        )
        .unwrap();
        parity
    }

    #[test]
    fn test_encode_recover_all_parity() {
        let data = make_data();
        let parity = encode_stripe(&data);

        let mut r0 = vec![0u8; SIZE];
        let mut r1 = vec![0u8; SIZE];
        let mut r2 = vec![0u8; SIZE];
        recover(
            std::array::from_fn(|i| (i, data[i].as_slice())),
            [(K, &mut r0), (K + 1, &mut r1), (K + 2, &mut r2)],
        )
        .unwrap();

        assert_eq!(r0, parity[0]);
        assert_eq!(r1, parity[1]);
        assert_eq!(r2, parity[2]);
    }

    #[test]
    fn test_recover_3_data_blocks() {
        let data = make_data();
        let [p0, p1, p2] = encode_stripe(&data);

        let mut r3 = vec![0u8; SIZE];
        let mut r4 = vec![0u8; SIZE];
        let mut r5 = vec![0u8; SIZE];
        recover(
            [
                (0, data[0].as_slice()),
                (1, data[1].as_slice()),
                (2, data[2].as_slice()),
                (6, p0.as_slice()),
                (7, p1.as_slice()),
                (8, p2.as_slice()),
            ],
            [(3, &mut r3), (4, &mut r4), (5, &mut r5)],
        )
        .unwrap();

        assert_eq!(r3, data[3]);
        assert_eq!(r4, data[4]);
        assert_eq!(r5, data[5]);
    }

    #[test]
    fn test_recover_mixed_data_and_parity() {
        let data = make_data();
        let [p0, p1, p2] = encode_stripe(&data);

        let mut r0 = vec![0u8; SIZE];
        let mut r6 = vec![0u8; SIZE];
        let mut r8 = vec![0u8; SIZE];
        recover(
            [
                (1, data[1].as_slice()),
                (2, data[2].as_slice()),
                (3, data[3].as_slice()),
                (4, data[4].as_slice()),
                (5, data[5].as_slice()),
                (7, p1.as_slice()),
            ],
            [(0, &mut r0), (6, &mut r6), (8, &mut r8)],
        )
        .unwrap();

        assert_eq!(r0, data[0]);
        assert_eq!(r6, p0);
        assert_eq!(r8, p2);
    }

    #[test]
    fn test_error_invalid_block_size_zero() {
        let data: [Vec<u8>; K] = std::array::from_fn(|_| vec![]);
        let mut p0 = vec![];
        let mut p1 = vec![];
        let mut p2 = vec![];
        assert_eq!(
            encode(
                std::array::from_fn(|i| data[i].as_slice()),
                [p0.as_mut_slice(), p1.as_mut_slice(), p2.as_mut_slice()],
            ),
            Err(Error::InvalidBlockSize(0))
        );
    }

    #[test]
    fn test_error_invalid_block_size_unaligned() {
        let sz = BLOCK_ALIGNMENT + 1;
        let data: [Vec<u8>; K] = std::array::from_fn(|_| vec![0u8; sz]);
        let mut p0 = vec![0u8; sz];
        let mut p1 = vec![0u8; sz];
        let mut p2 = vec![0u8; sz];
        assert_eq!(
            encode(
                std::array::from_fn(|i| data[i].as_slice()),
                [p0.as_mut_slice(), p1.as_mut_slice(), p2.as_mut_slice()],
            ),
            Err(Error::InvalidBlockSize(sz))
        );
    }

    #[test]
    fn test_error_block_size_mismatch() {
        let data: [Vec<u8>; K] =
            std::array::from_fn(|i| vec![0u8; if i == 3 { 2 * BLOCK_ALIGNMENT } else { SIZE }]);
        let mut p0 = vec![0u8; SIZE];
        let mut p1 = vec![0u8; SIZE];
        let mut p2 = vec![0u8; SIZE];
        assert_eq!(
            encode(
                std::array::from_fn(|i| data[i].as_slice()),
                [p0.as_mut_slice(), p1.as_mut_slice(), p2.as_mut_slice()],
            ),
            Err(Error::BlockSizeMismatch { expected: SIZE, got: 2 * BLOCK_ALIGNMENT })
        );
    }

    #[test]
    fn test_error_index_out_of_range() {
        let data = make_data();
        let [p0, _, _] = encode_stripe(&data);
        let mut r = vec![0u8; SIZE];
        assert_eq!(
            recover(
                [
                    (0, data[0].as_slice()),
                    (1, data[1].as_slice()),
                    (2, data[2].as_slice()),
                    (3, data[3].as_slice()),
                    (4, data[4].as_slice()),
                    (9, p0.as_slice()),
                ],
                [(5, &mut r)],
            ),
            Err(Error::IndexOutOfRange(9))
        );
    }

    #[test]
    fn test_error_duplicate_index() {
        let data = make_data();
        let [p0, _, _] = encode_stripe(&data);
        let mut r = vec![0u8; SIZE];
        assert_eq!(
            recover(
                [
                    (0, data[0].as_slice()),
                    (1, data[1].as_slice()),
                    (2, data[2].as_slice()),
                    (3, data[3].as_slice()),
                    (4, data[4].as_slice()),
                    (4, p0.as_slice()),
                ],
                [(5, &mut r)],
            ),
            Err(Error::DuplicateIndex(4))
        );
    }
}
