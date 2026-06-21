use crate::{BLOCK_ALIGNMENT, N};

/// Errors returned by [`encode`] and [`recover`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// Block size is zero or not a multiple of [`BLOCK_ALIGNMENT`].
    InvalidBlockSize(usize),

    /// Not all blocks have the same length. `expected` is the length of the first block.
    BlockSizeMismatch {
        /// The length inferred from the first block.
        expected: usize,
        /// The mismatched length.
        got: usize,
    },

    /// The same block index appeared more than once across the inputs.
    DuplicateIndex(usize),

    /// A block index is out of range; valid indices are `0..N`.
    IndexOutOfRange(usize),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidBlockSize(size) => write!(
                f,
                "block size {size} is not a positive multiple of {BLOCK_ALIGNMENT}"
            ),
            Error::BlockSizeMismatch { expected, got } => write!(
                f,
                "block size mismatch: expected {expected} bytes, got {got} bytes"
            ),
            Error::DuplicateIndex(idx) => write!(f, "duplicate block index {idx}"),
            Error::IndexOutOfRange(idx) => {
                write!(f, "block index {idx} is out of range (must be less than {N})")
            }
        }
    }
}

impl std::error::Error for Error {}
