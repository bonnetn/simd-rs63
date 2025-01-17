// Optimal SIMD lane count based on what swizzle_dyn can vectorize.
// See: https://doc.rust-lang.org/src/core/portable-simd/crates/core_simd/src/swizzle_dyn.rs.html
#[cfg(all(target_feature = "avx512vl", target_feature = "avx512vbmi"))]
pub const LANES: usize = 64;

#[cfg(all(target_feature = "avx2", not(all(target_feature = "avx512vl", target_feature = "avx512vbmi"))))]
pub const LANES: usize = 32;

#[cfg(not(any(target_feature = "avx2", all(target_feature = "avx512vl", target_feature = "avx512vbmi"))))]
pub const LANES: usize = 16;