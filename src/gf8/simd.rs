use std::simd::Simd;
use crate::gf8::lookup::GF8_MUL_LOOKUP;

/// Computes result = values * n
#[inline(always)]
pub fn gf8_simd_mul<const LANES: usize>(values: Simd<u8, LANES>, n: u8) -> Simd<u8, LANES> {
    const { assert!(LANES >= 16, "LANES must be >= 16: nibble lookup tables have 16 entries") };
    let mask = Simd::splat(0b1111_u8);
    let values_lo = values & mask;
    let values_hi = values >> 4;
    let lookup = GF8_MUL_LOOKUP[n as usize];
    let mut result = Simd::from_array(std::array::from_fn(|i| lookup.hi[i & 15])).swizzle_dyn(values_hi);
    result ^= Simd::from_array(std::array::from_fn(|i| lookup.lo[i & 15])).swizzle_dyn(values_lo);
    result
}

/// Computes acc ^= values * n
#[inline(always)]
pub fn gf8_simd_mul_xor<const LANES: usize>(acc: &mut Simd<u8, LANES>, mut values: Simd<u8, LANES>, n: u8) {
    const { assert!(LANES >= 16, "LANES must be >= 16: nibble lookup tables have 16 entries") };
    let mask = Simd::splat(0b1111_u8);
    let values_lo = values & mask;
    let lookup = GF8_MUL_LOOKUP[n as usize];
    values >>= 4;
    *acc ^= Simd::from_array(std::array::from_fn(|i| lookup.lo[i & 15])).swizzle_dyn(values_lo);
    *acc ^= Simd::from_array(std::array::from_fn(|i| lookup.hi[i & 15])).swizzle_dyn(values);
}

