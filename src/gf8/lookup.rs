use crate::gf8::GF8;

#[derive(Copy, Clone)]
pub struct MulLookup {
    pub hi: [u8; 16],
    pub lo: [u8; 16],
}

pub const GF8_MUL_LOOKUP: [MulLookup; 256] = build_lookup();
pub const MUL_TABLE: [[u8; 256]; 256] = build_mul_table();
pub const INV_TABLE: [u8; 256] = build_inv_table();

/// Builds lookup tables for fast multiplication by a fixed GF(2^8) element.
/// 
/// For all n, this precomputes:
///   lo[nibble] = n * i
///   hi[nibble] = n * (i << 4)
/// 
/// where i ranges over all nibles values (256)
/// 
/// Any byte `x` can be decomposed:
///   x = high_nibble << 4 + low_nibble
/// So
///     n * x
///   = n * ((hi_nibble << 4) + lo_nibble)
///   = n *  (hi_nibble << 4) + n * lo_nibble
/// 
/// Therefore (xor is equivalent to + in GF(8)):
///   n * x = hi[byte >> 4] ^ lo[byte & 0x0f] 
const fn build_lookup() -> [MulLookup; 256] {
    let mut result_hi = [[0_u8; 16]; 256];
    let mut result_lo = [[0_u8; 16]; 256];

    let mut x: usize = 0;
    while x < 256 {
        let mut y: usize = 0;
        while y < 16 {
            let lo = GF8::new(y as u8).mul(GF8::new(x as u8));
            let hi = lo.mul(GF8::new(16));

            result_hi[x][y] = hi.value();
            result_lo[x][y] = lo.value();

            y += 1;
        }
        x += 1;
    }

    let mut result = [MulLookup { hi: [0; 16], lo: [0; 16] }; 256];
    let mut x: usize = 0;
    while x < 256 {
        result[x].hi = result_hi[x];
        result[x].lo = result_lo[x];
        x += 1;
    }

    result
}

const fn build_mul_table() -> [[u8; 256]; 256] {
    let mut t = [[0u8; 256]; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut j = 0usize;
        while j < 256 {
            t[i][j] = gf_mul_scalar(i as u8, j as u8);
            j += 1;
        }
        i += 1;
    }
    t
}

const fn build_inv_table() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 1usize;
    while i < 256 {
        // a^(-1) = a^254 by Fermat's little theorem (|GF(2^8)*| = 255)
        let mut result = 1u8;
        let mut base = i as u8;
        let mut exp = 254u8;
        while exp > 0 {
            if exp & 1 != 0 { result = gf_mul_scalar(result, base); }
            base = gf_mul_scalar(base, base);
            exp >>= 1;
        }
        t[i] = result;
        i += 1;
    }
    t
}

const fn gf_double(a: u8) -> u8 {
    let shifted = (a as u16) << 1;
    if a & 0x80 != 0 { (shifted ^ 0x11D) as u8 } else { shifted as u8 }
}

const fn gf_mul_scalar(mut a: u8, mut b: u8) -> u8 {
    let mut result = 0u8;
    while b > 0 {
        if b & 1 != 0 { result ^= a; }
        a = gf_double(a);
        b >>= 1;
    }
    result
}