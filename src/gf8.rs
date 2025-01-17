use crate::gf8::lookup::{INV_TABLE, MUL_TABLE};

mod simd;
mod lookup;

pub use simd::gf8_simd_mul;
pub use simd::gf8_simd_mul_xor;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
#[repr(transparent)]
pub struct GF8(u8);

impl GF8 {
    pub const fn new(value: u8) -> Self {
        GF8(value)
    }

    pub const fn inv(&self) -> GF8 {
        if self.0 == 0 {
            panic!("division by zero in GF8");
        }
        GF8(INV_TABLE[self.0 as usize])
    }

    pub const fn mul(&self, rhs: GF8) -> GF8 {
        GF8(MUL_TABLE[self.0 as usize][rhs.0 as usize])
    }

    pub const fn add(&self, rhs: GF8) -> GF8 {
        GF8(self.0 ^ rhs.0)
    }

    pub const fn value(&self) -> u8 {
        self.0
    }
}
