use crate::{
    prelude::{Ignored, Needed},
    tensor_ops::cpu_kernels::{BinaryDerivative2, UnaryDerivative2},
};

use num_traits::Float;

impl<F: Float> UnaryDerivative2<F> for super::ScalarMulKernelOp<F> {
    type BackInpNeeded = Ignored;
    type BackOutNeeded = Ignored;

    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        x * self.scalar
    }
    #[inline(always)]
    fn df(&self, _x: (), _f: ()) -> F {
        self.scalar
    }
}

impl<F: Float> BinaryDerivative2<F> for super::BinaryMulKernelOp {
    type BackLhsNeeded = Needed;
    type BackRhsNeeded = Needed;
    type BackOutNeeded = Ignored;

    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        x * y
    }

    fn df(&self, &x: &F, &y: &F, _f: ()) -> (F, F) {
        (y, x)
    }
}
