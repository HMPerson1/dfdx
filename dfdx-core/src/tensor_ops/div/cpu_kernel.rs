use crate::{
    prelude::{Ignored, Needed},
    tensor_ops::cpu_kernels::{BinaryDerivative2, UnaryDerivative2},
};
use num_traits::Float;

impl<F: Float> UnaryDerivative2<F> for super::ScalarDivKernelOp<F> {
    type BackInpNeeded = Ignored;
    type BackOutNeeded = Ignored;
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        x / self.scalar
    }
    #[inline(always)]
    fn df(&self, _x: (), _f: ()) -> F {
        F::one() / self.scalar
    }
}

impl<F: Float> BinaryDerivative2<F> for super::BinaryDivKernelOp {
    type BackLhsNeeded = Ignored;
    type BackRhsNeeded = Needed;
    type BackOutNeeded = Needed;
    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        x / y
    }
    #[inline(always)]
    fn df(&self, _x: (), &y: &F, &f: &F) -> (F, F) {
        let frac_1_y = y.recip();
        (frac_1_y, -f * frac_1_y)
    }
}
