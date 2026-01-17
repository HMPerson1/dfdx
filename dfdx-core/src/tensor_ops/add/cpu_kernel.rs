use crate::{
    prelude::Ignored,
    tensor_ops::cpu_kernels::{BinaryDerivative2, UnaryDerivative2},
};
use num_traits::Float;

impl<F: Float> BinaryDerivative2<F> for super::BinaryAddKernelOp {
    type BackLhsNeeded = Ignored;
    type BackRhsNeeded = Ignored;
    type BackOutNeeded = Ignored;
    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        x + y
    }
    #[inline(always)]
    fn df(&self, _x: (), _y: (), _f: ()) -> (F, F) {
        (F::one(), F::one())
    }
}

impl<F: Float> UnaryDerivative2<F> for super::ScalarAddKernelOp<F> {
    type BackInpNeeded = Ignored;
    type BackOutNeeded = Ignored;
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        x + self.scalar
    }
    #[inline(always)]
    fn df(&self, _x: (), _f: ()) -> F {
        F::one()
    }
}
