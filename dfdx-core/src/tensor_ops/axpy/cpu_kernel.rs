use std::alloc::Allocator;

use crate::{
    shapes::Dtype,
    tensor::{Cpu, Error},
};

impl<E: Dtype, A: Allocator + Clone> super::AxpyKernel<E> for Cpu<A> {
    fn forward(&self, a: &mut Self::SharedVec, alpha: E, b: &Self::SharedVec, beta: E) -> Result<(), Error> {
        for (a_i, b_i) in Self::SharedVec::make_mut(a).iter_mut().zip(b.iter()) {
            *a_i = *a_i * alpha + *b_i * beta;
        }
        Ok(())
    }
}
