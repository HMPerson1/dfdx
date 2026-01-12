use std::alloc::Allocator;

use crate::{
    shapes::Dtype,
    tensor::{Cpu, Error},
};

impl<E: Dtype, A: Allocator + Clone + 'static> super::AxpyKernel<E> for Cpu<A> {
    fn forward(&self, a: &mut Self::Vec, alpha: E, b: &Self::Vec, beta: E) -> Result<(), Error> {
        for (a_i, b_i) in a.iter_mut().zip(b.iter()) {
            *a_i = *a_i * alpha + *b_i * beta;
        }
        Ok(())
    }
}
