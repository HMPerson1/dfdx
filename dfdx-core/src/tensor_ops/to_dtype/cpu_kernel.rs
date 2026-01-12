use num_traits::AsPrimitive;
use std::{alloc::Allocator, vec::Vec};

use crate::prelude::{Cpu, Error, Shape, Tensor, Unit};

impl<E1: Unit + AsPrimitive<E2>, E2: Unit, A: Allocator + Clone> super::ToDtypeKernel<E1, E2> for Cpu<A> {
    fn forward<S: Shape>(inp: Tensor<S, E1, Self>) -> Result<Tensor<S, E2, Self>, Error> {
        let mut data = Vec::new_in(inp.device.alloc.clone());
        data.extend(inp.data.iter().map(|x| (*x).as_()));

        Ok(Tensor {
            id: crate::prelude::unique_id(),
            // extra memcpy, oh well
            data: data.into(),
            shape: inp.shape,
            strides: inp.strides,
            device: inp.device.clone(),
            tape: inp.tape,
        })
    }
}
