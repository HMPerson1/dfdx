use crate::{
    shapes::{Dtype, Shape},
    tensor::{cpu::NdIndex, *},
};

use std::{alloc::Allocator, sync::Arc};

impl<E: Dtype, A: Allocator + Clone> super::RollKernel<E> for Cpu<A> {
    fn forward<S: Shape>(
        &self,
        op: super::RollOp,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        let dims = inp.shape.concrete();
        let strides = inp.shape.strides();
        let mut data = self.try_alloc_zeros::<E>(inp.shape.num_elements())?;
        let data_mut = Arc::get_mut(&mut data).unwrap();
        let mut idx = NdIndex::new(inp.shape, inp.strides);
        while let Some((old_i, mut idx)) = idx.next_with_idx() {
            idx[op.axis] = (idx[op.axis] + op.amount) % dims[op.axis];
            let new_i = idx
                .into_iter()
                .zip(strides)
                .map(|(i, s)| i * s)
                .sum::<usize>();
            data_mut[new_i] = inp.data[old_i];
        }
        Ok(Tensor {
            id: unique_id(),
            data,
            shape: inp.shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }
    fn backward<S: Shape>(
        &self,
        op: super::RollOp,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::OwnedVec,
        grad_out: &Self::OwnedVec,
    ) -> Result<(), Error> {
        let dims = inp.shape.concrete();
        let strides = inp.shape.strides();
        let mut idx = NdIndex::new(inp.shape, inp.strides);
        while let Some((old_i, mut idx)) = idx.next_with_idx() {
            idx[op.axis] = (idx[op.axis] + op.amount) % dims[op.axis];
            let new_i = idx
                .into_iter()
                .zip(strides)
                .map(|(i, s)| i * s)
                .sum::<usize>();
            grad_inp[old_i] += grad_out[new_i];
        }
        Ok(())
    }
}
