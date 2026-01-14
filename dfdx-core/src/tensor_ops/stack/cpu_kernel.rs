use crate::{
    shapes::*,
    tensor::{unique_id, Cpu, Error, Tensor},
};

use std::{alloc::Allocator, rc::Rc, vec::Vec};

impl<E: Dtype, A: Allocator + Clone> super::StackKernel<E> for Cpu<A> {
    fn forward<S: Shape, Num: Dim>(
        &self,
        num: Num,
        inp: &[Tensor<S, E, Self>],
    ) -> Result<Tensor<S::Larger, E, Self>, Error>
    where
        S: super::AddDim<Num>,
    {
        debug_assert_eq!(inp.len(), num.size());

        // check that all the strides are the same
        let item_strides = inp[0].strides;
        for i in inp.iter() {
            assert_eq!(i.strides, item_strides);
        }
        let shape: S::Larger = inp[0].shape().add_dim(num);

        // build the new strides
        let mut strides = shape.strides();
        strides[0] = inp[0].data.len();
        for d in 1..<S::Larger as Shape>::NUM_DIMS {
            strides[d] = item_strides[d - 1];
        }

        // copy the data
        let mut data = self.try_alloc_zeros(inp.len() * inp[0].data.len())?;
        let data_mut = Rc::get_mut(&mut data).unwrap();
        let mut i = 0;
        for item in inp {
            let buf: &[E] = item.data.as_ref();
            data_mut[i..i + buf.len()].copy_from_slice(buf);
            i += buf.len();
        }

        Ok(Tensor {
            id: unique_id(),
            data,
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }
    fn backward(
        &self,
        mut grad_inp: Vec<&mut Self::OwnedVec>,
        grad_out: &Self::OwnedVec,
    ) -> Result<(), Error> {
        let mut offset = 0;
        for item in grad_inp.drain(..) {
            for gi in item.iter_mut() {
                *gi += grad_out[offset];
                offset += 1;
            }
        }
        Ok(())
    }
}
