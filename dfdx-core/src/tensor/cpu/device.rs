use crate::shapes::{Shape, Unit};
use crate::tensor::{cpu::LendingIterator, storage_traits::*, Error, Tensor};
use std::vec::Vec;

/// A device that stores data on the heap.
///
/// The [Default] impl seeds the underlying rng with seed of 0.
///
/// Use [Cpu::seed_from_u64] to control what seed is used.
#[derive(Clone, Debug)]
pub struct Cpu {
}

impl Default for Cpu {
    fn default() -> Self {
        Self {
        }
    }
}

impl<E: Unit> Storage<E> for Cpu {
    type Vec = Vec<E>;

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Error> {
        self.try_alloc_zeros(len)
    }

    fn len(&self, v: &Self::Vec) -> usize {
        v.len()
    }

    fn tensor_to_vec<S: Shape, T>(&self, tensor: &Tensor<S, E, Self, T>) -> Vec<E> {
        let mut buf = Vec::with_capacity(tensor.shape.num_elements());
        let mut iter = tensor.iter();
        while let Some(v) = iter.next() {
            buf.push(*v);
        }
        buf
    }
}

impl Synchronize for Cpu {
    fn try_synchronize(&self) -> Result<(), Error> {
        Ok(())
    }
}
