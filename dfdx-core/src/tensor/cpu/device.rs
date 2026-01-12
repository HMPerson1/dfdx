use crate::shapes::{Shape, Unit};
use crate::tensor::{cpu::LendingIterator, storage_traits::*, Error, Tensor};
use std::alloc::{Allocator, Global};
use std::sync::Arc;
use std::vec::Vec;

/// A device that stores data on the heap.
///
/// The [Default] impl seeds the underlying rng with seed of 0.
///
/// Use [Cpu::seed_from_u64] to control what seed is used.
#[derive(Clone, Debug)]
pub struct Cpu<A: Allocator = Global> {
    pub alloc: A,
}

impl<A: Allocator> Cpu<A> {
    pub fn with_allocator(alloc: A) -> Self {
        Self { alloc }
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self { alloc: Global }
    }
}

impl<E: Unit, A: Allocator + Clone> Storage<E> for Cpu<A> {
    type SharedVec = Arc<[E], A>;
    type OwnedVec = Vec<E, A>;

    fn try_alloc_grad(&self, len: usize) -> Result<Vec<E, A>, Error> {
        Ok(std::vec::from_elem_in(E::default(), len, self.alloc.clone()))
    }
    
    fn grad_to_tensor(&self, v: &Self::OwnedVec) -> Arc<[E], A> {
        self.try_alloc_copy(v).unwrap()
    }

    fn len(&self, v: &Self::SharedVec) -> usize {
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

impl<A: Allocator> Synchronize for Cpu<A> {
    fn try_synchronize(&self) -> Result<(), Error> {
        Ok(())
    }
}
