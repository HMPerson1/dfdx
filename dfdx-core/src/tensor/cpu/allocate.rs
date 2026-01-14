#![allow(clippy::needless_range_loop)]

use crate::{
    shapes::*,
    tensor::{masks::triangle_mask, storage_traits::*, unique_id, Error, Tensor},
};

use super::{Cpu, LendingIterator};

#[cfg(test)]
use rand::{distributions::Distribution, Rng};
use std::{alloc::Allocator, mem, slice, rc::Rc};

impl<A: Allocator + Clone> Cpu<A> {
    #[inline]
    pub(crate) fn try_alloc_zeros<E: Unit>(&self, numel: usize) -> Result<Rc<[E], A>, Error> {
        let data = Rc::new_zeroed_slice_in(numel, self.alloc.clone());
        // SAFETY: E: SafeZeros
        Ok(unsafe { data.assume_init() })
    }

    #[inline]
    pub(crate) fn try_alloc_elem<E: Unit>(
        &self,
        numel: usize,
        elem: E,
    ) -> Result<Rc<[E], A>, Error> {
        let mut data = Rc::new_uninit_slice_in(numel, self.alloc.clone());
        for p in Rc::get_mut(&mut data).unwrap() {
            p.write(elem);
        }
        // SAFETY: each element of the slice has been written to
        Ok(unsafe { data.assume_init() })
    }

    pub(crate) fn try_alloc_copy<E: Unit>(&self, src: &[E]) -> Result<Rc<[E], A>, Error> {
        let len = src.len();
        let mut data = Rc::new_uninit_slice_in(len, self.alloc.clone());
        // SAFETY: casting to `MaybeUninit` is always safe
        let src = unsafe { slice::from_raw_parts(src.as_ptr() as *const mem::MaybeUninit<E>, len) };
        Rc::get_mut(&mut data).unwrap().copy_from_slice(src);
        // SAFETY: copy_from_slice wrote to each element
        Ok(unsafe { data.assume_init() })
    }
}

impl<E: Unit, A: Allocator + Clone> ZerosTensor<E> for Cpu<A> {
    #[inline]
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let strides = shape.strides();
        let data = self.try_alloc_zeros(shape.num_elements())?;
        Ok(Tensor {
            id: unique_id(),
            data,
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }
}

impl<E: Unit, A: Allocator + Clone> ZeroFillStorage<E> for Cpu<A> {
    fn try_fill_with_zeros(&self, storage: &mut Self::SharedVec) -> Result<(), Error> {
        Self::SharedVec::make_mut(storage).fill(Default::default());
        Ok(())
    }
}

impl<E: Unit, A: Allocator + Clone> OnesTensor<E> for Cpu<A> {
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let strides = shape.strides();
        let data = self.try_alloc_elem(shape.num_elements(), E::ONE)?;
        Ok(Tensor {
            id: unique_id(),
            data,
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }
}

impl<E: Unit, A: Allocator + Clone> TriangleTensor<E> for Cpu<A> {
    fn try_upper_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let strides = shape.strides();
        let mut data = self.try_alloc_elem::<E>(shape.num_elements(), val)?;
        let offset = diagonal.into().unwrap_or(0);
        triangle_mask(Rc::get_mut(&mut data).unwrap(), &shape, true, offset);
        Ok(Tensor {
            id: unique_id(),
            data,
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }

    fn try_lower_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let strides = shape.strides();
        let mut data = self.try_alloc_elem::<E>(shape.num_elements(), val)?;
        let offset = diagonal.into().unwrap_or(0);
        triangle_mask(Rc::get_mut(&mut data).unwrap(), &shape, false, offset);
        Ok(Tensor {
            id: unique_id(),
            data,
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }
}

impl<E: Unit, A: Allocator + Clone> OneFillStorage<E> for Cpu<A> {
    fn try_fill_with_ones(&self, storage: &mut Self::SharedVec) -> Result<(), Error> {
        Self::SharedVec::make_mut(storage).fill(E::ONE);
        Ok(())
    }

    fn try_fill_grad_with_ones(&self, storage: &mut Self::OwnedVec) -> Result<(), Error> {
        storage.fill(E::ONE);
        Ok(())
    }
}

#[cfg(test)]
impl<E: Unit, A: Allocator + Clone> SampleTensor<E> for Cpu<A> {
    fn try_sample_like<S: HasShape, D: Distribution<E>>(
        &self,
        src: &S,
        distr: D,
    ) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let mut tensor = self.try_zeros_like(src)?;
        {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::from_entropy();
            for v in Rc::get_mut(&mut tensor.data).unwrap().iter_mut() {
                *v = rng.sample(&distr);
            }
        }
        Ok(tensor)
    }
    fn try_fill_with_distr<D: Distribution<E>>(
        &self,
        storage: &mut Self::SharedVec,
        distr: D,
    ) -> Result<(), Error> {
        {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::from_entropy();
            for v in Self::SharedVec::make_mut(storage).iter_mut() {
                *v = rng.sample(&distr);
            }
        }
        Ok(())
    }
}

impl<E: Unit, A: Allocator + Clone> CopySlice<E> for Cpu<A> {
    fn copy_from<S: Shape, T>(dst: &mut Tensor<S, E, Self, T>, src: &[E]) {
        std::rc::Rc::make_mut(&mut dst.data).copy_from_slice(src);
    }
    fn copy_into<S: Shape, T>(src: &Tensor<S, E, Self, T>, dst: &mut [E]) {
        dst.copy_from_slice(src.data.as_ref());
    }
}

impl<E: Unit, A: Allocator + Clone> TensorFromVec<E> for Cpu<A> {
    fn try_tensor_from_vec<S: Shape>(
        &self,
        src: &[E],
        shape: S,
    ) -> Result<Tensor<S, E, Self>, Error> {
        let num_elements = shape.num_elements();

        if src.len() != num_elements {
            Err(Error::WrongNumElements)
        } else {
            Ok(Tensor {
                id: unique_id(),
                data: self.try_alloc_copy(src)?,
                shape,
                strides: shape.strides(),
                device: self.clone(),
                tape: Default::default(),
            })
        }
    }
}

impl<E: Unit, A: Allocator + Clone> TensorToArray<Rank0, E> for Cpu<A> {
    type Array = E;
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank0, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = Default::default();
        out.clone_from(&tensor.data[0]);
        out
    }
}

impl<E: Unit, A: Allocator + Clone, const M: usize> TensorToArray<Rank1<M>, E> for Cpu<A> {
    type Array = [E; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank1<M>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [Default::default(); M];
        let mut iter = tensor.iter();
        for m in 0..M {
            out[m].clone_from(iter.next().unwrap());
        }
        out
    }
}

impl<E: Unit, A: Allocator + Clone, const M: usize, const N: usize> TensorToArray<Rank2<M, N>, E>
    for Cpu<A>
{
    type Array = [[E; N]; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank2<M, N>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [[Default::default(); N]; M];
        let mut iter = tensor.iter();
        for m in 0..M {
            for n in 0..N {
                out[m][n].clone_from(iter.next().unwrap());
            }
        }
        out
    }
}

impl<E: Unit, A: Allocator + Clone, const M: usize, const N: usize, const O: usize>
    TensorToArray<Rank3<M, N, O>, E> for Cpu<A>
{
    type Array = [[[E; O]; N]; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank3<M, N, O>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [[[Default::default(); O]; N]; M];
        let mut iter = tensor.iter_with_index();
        while let Some((v, [m, n, o])) = iter.next() {
            out[m][n][o].clone_from(v);
        }
        out
    }
}

impl<
        E: Unit,
        A: Allocator + Clone,
        const M: usize,
        const N: usize,
        const O: usize,
        const P: usize,
    > TensorToArray<Rank4<M, N, O, P>, E> for Cpu<A>
{
    type Array = [[[[E; P]; O]; N]; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank4<M, N, O, P>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [[[[Default::default(); P]; O]; N]; M];
        let mut iter = tensor.iter_with_index();
        while let Some((v, [m, n, o, p])) = iter.next() {
            out[m][n][o][p].clone_from(v);
        }
        out
    }
}
