use std::{alloc::Allocator, borrow::Cow, hint, rc::Rc};

use super::ops::{BinaryKernel, BinaryKernel2, IsNeeded, UnaryKernel, UnaryKernel2};
use crate::{
    shapes::{Dtype, HasShape, Shape},
    tensor::{
        cpu::{Cpu, LendingIterator, NdIndex},
        unique_id, Error, Tensor, Tensorlike, ZerosTensor,
    },
};

pub trait UnaryDerivative<E> {
    /// Whether the [UnaryDerivative::df] function can re-use the output
    /// from [UnaryDerivative::f].
    const DF_USES_FX: bool;
    /// Whether the derivative of this op can be computed without
    /// any data.
    const HAS_CONST_DF: bool;

    fn f(&self, x: &E) -> E;

    /// Receives `f(x)` if [UnaryDerivative::DF_USES_FX] is true,
    /// otherwise `x`.
    fn df(&self, x: &E) -> E;

    fn const_df(&self) -> E {
        unimplemented!()
    }
}

pub trait BinaryDerivative<E>: std::fmt::Debug {
    /// Whether the derivative of this op can be computed without
    /// any data.
    const HAS_CONST_DF: bool;
    fn f(&self, x: &E, y: &E) -> E;
    fn dfdx(&self, x: &E, y: &E) -> E;
    fn dfdy(&self, x: &E, y: &E) -> E;
    fn const_dfdx(&self) -> E {
        unimplemented!()
    }
    fn const_dfdy(&self) -> E {
        unimplemented!()
    }
}

impl<E: Dtype, Op: UnaryDerivative<E>, A: Allocator + Clone> UnaryKernel<Op, E> for Cpu<A> {
    const BACKWARD_WITHOUT_INP: bool = Op::DF_USES_FX;
    const BACKWARD_WITHOUT_DATA: bool = Op::HAS_CONST_DF;

    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        let mut out = match inp {
            Cow::Borrowed(inp) => {
                // allocate a new data buffer
                Tensor {
                    id: unique_id(),
                    data: inp.data.clone(),
                    shape: inp.shape,
                    strides: inp.strides,
                    device: self.clone(),
                    tape: Default::default(),
                }
            }
            Cow::Owned(mut inp) => {
                // re-use the data buffer
                inp.id = unique_id();
                inp
            }
        };
        // NOTE: we can iterate over buf here because we know inp & out
        // have exact same strides due to clone.
        for x in out.buf_iter_mut() {
            *x = op.f(x);
        }
        Ok(out)
    }
    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: &impl Tensorlike<S, E, Self>,
        grad_inp: &mut Self::OwnedVec,
        out: &impl Tensorlike<S, E, Self>,
        grad_out: &Self::OwnedVec,
    ) -> Result<(), Error> {
        match (inp.data(), out.data()) {
            (None, None) => {
                let df = op.const_df();
                for (i, x) in grad_inp.iter_mut().enumerate() {
                    *x += df * grad_out[i];
                }
            }
            (None, Some(out)) => {
                for (i, x) in grad_inp.iter_mut().enumerate() {
                    *x += op.df(&out[i]) * grad_out[i];
                }
            }
            (Some(inp), None) => {
                for (i, x) in grad_inp.iter_mut().enumerate() {
                    *x += op.df(&inp[i]) * grad_out[i];
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}

impl<E: Dtype, Op: BinaryDerivative<E>, A: Allocator + Clone> BinaryKernel<Op, E> for Cpu<A> {
    const BACKWARD_WITHOUT_DATA: bool = Op::HAS_CONST_DF;
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: Cow<Tensor<S, E, Self>>,
        rhs: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        match (lhs, rhs) {
            (Cow::Borrowed(lhs), Cow::Borrowed(rhs)) => {
                let mut out = self.try_zeros_like(&lhs.shape)?;
                let mut lhs_iter = lhs.iter();
                let mut rhs_iter = rhs.iter();
                for o in out.buf_iter_mut() {
                    let l = lhs_iter.next().unwrap();
                    let r = rhs_iter.next().unwrap();
                    *o = op.f(l, r);
                }
                Ok(out)
            }
            (Cow::Owned(mut lhs), Cow::Owned(mut rhs)) => {
                let lhs_valid = lhs.strides == lhs.shape.strides();
                let rhs_valid = rhs.strides == rhs.shape.strides();
                if lhs_valid || rhs_valid {
                    let lhs_count = std::rc::Rc::strong_count(&lhs.data);
                    let rhs_count = std::rc::Rc::strong_count(&rhs.data);
                    if rhs_valid && (rhs_count == 1 || !lhs_valid || lhs_count != 1) {
                        rhs.id = unique_id();
                        let lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                        for (i, r) in rhs.buf_iter_mut().enumerate() {
                            *r = op.f(&lhs.data[lhs_idx.get_strided_index(i)], r);
                        }
                        Ok(rhs)
                    } else {
                        lhs.id = unique_id();
                        let rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                        for (i, l) in lhs.buf_iter_mut().enumerate() {
                            *l = op.f(l, &rhs.data[rhs_idx.get_strided_index(i)]);
                        }
                        Ok(lhs)
                    }
                } else {
                    <Self as BinaryKernel<Op, E>>::forward(
                        self,
                        op,
                        Cow::Borrowed(&lhs),
                        Cow::Borrowed(&rhs),
                    )
                }
            }
            _ => unreachable!(),
        }
    }
    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: &impl Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::OwnedVec,
        rhs: &impl Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::OwnedVec,
        grad_out: &Self::OwnedVec,
    ) -> Result<(), Error> {
        match (lhs.data(), rhs.data()) {
            (Some(lhs_buf), Some(rhs_buf)) => {
                let mut lhs_idx = NdIndex::new(*lhs.shape(), lhs.strides());
                let mut rhs_idx = NdIndex::new(*rhs.shape(), rhs.strides());
                // NOTE: we can use .buf_iter() here because we know the outcome of this op is
                // contiguous from forward
                for &go in grad_out.iter() {
                    let lhs_i = lhs_idx.next().unwrap();
                    let rhs_i = rhs_idx.next().unwrap();
                    let l = &lhs_buf[lhs_i];
                    let r = &rhs_buf[rhs_i];
                    grad_lhs[lhs_i] += op.dfdx(l, r) * go;
                    grad_rhs[rhs_i] += op.dfdy(l, r) * go;
                }
            }
            (None, None) => {
                assert!(Op::HAS_CONST_DF);
                let mut lhs_idx = NdIndex::new(*lhs.shape(), lhs.strides());
                let mut rhs_idx = NdIndex::new(*rhs.shape(), rhs.strides());
                let dx = op.const_dfdx();
                let dy = op.const_dfdy();
                for &go in grad_out.iter() {
                    let lhs_i = lhs_idx.next().unwrap();
                    let rhs_i = rhs_idx.next().unwrap();
                    grad_lhs[lhs_i] += dx * go;
                    grad_rhs[rhs_i] += dy * go;
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}

pub trait UnaryDerivative2<E> {
    type BackInpNeeded: IsNeeded;
    type BackOutNeeded: IsNeeded;

    fn f(&self, x: &E) -> E;
    fn df(
        &self,
        x: <Self::BackInpNeeded as IsNeeded>::Output<&E>,
        f: <Self::BackOutNeeded as IsNeeded>::Output<&E>,
    ) -> E;
}

pub trait BinaryDerivative2<E>: std::fmt::Debug {
    type BackLhsNeeded: IsNeeded;
    type BackRhsNeeded: IsNeeded;
    type BackOutNeeded: IsNeeded;

    fn f(&self, x: &E, y: &E) -> E;
    fn df(
        &self,
        x: <Self::BackLhsNeeded as IsNeeded>::Output<&E>,
        y: <Self::BackRhsNeeded as IsNeeded>::Output<&E>,
        f: <Self::BackOutNeeded as IsNeeded>::Output<&E>,
    ) -> (E, E);
}

impl<E: Dtype, Op: UnaryDerivative2<E>, A: Allocator + Clone> UnaryKernel2<Op, E> for Cpu<A> {
    type BackInpNeeded = Op::BackInpNeeded;
    type BackOutNeeded = Op::BackOutNeeded;

    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        let mut out = inp;
        out.id = unique_id();
        for x in out.buf_iter_mut() {
            *x = op.f(x);
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: <Self::BackInpNeeded as IsNeeded>::Output<Tensor<S, E, Self>>,
        grad_inp: &mut Self::OwnedVec,
        out: <Self::BackOutNeeded as IsNeeded>::Output<Tensor<S, E, Self>>,
        grad_out: &Self::OwnedVec,
    ) -> Result<(), Error> {
        for (i, x) in grad_inp.iter_mut().enumerate() {
            *x += op.df(
                Self::BackInpNeeded::fmap(&inp, |x| &x.data[i]),
                Self::BackOutNeeded::fmap(&out, |x| &x.data[i]),
            ) * grad_out[i];
        }
        Ok(())
    }
}

impl<E: Dtype, Op: BinaryDerivative2<E>, A: Allocator + Clone> BinaryKernel2<Op, E> for Cpu<A> {
    type BackLhsNeeded = Op::BackLhsNeeded;
    type BackRhsNeeded = Op::BackRhsNeeded;
    type BackOutNeeded = Op::BackOutNeeded;

    fn forward<S: Shape>(
        &self,
        op: Op,
        mut lhs: Tensor<S, E, Self>,
        mut rhs: Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        // insane gymastics to enable auto-vectorization which somehow also reduces code size
        enum OpStrategy<L, R> {
            ClobberRhs(Option<R>, bool),
            ClobberLhs(Option<L>, bool),
            AllocNew,
        }
        #[inline]
        fn mk_strategy<L, R>(l: Option<Option<L>>, r: Option<Option<R>>) -> OpStrategy<L, R> {
            match (l, r) {
                (l, Some(Some(d))) => OpStrategy::ClobberRhs(Some(d), l.is_some()),
                (Some(Some(d)), r) => OpStrategy::ClobberLhs(Some(d), r.is_some()),
                (l, Some(_)) => OpStrategy::ClobberRhs(None, l.is_some()),
                (Some(_), r) => OpStrategy::ClobberLhs(None, r.is_some()),
                (None, None) => OpStrategy::AllocNew,
            }
        }

        let lhs_data = (lhs.strides == lhs.shape.strides()).then(|| Rc::get_mut(&mut lhs.data));
        let rhs_data = (rhs.strides == rhs.shape.strides()).then(|| Rc::get_mut(&mut rhs.data));
        match mk_strategy(lhs_data, rhs_data) {
            OpStrategy::ClobberRhs(rhs_data, lhs_contig) => {
                let rhs_data = match rhs_data {
                    Some(d) => d,
                    None => Rc::make_mut(&mut rhs.data),
                };
                rhs.id = unique_id();
                if lhs_contig {
                    for (r, l) in rhs_data.into_iter().zip(&lhs.data[..]) {
                        *r = op.f(l, r);
                    }
                } else {
                    let lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                    for (i, r) in rhs_data.into_iter().enumerate() {
                        *r = op.f(&lhs.data[lhs_idx.get_strided_index_slow(i)], r);
                    }
                }
                Ok(rhs)
            }
            OpStrategy::ClobberLhs(lhs_data, rhs_contig) => {
                let lhs_data = match lhs_data {
                    Some(d) => d,
                    None => Rc::make_mut(&mut lhs.data),
                };
                lhs.id = unique_id();
                if rhs_contig {
                    for (l, r) in lhs_data.into_iter().zip(&rhs.data[..]) {
                        *l = op.f(l, r);
                    }
                } else {
                    let rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                    for (i, l) in lhs_data.into_iter().enumerate() {
                        *l = op.f(l, &rhs.data[rhs_idx.get_strided_index_slow(i)]);
                    }
                }
                Ok(lhs)
            }
            OpStrategy::AllocNew => {
                hint::cold_path();
                let mut out = self.try_zeros_like(&lhs.shape)?;
                let lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                let rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                for (i, o) in out.buf_iter_mut().enumerate() {
                    let l = &lhs.data[lhs_idx.get_strided_index_slow(i)];
                    let r = &rhs.data[rhs_idx.get_strided_index_slow(i)];
                    *o = op.f(l, r);
                }
                Ok(out)
            }
        }
    }

    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs_ghost: crate::prelude::GhostTensor<S, E, Self>,
        lhs_data: <Self::BackLhsNeeded as IsNeeded>::Output<Self::SharedVec>,
        grad_lhs: &mut Self::OwnedVec,
        rhs_ghost: crate::prelude::GhostTensor<S, E, Self>,
        rhs_data: <Self::BackRhsNeeded as IsNeeded>::Output<Self::SharedVec>,
        grad_rhs: &mut Self::OwnedVec,
        out: <Self::BackOutNeeded as IsNeeded>::Output<Tensor<S, E, Self>>,
        grad_out: &Self::OwnedVec,
    ) -> Result<(), Error> {
        let lhs_idx = NdIndex::new(*lhs_ghost.shape(), lhs_ghost.strides());
        let rhs_idx = NdIndex::new(*rhs_ghost.shape(), rhs_ghost.strides());
        // NOTE: we can use .buf_iter() here because we know the outcome of this op is
        // contiguous from forward
        for (out_i, &go) in grad_out.iter().enumerate() {
            let lhs_i = lhs_idx.get_strided_index_slow(out_i);
            let rhs_i = rhs_idx.get_strided_index_slow(out_i);
            let l = Self::BackLhsNeeded::fmap(&lhs_data, |x| &x[lhs_i]);
            let r = Self::BackRhsNeeded::fmap(&rhs_data, |x| &x[rhs_i]);
            let f = Self::BackOutNeeded::fmap(&out, |x| &x.data[out_i]);
            let (dfdx, dfdy) = op.df(l, r, f);
            grad_lhs[lhs_i] += dfdx * go;
            grad_rhs[rhs_i] += dfdy * go;
        }
        Ok(())
    }
}
