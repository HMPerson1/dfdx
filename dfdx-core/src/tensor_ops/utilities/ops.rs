use crate::{
    shapes::{Dtype, HasShape, Shape},
    tensor::*,
};
use std::{borrow::Cow, sync::Arc};

pub trait UnaryKernel<Op, E: Dtype>: Storage<E> {
    const BACKWARD_WITHOUT_INP: bool;
    const BACKWARD_WITHOUT_DATA: bool;
    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Error>;
    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: &impl Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl Tensorlike<S, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Error>;
}

pub trait BinaryKernel<Op, E: Dtype>: Storage<E> {
    const BACKWARD_WITHOUT_DATA: bool;
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: Cow<Tensor<S, E, Self>>,
        rhs: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Error>;
    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: &impl Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &impl Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error>;
}

pub(crate) fn try_unary_op<
    Op: 'static + Clone,
    S: Shape,
    E: Dtype,
    D: UnaryKernel<Op, E>,
    T: Tape<E, D>,
>(
    op: Op,
    inp: Tensor<S, E, D, T>,
) -> Result<Tensor<S, E, D, T>, crate::tensor::Error> {
    let (inp, mut tape) = inp.split_tape();
    let inp_ghost = inp.ghost();
    let dev = inp.device.clone();
    if !T::OWNS_TAPE || D::BACKWARD_WITHOUT_DATA {
        let out = inp_ghost.dev.forward(op.clone(), Cow::Owned(inp))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            dev.backward(op, &inp_ghost, grad_inp, &out_ghost, grad_out)
        });
        Ok(out.put_tape(tape))
    } else if D::BACKWARD_WITHOUT_INP {
        let out = inp_ghost.dev.forward(op.clone(), Cow::Owned(inp))?;
        let out_ghost = out.ghost();
        let out_clone = out.clone();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            dev.backward(op, &inp_ghost, grad_inp, &out_clone, grad_out)
        });
        Ok(out.put_tape(tape))
    } else {
        let out = inp.device.forward(op.clone(), Cow::Borrowed(&inp))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            dev.backward(op, &inp, grad_inp, &out_ghost, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

pub(crate) fn try_binary_op<
    Op: 'static + Copy,
    S: Shape,
    E: Dtype,
    D: BinaryKernel<Op, E>,
    RhsTape,
    LhsTape: Tape<E, D> + Merge<RhsTape>,
>(
    op: Op,
    lhs: Tensor<S, E, D, LhsTape>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Result<Tensor<S, E, D, LhsTape>, crate::tensor::Error> {
    assert_eq!(lhs.shape(), rhs.shape());
    let (lhs, ltape) = lhs.split_tape();
    let (rhs, rtape) = rhs.split_tape();
    let lhs_ghost = lhs.ghost();
    let rhs_ghost = rhs.ghost();
    let mut tape = ltape.merge(rtape);
    if !LhsTape::OWNS_TAPE || D::BACKWARD_WITHOUT_DATA {
        let out = lhs_ghost
            .dev
            .forward(op, Cow::Owned(lhs), Cow::Owned(rhs))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_lhs, grad_rhs, grad_out) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs_ghost
                .dev
                .backward(op, &lhs_ghost, grad_lhs, &rhs_ghost, grad_rhs, grad_out)
        });
        Ok(out.put_tape(tape))
    } else {
        let out = lhs
            .device
            .forward(op, Cow::Borrowed(&lhs), Cow::Borrowed(&rhs))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_lhs, grad_rhs, grad_out) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs.device
                .backward(op, &lhs, grad_lhs, &rhs, grad_rhs, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

// i guess this is pointed functor
pub trait IsNeeded {
    type Output<T>;
    fn make<T>(f: impl FnOnce() -> T) -> Self::Output<T>;
    fn fmap<T, U>(x: &Self::Output<T>, f: impl FnOnce(&T) -> &U) -> Self::Output<&U>;
}
pub struct Needed;
// identity monad
impl IsNeeded for Needed {
    type Output<T> = T;
    fn make<T>(f: impl FnOnce() -> T) -> T {
        f()
    }

    fn fmap<T, U>(x: &T, f: impl FnOnce(&T) -> &U) -> &U {
        f(x)
    }
}
pub struct Ignored;
// const monad
impl IsNeeded for Ignored {
    type Output<T> = ();
    fn make<T>(_f: impl FnOnce() -> T) -> () {}
    fn fmap<T, U>(_x: &(), _f: impl FnOnce(&T) -> &U) -> () {}
}

pub trait UnaryKernel2<Op, E: Dtype>: Storage<E> {
    type BackInpNeeded: IsNeeded;
    type BackOutNeeded: IsNeeded;
    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Error>;
    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: <Self::BackInpNeeded as IsNeeded>::Output<Tensor<S, E, Self>>,
        grad_inp: &mut Self::Vec,
        out: <Self::BackOutNeeded as IsNeeded>::Output<Tensor<S, E, Self>>,
        grad_out: &Self::Vec,
    ) -> Result<(), Error>;
}

pub trait BinaryKernel2<Op, E: Dtype>: Storage<E> {
    type BackLhsNeeded: IsNeeded;
    type BackRhsNeeded: IsNeeded;
    type BackOutNeeded: IsNeeded;
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: Tensor<S, E, Self>,
        rhs: Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Error>;
    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs_ghost: GhostTensor<S, E, Self>,
        lhs_data: <Self::BackLhsNeeded as IsNeeded>::Output<Arc<Self::Vec>>,
        grad_lhs: &mut Self::Vec,
        rhs_ghost: GhostTensor<S, E, Self>,
        rhs_data: <Self::BackRhsNeeded as IsNeeded>::Output<Arc<Self::Vec>>,
        grad_rhs: &mut Self::Vec,
        out: <Self::BackOutNeeded as IsNeeded>::Output<Tensor<S, E, Self>>,
        grad_out: &Self::Vec,
    ) -> Result<(), Error>;
}

pub fn try_unary_op2<
    Op: 'static + Clone,
    S: Shape,
    E: Dtype,
    D: UnaryKernel2<Op, E>,
    T: Tape<E, D>,
>(
    op: Op,
    inp: Tensor<S, E, D, T>,
) -> Result<Tensor<S, E, D, T>, crate::tensor::Error> {
    let (inp, mut tape) = inp.split_tape();
    let inp_ghost = inp.ghost();
    let dev = inp.device.clone();

    let inp_saved = D::BackInpNeeded::make(|| inp.clone());
    let out = inp_ghost.dev.forward(op.clone(), inp)?;
    let out_ghost = out.ghost();
    let out_saved = D::BackOutNeeded::make(|| out.clone());

    tape.add_backward_op(move |grads| {
        grads.try_alloc_for(&inp_ghost)?;
        grads.try_alloc_for(&out_ghost)?;
        let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
        dev.backward(op, inp_saved, grad_inp, out_saved, grad_out)
    });

    Ok(out.put_tape(tape))
}

pub fn try_binary_op2<
    Op: 'static + Clone,
    S: Shape,
    E: Dtype,
    D: BinaryKernel2<Op, E>,
    RhsTape,
    LhsTape: Tape<E, D> + Merge<RhsTape>,
>(
    op: Op,
    lhs: Tensor<S, E, D, LhsTape>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Result<Tensor<S, E, D, LhsTape>, crate::tensor::Error> {
    assert_eq!(lhs.shape(), rhs.shape());
    let (lhs, ltape) = lhs.split_tape();
    let (rhs, rtape) = rhs.split_tape();
    let lhs_ghost = lhs.ghost();
    let rhs_ghost = rhs.ghost();
    let mut tape = ltape.merge(rtape);
    let dev = lhs.device.clone();

    let lhs_data_saved = D::BackLhsNeeded::make(|| lhs.data.clone());
    let rhs_data_saved = D::BackRhsNeeded::make(|| rhs.data.clone());
    let out = dev.forward(op.clone(), lhs, rhs)?;
    let out_ghost = out.ghost();
    let out_saved = D::BackOutNeeded::make(|| out.clone());

    tape.add_backward_op(move |grads| {
        grads.try_alloc_for(&lhs_ghost)?;
        grads.try_alloc_for(&rhs_ghost)?;
        grads.try_alloc_for(&out_ghost)?;
        let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
        dev.backward(
            op,
            lhs_ghost,
            lhs_data_saved,
            grad_lhs,
            rhs_ghost,
            rhs_data_saved,
            grad_rhs,
            out_saved,
            grad_out,
        )
    });

    Ok(out.put_tape(tape))
}
