mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

#[cfg(feature = "webgpu")]
mod webgpu_kernel;

use crate::{
    shapes::{Dtype, Shape},
    tensor::*,
};

use super::WeightDecay;

/// Configuration of hyperparameters for RMSprop.
#[derive(Debug, Clone, Copy)]
pub struct RMSpropConfig {
    /// Learning rate. Defaults to `1e-2`.
    pub lr: f64,

    /// Value for exponential moving average. Defaults to `0.9`.
    pub alpha: f64,

    /// Epsilon for stability. Defaults to `1e-8`.
    pub eps: f64,

    /// Optional momentum. Defaults to `None`.
    pub momentum: Option<f64>,

    /// Whether the avg should be centered by the grad's avg value.
    /// Defaults to `false`.
    pub centered: bool,

    /// Optional weight decay. Defaults to `None`.
    pub weight_decay: Option<WeightDecay>,
}

impl Default for RMSpropConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-8,
            momentum: None,
            centered: false,
            weight_decay: None,
        }
    }
}

pub trait RMSpropKernel<E: Dtype>: Storage<E> {
    fn rmsprop_kernel(
        &self,
        cfg: &RMSpropConfig,
        param: &mut Self::SharedVec,
        momentum: &mut Self::OwnedVec,
        square_avg: &mut Self::OwnedVec,
        grad_avg: &mut Self::OwnedVec,
        grad: &Self::OwnedVec,
    ) -> Result<(), Error>;
}

impl RMSpropConfig {
    /// Update a single tensor using RMSprop.
    pub fn try_update<S: Shape, E: Dtype, D: RMSpropKernel<E>>(
        &self,
        param: &mut Tensor<S, E, D>,
        momentum: &mut D::OwnedVec,
        square_avg: &mut D::OwnedVec,
        grad_avg: &mut D::OwnedVec,
        grad: &D::OwnedVec,
    ) -> Result<(), crate::tensor::Error> {
        param.device.rmsprop_kernel(
            self,
            &mut param.data,
            momentum,
            square_avg,
            grad_avg,
            grad,
        )
    }
}
