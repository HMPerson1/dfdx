//! Implementations of [OwnedTape], [NoneTape], and generic Nd array containers via [Gradients].
#![allow(clippy::type_complexity)]

use rustc_hash::{FxHashMap, FxHashSet};
use std::ptr;
use std::{boxed::Box, vec::Vec};

use super::tensorlike::Tensorlike;
use super::{storage_traits::Storage, unique_id, Error, Tensor, UniqueId};
use crate::shapes::Shape;

/// A generic container for keeping gradients of tensors keyed by the
/// tensor's [UniqueId].
///
/// You can:
/// 1. Insert array values into it
/// 2. Remove entries
/// 3. Access references to arrays
/// 4. Access mutable references to arrays
#[derive(Clone, Debug)]
pub struct Gradients<E, D: Storage<E>> {
    gradient_by_id: FxHashMap<UniqueId, D::OwnedVec>,
    leaf_ids: Option<FxHashSet<UniqueId>>,
}

impl<E, D: Storage<E>> Gradients<E, D> {
    /// Creates a [Gradients] object without any leaf tensor ids.
    /// **This will never drop gradients for temporary tensors**.
    ///
    /// This is why this method is called `leaky`, because
    /// it will keep gradients from previous passes if it is
    /// used consecutively.
    pub fn leaky() -> Self {
        Self {
            gradient_by_id: Default::default(),
            leaf_ids: None,
        }
    }
}

impl<E, D: Storage<E>> Gradients<E, D> {
    /// Inserts a gradient for `t`
    pub fn try_alloc_for<S: Shape>(&mut self, t: &impl Tensorlike<S, E, D>) -> Result<(), Error> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.gradient_by_id.entry(t.id()) {
            e.insert(t.try_alloc_grad()?);
        }
        Ok(())
    }

    /// Drops all gradients except for the ids specified in the parameter.
    pub fn retain_leafs(&mut self, ids: &[UniqueId]) {
        self.leaf_ids
            .get_or_insert_with(Default::default)
            .extend(ids);
        self.drop_non_leafs();
    }

    /// Marks all existing gradients as leaf gradients.
    pub fn retain_current_grads_as_leafs(&mut self) {
        self.leaf_ids = Some(self.gradient_by_id.keys().copied().collect());
    }

    /// Keeps all gradients marked previously by [Gradients::retain_leafs], and drops all
    /// others.
    pub fn drop_non_leafs(&mut self) {
        if let Some(leafs) = &self.leaf_ids {
            self.gradient_by_id.retain(|k, _| leafs.contains(k));
        }
    }

    /// Returns a reference to the underlying gradient if found.
    pub fn get_ref_checked<S: Shape, T>(&self, t: &Tensor<S, E, D, T>) -> Option<&D::OwnedVec> {
        self.gradient_by_id.get(&t.id)
    }

    /// Retrieves mutable gradient for `t`, allocating one if it isn't present.
    pub(crate) fn get_mut<S: Shape>(
        &mut self,
        t: &impl Tensorlike<S, E, D>,
    ) -> Result<&mut D::OwnedVec, Error> {
        use std::collections::hash_map::Entry;
        match self.gradient_by_id.entry(t.id()) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => Ok(entry.insert(t.try_alloc_grad()?)),
        }
    }

    /// Returns an immutable reference to the data associated with `t`.
    ///
    /// **Panics** if data associated with `t` is not found. This indicates an unrecoverable bug.
    pub(crate) fn get_ref<S: Shape>(
        &mut self,
        t: &impl Tensorlike<S, E, D>,
    ) -> Result<&D::OwnedVec, Error> {
        self.get_mut(t).map(|x| &*x)
    }

    /// Clones the gradient and transforms it into a tensor.
    ///
    /// # Panics
    /// If no data is associated with `t` yet, this will panic due to an unwrap()
    /// on a .get() to the underlying hashmap.
    pub fn get<S: Shape>(&self, t: &impl Tensorlike<S, E, D>) -> Tensor<S, E, D> {
        let buf = self.gradient_by_id.get(&t.id()).unwrap();
        Tensor {
            id: unique_id(),
            data: t.dev().grad_to_tensor(buf),
            shape: *t.shape(),
            strides: t.strides(),
            device: t.dev().clone(),
            tape: Default::default(),
        }
    }

    /// Borrows a pair of a gradients `(&mut L, &R)`.
    /// `l` is the gradient to update, and `r` is the gradient to backprop.
    ///
    /// **Panics** if `l` and `r` have the same id.
    pub(crate) fn mut_and_ref<'a, L: Shape, R: Shape>(
        &'a mut self,
        l: &impl Tensorlike<L, E, D>,
        r: &impl Tensorlike<R, E, D>,
    ) -> Result<(&'a mut D::OwnedVec, &'a D::OwnedVec), Error> {
        self.gradient_by_id.reserve(2);
        let l_ptr = self.get_mut(l)? as *mut _;
        let r_ptr = self.get_ref(r)? as *const _;
        assert!(!ptr::eq(l_ptr, r_ptr));
        let l_ref = unsafe { &mut *l_ptr };
        let r_ref = unsafe { &*r_ptr };
        Ok((l_ref, r_ref))
    }

    /// Borrows a triplet of gradients `(&mut L1, &mut L2, &R)`.
    pub(crate) fn muts_and_ref<'a, L1: Shape, L2: Shape, R: Shape>(
        &'a mut self,
        l1: &impl Tensorlike<L1, E, D>,
        l2: &impl Tensorlike<L2, E, D>,
        r: &impl Tensorlike<R, E, D>,
    ) -> Result<(&'a mut D::OwnedVec, &'a mut D::OwnedVec, &'a D::OwnedVec), Error> {
        self.gradient_by_id.reserve(3);
        let l1_ptr = self.get_mut(l1)? as *mut _;
        let l2_ptr = self.get_mut(l2)? as *mut _;
        let r_ptr = self.get_ref(r)? as *const _;
        assert!(!ptr::eq(l1_ptr, l2_ptr));
        assert!(!ptr::eq(l1_ptr, r_ptr));
        assert!(!ptr::eq(l2_ptr, r_ptr));
        let l1_ref = unsafe { &mut *l1_ptr };
        let l2_ref = unsafe { &mut *l2_ptr };
        let r_ref = unsafe { &*r_ptr };
        Ok((l1_ref, l2_ref, r_ref))
    }

    #[inline]
    pub(crate) fn many_and_ref<'a, L: Shape, R: Shape>(
        &'a mut self,
        ls: &Vec<impl Tensorlike<L, E, D>>,
        r: &impl Tensorlike<R, E, D>,
    ) -> Result<(Vec<&'a mut D::OwnedVec>, &'a D::OwnedVec), Error> {
        self.gradient_by_id.reserve(ls.len() + 1);
        for i in 0..ls.len() {
            assert_ne!(ls[i].id(), r.id());
            for j in (i + 1)..ls.len() {
                assert_ne!(ls[i].id(), ls[j].id());
            }
        }
        let l_refs: Vec<&mut D::OwnedVec> = ls
            .iter()
            .map(|l| {
                self.get_mut(l)
                    .map(|l_ptr| unsafe { &mut *(l_ptr as *mut _) })
            })
            .collect::<Result<_, _>>()?;
        let r_ptr = self.get_ref(r)? as *const _;
        let r_ref = unsafe { &*r_ptr };
        Ok((l_refs, r_ref))
    }
}

/// Contains a [Gradients] and list of backward operations.
pub struct OwnedTape<'a, E, D: Storage<E>> {
    /// A list of (Time, BackwardOp) pairs. The Time is used to ensure operations
    /// from merged tapes are executed in the correct order.
    pub(crate) operations: Vec<(UniqueId, BackwardOp<'a, E, D, D::Allocator>), D::Allocator>,
    pub(crate) gradients: Gradients<E, D>,
}

impl<E: std::fmt::Debug, D: Storage<E> + std::fmt::Debug> std::fmt::Debug for OwnedTape<'_, E, D>
where
    <D as Storage<E>>::OwnedVec: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedTape")
            .field("num_operations", &self.operations.len())
            .field("gradients", &self.gradients)
            .finish()
    }
}

impl<E, D: Storage<E>> OwnedTape<'_, E, D> {
    pub(crate) fn from_gradients_with_device(gradients: Gradients<E, D>, dev: &D) -> Self {
        Self {
            operations: Vec::new_in(dev.allocator()),
            gradients,
        }
    }
    /// Compute the [Gradients]! This just runs all the operations on a new [Gradients] struct.
    ///
    /// Note that this method takes ownership of self, so it can't be called twice!
    pub(crate) fn execute(&mut self) -> Result<Gradients<E, D>, Error> {
        // We must ensure that the operations are sorted in execution time order.
        // Otherwise an backward operation may not be executed in the right order
        // if multiple tapes were merged together.
        self.operations.sort_by_key(|(k, _)| *k);
        // In case the same operation is present multiple times, we dedup it.
        self.operations.dedup_by_key(|(k, _)| *k);
        for (_, operation) in self.operations.drain(..).rev() {
            (operation)(&mut self.gradients)?;
        }
        Ok(std::mem::replace(&mut self.gradients, Gradients::leaky()))
    }

    pub fn hint_reserve(&mut self, ops: usize, grads: usize) {
        self.operations.reserve(ops);
        self.gradients.gradient_by_id.reserve(grads);
    }

    pub fn get_sizes(&self) -> (usize, usize) {
        (self.operations.len(), self.gradients.gradient_by_id.len())
    }
}

type BackwardOp<'a, E, D, A> = Box<dyn 'a + FnOnce(&mut Gradients<E, D>) -> Result<(), Error>, A>;

/// Contains nothing. When [Tape::add_backward_op] is called, this struct does nothing.
#[derive(Default, Debug, Clone, Copy)]
pub struct NoneTape;

/// Something that can track backward operations.
pub trait Tape<'a, E, D: Storage<E>>: Merge<Self> + Merge<NoneTape> {
    fn with_device(dev: &D) -> Self;
    /// Whether this object is currently tracking gradients. This is known at compile time.
    const OWNS_TAPE: bool;
    fn add_backward_op<F>(&mut self, operation: F)
    where
        F: 'a + FnOnce(&mut Gradients<E, D>) -> Result<(), Error>;
}

impl<'a, E, D: Storage<E>> Tape<'a, E, D> for OwnedTape<'a, E, D> {
    fn with_device(dev: &D) -> Self {
        Self::from_gradients_with_device(Gradients::leaky(), dev)
    }
    const OWNS_TAPE: bool = true;
    fn add_backward_op<F>(&mut self, operation: F)
    where
        F: 'a + FnOnce(&mut Gradients<E, D>) -> Result<(), Error>,
    {
        self.operations.push((
            unique_id(),
            Box::new_in(operation, self.operations.allocator().clone()),
        ));
    }
}

impl<'a, E, D: Storage<E>> Tape<'a, E, D> for NoneTape {
    fn with_device(_: &D) -> Self {
        Self {}
    }
    const OWNS_TAPE: bool = false;
    fn add_backward_op<F>(&mut self, _: F)
    where
        F: 'a + FnOnce(&mut Gradients<E, D>) -> Result<(), Error>,
    {
    }
}

/// Combine two things
pub trait Merge<T: ?Sized> {
    /// Merges `T` into `self`
    fn merge(self, other: T) -> Self;
}

impl Merge<NoneTape> for NoneTape {
    fn merge(self, _: NoneTape) -> Self {
        self
    }
}

impl<E, D: Storage<E>> Merge<NoneTape> for OwnedTape<'_, E, D> {
    fn merge(self, _: NoneTape) -> Self {
        self
    }
}

impl<'a, E, D: Storage<E>> Merge<OwnedTape<'a, E, D>> for OwnedTape<'a, E, D> {
    fn merge(mut self, mut other: Self) -> Self {
        self.gradients
            .gradient_by_id
            .extend(other.gradients.gradient_by_id);
        if let Some(leafs) = other.gradients.leaf_ids {
            self.gradients
                .leaf_ids
                .get_or_insert_with(Default::default)
                .extend(leafs);
        }
        self.operations.append(&mut other.operations);
        self
    }
}

#[cfg(feature = "std")]
impl<E, D: Storage<E>> Merge<NoneTape> for std::rc::Rc<std::sync::Mutex<OwnedTape<E, D>>> {
    fn merge(self, _: NoneTape) -> Self {
        self
    }
}

#[cfg(feature = "std")]
impl<E, D: Storage<E>> Merge<Self> for std::rc::Rc<std::sync::Mutex<OwnedTape<E, D>>> {
    fn merge(self, other: Self) -> Self {
        if !std::rc::Rc::ptr_eq(&self, &other) {
            let mut lhs = self.lock().unwrap();
            let mut rhs = other.lock().unwrap();
            lhs.gradients
                .gradient_by_id
                .extend(std::mem::take(&mut rhs.gradients.gradient_by_id));
            if let Some(leafs) = &mut rhs.gradients.leaf_ids {
                lhs.gradients
                    .leaf_ids
                    .get_or_insert_with(Default::default)
                    .extend(std::mem::take(leafs));
            }
            lhs.operations.append(&mut rhs.operations);
        }
        self
    }
}

#[cfg(feature = "std")]
impl<E, D: Storage<E>> Tape<E, D> for std::rc::Rc<std::sync::Mutex<OwnedTape<E, D>>> {
    const OWNS_TAPE: bool = true;
    fn add_backward_op<F>(&mut self, operation: F)
    where
        F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), Error>,
    {
        let mut tape = self.lock().unwrap();
        tape.add_backward_op(operation);
    }
}
