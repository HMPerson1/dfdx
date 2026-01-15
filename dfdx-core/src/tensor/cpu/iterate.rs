use super::{super::Tensor, Cpu};
use crate::shapes::{Shape, Unit};
use std::{alloc::Allocator, ops};

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct NdIndex<S: Shape> {
    pub(crate) inner: ops::Range<usize>,
    pub(crate) shape: S::Concrete,
    pub(crate) strides: S::Concrete,
    pub(crate) contiguous: bool,
}

impl<S: Shape> NdIndex<S> {
    #[inline]
    pub(crate) fn new(shape: S, strides: S::Concrete) -> Self {
        Self {
            inner: ops::Range {
                start: 0,
                end: shape.num_elements(),
            },
            shape: shape.concrete(),
            strides,
            contiguous: strides == shape.strides(),
        }
    }
}

impl<S: Shape> NdIndex<S> {
    /// equivalent to `self.apply_strides(self.to_nd_index(i_contig))`
    #[inline]
    pub(crate) fn get_strided_index(&self, idx: usize) -> usize {
        if self.contiguous {
            idx
        } else {
            self.get_strided_index_slow(idx)
        }
    }

    #[inline]
    pub(crate) fn get_strided_index_slow(&self, mut i_contig: usize) -> usize {
        let mut out = 0;

        for ax in 0..S::NUM_DIMS {
            let ax_size = self.shape[S::NUM_DIMS - 1 - ax];
            let ax_stride = self.strides[S::NUM_DIMS - 1 - ax];
            out += (i_contig % ax_size) * ax_stride;
            i_contig /= ax_size;
        }

        out
    }

    #[inline]
    pub(crate) fn to_nd_index(&self, mut i_contig: usize) -> S::Concrete {
        let mut ret = S::Concrete::default();

        for ax in 0..S::NUM_DIMS {
            let ax_size = self.shape[S::NUM_DIMS - 1 - ax];
            ret[S::NUM_DIMS - 1 - ax] = i_contig % ax_size;
            i_contig /= ax_size;
        }

        ret
    }

    #[inline]
    pub(crate) fn apply_strides(&self, idx: S::Concrete) -> usize {
        idx.into_iter()
            .zip(self.strides)
            .map(|(i, s)| i * s)
            .sum()
    }

    #[inline(always)]
    pub(crate) fn next(&mut self) -> Option<usize> {
        self.inner.next().map(|i| self.get_strided_index(i))
    }

    #[inline(always)]
    pub(crate) fn next_with_idx(&mut self) -> Option<(usize, S::Concrete)> {
        self.inner.next().map(|i_contig| {
            let idx = self.to_nd_index(i_contig);
            (self.apply_strides(idx), idx)
        })
    }
}

pub(crate) struct StridedRefIter<'a, S: Shape, E> {
    data: &'a [E],
    index: NdIndex<S>,
}

pub(crate) struct StridedMutIter<'a, S: Shape, E> {
    data: &'a mut [E],
    index: NdIndex<S>,
}

pub(crate) struct StridedRefIndexIter<'a, S: Shape, E> {
    data: &'a [E],
    index: NdIndex<S>,
}

pub(crate) struct StridedMutIndexIter<'a, S: Shape, E> {
    data: &'a mut [E],
    index: NdIndex<S>,
}

impl<S: Shape, E: Unit, T, A: Allocator + Clone> Tensor<S, E, Cpu<A>, T> {
    #[inline]
    pub(crate) fn buf_iter(&self) -> std::slice::Iter<'_, E> {
        self.data.iter()
    }

    #[inline]
    pub(crate) fn buf_iter_mut(&mut self) -> std::slice::IterMut<'_, E> {
        std::rc::Rc::make_mut(&mut self.data).iter_mut()
    }

    #[inline]
    pub(crate) fn iter(&self) -> StridedRefIter<'_, S, E> {
        StridedRefIter {
            data: self.data.as_ref(),
            index: NdIndex::new(self.shape, self.strides),
        }
    }

    #[inline]
    pub(crate) fn iter_mut(&mut self) -> StridedMutIter<'_, S, E> {
        StridedMutIter {
            data: &mut std::rc::Rc::make_mut(&mut self.data)[..],
            index: NdIndex::new(self.shape, self.strides),
        }
    }

    #[inline]
    pub(crate) fn iter_with_index(&self) -> StridedRefIndexIter<'_, S, E> {
        StridedRefIndexIter {
            data: self.data.as_ref(),
            index: NdIndex::new(self.shape, self.strides),
        }
    }

    #[inline]
    pub(crate) fn iter_mut_with_index(&mut self) -> StridedMutIndexIter<'_, S, E> {
        StridedMutIndexIter {
            data: &mut std::rc::Rc::make_mut(&mut self.data)[..],
            index: NdIndex::new(self.shape, self.strides),
        }
    }
}

pub(crate) trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;
    fn next(&'_ mut self) -> Option<Self::Item<'_>>;
}

impl<'q, S: Shape, E> LendingIterator for StridedRefIter<'q, S, E> {
    type Item<'a> = &'a E where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index.next().map(|i| &self.data[i])
    }
}

impl<'q, S: Shape, E> LendingIterator for StridedMutIter<'q, S, E> {
    type Item<'a> = &'a mut E where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index.next().map(|i| &mut self.data[i])
    }
}

impl<'q, S: Shape, E> LendingIterator for StridedRefIndexIter<'q, S, E> {
    type Item<'a> = (&'a E, S::Concrete) where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index
            .next_with_idx()
            .map(|(i, idx)| (&self.data[i], idx))
    }
}

impl<'q, S: Shape, E> LendingIterator for StridedMutIndexIter<'q, S, E> {
    type Item<'a> = (&'a mut E, S::Concrete) where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index
            .next_with_idx()
            .map(|(i, idx)| (&mut self.data[i], idx))
    }
}

#[cfg(test)]
mod tests {
    use crate::shapes::{Rank1, Rank2, Rank3};

    use super::*;

    #[test]
    fn test_0d_contiguous_iter() {
        let mut i = NdIndex::new((), ().strides());
        assert_eq!(i.next(), Some(0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_1d_contiguous_iter() {
        let shape: Rank1<3> = Default::default();
        let mut i = NdIndex::new(shape, shape.strides());
        assert_eq!(i.next(), Some(0));
        assert_eq!(i.next(), Some(1));
        assert_eq!(i.next(), Some(2));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_contiguous_iter() {
        let shape: Rank2<2, 3> = Default::default();
        let mut i = NdIndex::new(shape, shape.strides());
        assert_eq!(i.next(), Some(0));
        assert_eq!(i.next(), Some(1));
        assert_eq!(i.next(), Some(2));
        assert_eq!(i.next(), Some(3));
        assert_eq!(i.next(), Some(4));
        assert_eq!(i.next(), Some(5));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_broadcasted_0_iter() {
        let shape: Rank2<2, 3> = Default::default();
        let mut i = NdIndex::new(shape, [0, 1]);
        assert_eq!(i.next(), Some(0));
        assert_eq!(i.next(), Some(1));
        assert_eq!(i.next(), Some(2));
        assert_eq!(i.next(), Some(0));
        assert_eq!(i.next(), Some(1));
        assert_eq!(i.next(), Some(2));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_broadcasted_1_iter() {
        let shape: Rank2<2, 3> = Default::default();
        let mut i = NdIndex::new(shape, [1, 0]);
        assert_eq!(i.next(), Some(0));
        assert_eq!(i.next(), Some(0));
        assert_eq!(i.next(), Some(0));
        assert_eq!(i.next(), Some(1));
        assert_eq!(i.next(), Some(1));
        assert_eq!(i.next(), Some(1));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_permuted_iter() {
        let shape: Rank2<3, 2> = Default::default();
        let mut i = NdIndex::new(shape, [1, 3]);
        assert_eq!(i.next(), Some(0));
        assert_eq!(i.next(), Some(3));
        assert_eq!(i.next(), Some(1));
        assert_eq!(i.next(), Some(4));
        assert_eq!(i.next(), Some(2));
        assert_eq!(i.next(), Some(5));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_3d_broadcasted_iter() {
        let shape: Rank3<3, 1, 2> = Default::default();
        let mut i = NdIndex::new(shape, [2, 0, 1]);
        assert_eq!(i.next(), Some(0));
        assert_eq!(i.next(), Some(1));
        assert_eq!(i.next(), Some(2));
        assert_eq!(i.next(), Some(3));
        assert_eq!(i.next(), Some(4));
        assert_eq!(i.next(), Some(5));
        assert!(i.next().is_none());
    }
}
