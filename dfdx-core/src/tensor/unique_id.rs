use std::{array, iter};

/// An id used in to associate gradients with Tensors.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct UniqueId(usize);

/// Generate a [UniqueId].
pub(crate) fn unique_id() -> UniqueId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    UniqueId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

/// An id used to order and deduplicate gradient tape operations
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub(crate) struct BackOpUniqueId(usize);

/// Generate a [BackOpUniqueId].
pub(crate) fn unique_id_b() -> BackOpUniqueId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    BackOpUniqueId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

#[derive(Clone, Debug)]
pub(crate) struct IdMap<V> {
    /// index =~= max - uniqueid
    inner: Vec<Option<V>>,
    /// invariant: valid only if `inner` is non-empty
    max: usize,
}

impl<V> IdMap<V> {
    pub(crate) fn new() -> Self {
        Self {
            inner: Vec::new(),
            max: 0,
        }
    }

    pub(crate) fn entry(&mut self, k: UniqueId) -> Entry<'_, V> {
        if self.inner.is_empty() {
            self.max = k.0;
        }
        if k.0 > self.max {
            // afaik this branch is never taken, but, for completeness, we handle it anyway
            self.reindex(k);
        }
        let i = self.max - k.0;
        if i >= self.inner.len() {
            self.expand_to_index(i);
        }

        Entry::make(&mut self.inner[i])
    }

    /// Panics if `ks` are not disjoint
    #[inline]
    pub(crate) fn entries_disjoint<const N: usize>(
        &mut self,
        ks: [UniqueId; N],
    ) -> [Entry<'_, V>; N] {
        if N == 0 {
            return array::from_fn(|_| unreachable!());
        }

        let k_max = ks.iter().copied().max().unwrap();
        if self.inner.is_empty() {
            self.max = k_max.0;
        }
        if k_max.0 > self.max {
            // afaik this branch is never taken, but, for completeness, we handle it anyway
            self.reindex(k_max);
        }
        let k_min = ks.iter().copied().min().unwrap();
        let i_max = self.max - k_min.0;
        if i_max >= self.inner.len() {
            self.expand_to_index(i_max);
        }

        self.inner
            .get_disjoint_mut(ks.map(|k| self.max - k.0))
            .unwrap()
            .map(|e| Entry::make(e))
    }

    #[inline(always)]
    fn reindex(&mut self, k: UniqueId) {
        // this function is ~never called, but it's still `#[inline(always)]
        // to ensure that the write to `self.max` is visible to callers.
        // the actual work is in the function marked `#[inline(never)]`.

        #[inline(never)]
        #[cold]
        fn do_reindex<V>(this: &mut IdMap<V>, k: UniqueId) {
            // wastes space if `inner` begins with a bunch of empty slots, but meh...
            let shift = k.0 - this.max;
            this.inner
                .splice(0..0, iter::repeat_with(|| None).take(shift));
        }

        do_reindex(self, k);
        self.max = k.0;
    }

    #[cold]
    fn expand_to_index(&mut self, i_max: usize) {
        let new_len = i_max.checked_add(1).unwrap();
        // eagerly fill any spare capacity if available
        let new_len = new_len.max(self.inner.capacity());
        self.inner.resize_with(new_len, || None);
    }

    #[inline]
    pub(crate) fn extend(&mut self, other: Self) {
        if other.inner.is_empty() {
            // apparently this is literally always true,
            // so it's worth explicitly special-casing as a no-op
        } else if self.inner.is_empty() {
            *self = other;
        } else {
            self.do_extend(other);
        }
    }

    #[cold]
    fn do_extend(&mut self, other: Self) {
        // could be more efficient, but meh...
        for (oi, oe) in other.inner.into_iter().enumerate() {
            if let Some(ov) = oe {
                self.entry(UniqueId(other.max - oi)).insert(ov);
            }
        }
    }

    pub(crate) fn get(&self, k: UniqueId) -> Option<&V> {
        if self.inner.is_empty() {
            return None;
        }
        let i = self.max - k.0;
        self.inner.get(i).and_then(|e| e.as_ref())
    }

    pub(crate) fn get_mut(&mut self, k: UniqueId) -> Option<&mut V> {
        if self.inner.is_empty() {
            return None;
        }
        let i = self.max - k.0;
        self.inner.get_mut(i).and_then(|e| e.as_mut())
    }

    pub(crate) fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }
}

pub(crate) enum Entry<'a, V> {
    Occupied(OccupiedEntry<'a, V>),
    Vacant(VacantEntry<'a, V>),
}

impl<'a, V> Entry<'a, V> {
    fn make(a: &'a mut Option<V>) -> Self {
        match a {
            Some(v) => Entry::Occupied(OccupiedEntry(v)),
            e @ None => Entry::Vacant(VacantEntry(e)),
        }
    }

    pub(crate) fn insert(self, value: V) -> OccupiedEntry<'a, V> {
        OccupiedEntry(match self {
            Entry::Occupied(OccupiedEntry(v)) => {
                *v = value;
                v
            }
            Entry::Vacant(e) => e.insert(value),
        })
    }
}

pub(crate) struct OccupiedEntry<'a, V>(&'a mut V);
impl<'a, V> OccupiedEntry<'a, V> {
    pub(crate) fn into_mut(self) -> &'a mut V {
        self.0
    }
}
pub(crate) struct VacantEntry<'a, V>(&'a mut Option<V>);
impl<'a, V> VacantEntry<'a, V> {
    pub(crate) fn insert(self, value: V) -> &'a mut V {
        self.0.insert(value)
    }
}
