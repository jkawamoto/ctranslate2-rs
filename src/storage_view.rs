// storage_view.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This module provides a Rust binding to the
//! [`ctranslate2::StorageView`](https://opennmt.net/CTranslate2/python/ctranslate2.StorageView.html).

use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;

use anyhow::Result;
use cxx::UniquePtr;

use crate::config::Device;

#[cxx::bridge]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("ct2rs/include/storage_view.h");

        type Device = crate::config::ffi::Device;

        type StorageView;

        fn storage_view(
            shape: &[usize],
            init: &mut [f32],
            device: Device,
        ) -> Result<UniquePtr<StorageView>>;

        fn device(self: &StorageView) -> Device;

        fn size(self: &StorageView) -> i64;

        fn rank(self: &StorageView) -> i64;

        fn to_string(storage: &StorageView) -> String;
    }
}

/// A Rust binding to the
/// [`ctranslate2::StorageView`](https://opennmt.net/CTranslate2/python/ctranslate2.StorageView.html).
pub struct StorageView<'a> {
    ptr: UniquePtr<ffi::StorageView>,
    phantom: PhantomData<&'a [f32]>,
}

impl<'a> StorageView<'a> {
    /// Creates a storage view with the given shape from the given array of float values.
    pub fn new(shape: &[usize], init: &'a mut [f32], device: Device) -> Result<Self> {
        Ok(Self {
            ptr: ffi::storage_view(shape, init, device)?,
            phantom: PhantomData,
        })
    }

    /// Device where the storage is allocated.
    pub fn device(&self) -> Device {
        self.ptr.device()
    }

    /// Returns the size of this storage.
    pub fn size(&self) -> i64 {
        self.ptr.size()
    }

    /// Returns the rank of this storage.
    pub fn rank(&self) -> i64 {
        self.ptr.rank()
    }

    /// Returns true if this storage is empty.
    pub fn empty(&self) -> bool {
        self.size() == 0
    }
}

impl Debug for StorageView<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StorageView {{{} }}",
            ffi::to_string(self).replace("\n", ", ")
        )
    }
}

impl Deref for StorageView<'_> {
    type Target = ffi::StorageView;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

unsafe impl Send for StorageView<'_> {}
unsafe impl Sync for StorageView<'_> {}

#[cfg(test)]
mod tests {
    use crate::config::Device;
    use crate::storage_view::StorageView;

    #[test]
    fn test_constructor() {
        let shape = vec![1, 2, 4];
        let mut data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let size = data.len() as i64;
        let rank = shape.len() as i64;

        let v = StorageView::new(&shape, &mut data, Default::default()).unwrap();

        assert_eq!(v.size(), size);
        assert_eq!(v.rank(), rank);
        assert!(!v.empty());
        assert_eq!(v.device(), Device::CPU);

        println!("{:?}", v);
    }
}
