// storage_view.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::fmt::{Debug, Formatter};
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

        fn storage_view_from_float(
            shape: &[usize],
            init: &[f32],
            device: Device,
        ) -> Result<UniquePtr<StorageView>>;

        fn storage_view_from_int8(
            shape: &[usize],
            init: &[i8],
            device: Device,
        ) -> Result<UniquePtr<StorageView>>;

        fn storage_view_from_int16(
            shape: &[usize],
            init: &[i16],
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
pub struct StorageView {
    ptr: UniquePtr<ffi::StorageView>,
}

impl StorageView {
    /// Creates a storage view with the given shape from the given array of float values.
    pub fn from_f32(shape: &[usize], init: &[f32], device: Device) -> Result<Self> {
        Ok(Self {
            ptr: ffi::storage_view_from_float(shape, init, device)?,
        })
    }

    /// Creates a storage view with the given shape from the given array of int8 values.
    pub fn from_i8(shape: &[usize], init: &[i8], device: Device) -> Result<Self> {
        Ok(Self {
            ptr: ffi::storage_view_from_int8(shape, init, device)?,
        })
    }

    /// Creates a storage view with the given shape from the given array of int16 values.
    pub fn from_i16(shape: &[usize], init: &[i16], device: Device) -> Result<Self> {
        Ok(Self {
            ptr: ffi::storage_view_from_int16(shape, init, device)?,
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

impl Debug for StorageView {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "StorageView {{{} }}", ffi::to_string(self).replace("\n", ", "))
    }
}

impl Deref for StorageView {
    type Target = ffi::StorageView;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

unsafe impl Send for StorageView {}
unsafe impl Sync for StorageView {}

#[cfg(test)]
mod tests {
    use crate::config::Device;
    use crate::storage_view::StorageView;

    #[test]
    fn test_from_f32() {
        let shape = vec![1, 2, 4];
        let data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let v = StorageView::from_f32(&shape, &data, Default::default()).unwrap();

        assert_eq!(v.size(), data.len() as i64);
        assert_eq!(v.rank(), shape.len() as i64);
        assert!(!v.empty());
        assert_eq!(v.device(), Device::CPU);

        println!("{:?}", v);
    }

    #[test]
    fn test_from_i8() {
        let shape = vec![2, 2];
        let data = vec![3, 4, 5, 6];
        let v = StorageView::from_i8(&shape, &data, Default::default()).unwrap();

        assert_eq!(v.size(), data.len() as i64);
        assert_eq!(v.rank(), shape.len() as i64);
        assert!(!v.empty());
        assert_eq!(v.device(), Device::CPU);
    }

    #[test]
    fn test_from_i16() {
        let shape = vec![2, 2];
        let data = vec![3, 4, 5, 6];
        let v = StorageView::from_i16(&shape, &data, Default::default()).unwrap();

        assert_eq!(v.size(), data.len() as i64);
        assert_eq!(v.rank(), shape.len() as i64);
        assert!(!v.empty());
        assert_eq!(v.device(), Device::CPU);
    }
}
