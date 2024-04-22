// types.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This module defines common structures.

#[cxx::bridge]
pub(crate) mod ffi {
    pub struct VecString {
        v: Vec<String>,
    }

    pub struct VecStr<'a> {
        v: Vec<&'a str>,
    }

    pub struct VecUSize {
        v: Vec<usize>,
    }

    struct _dummy<'a>{
        _vec_string: Vec<VecString>,
        _vec_str: Vec<VecStr<'a>>,
        _vec_usize: Vec<VecUSize>
    }
}