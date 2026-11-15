// lib.rs
//
// Copyright (c) 2023-2026 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! Raw Rust bindings for OpenNMT/CTranslate2.

// These crates link their libraries through their build scripts, and rustc only
// links a dependency that the crate references.
#[cfg(feature = "mkl")]
extern crate intel_onemkl_prebuild;
#[cfg(feature = "dnnl")]
extern crate onednn_src;
#[cfg(feature = "openblas")]
extern crate openblas_src;

mod sys;

pub use sys::*;
