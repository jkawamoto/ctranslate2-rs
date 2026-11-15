//! Raw Rust bindings for OpenNMT/CTranslate2.

// `intel-onemkl-prebuild` and `onednn-src` link their libraries through their
// build scripts, and rustc only links a dependency that the crate references.
#[cfg(feature = "mkl")]
extern crate intel_onemkl_prebuild;
#[cfg(feature = "dnnl")]
extern crate onednn_src;

mod sys;

pub use sys::*;
