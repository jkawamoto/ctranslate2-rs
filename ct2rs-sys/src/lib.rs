//! Raw Rust bindings for OpenNMT/CTranslate2.

// `intel-onemkl-prebuild` links MKL through its build script, and rustc only
// links a dependency that the crate references.
#[cfg(feature = "mkl")]
extern crate intel_onemkl_prebuild;

mod sys;

pub use sys::*;
