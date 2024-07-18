// lib.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This crate provides Rust bindings for [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2).
//!
//! This crate provides the following:
//!
//! * Rust bindings for
//!   [ctranslate2::Translator](https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html),
//!   [ctranslate2::Generator](https://opennmt.net/CTranslate2/python/ctranslate2.Generator.html),
//!   and
//!   [ctranslate2::Whisper](https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html)
//!   provided by CTranslate2, specifically [`sys::Translator`], [`sys::Generator`], and
//!   [`sys::Whisper`].
//! * More user-friendly versions of these, [`Translator`], [`Generator`], and [`Whisper`],
//!   which incorporate tokenizers for easier handling.
//!
//! # Basic Usage
//! The following example translates two strings using default settings and outputs each to
//! the standard output.
//!
//! ```no_run
//! # use anyhow::Result;
//! #
//! use ct2rs::{Config, Translator, TranslationOptions, GenerationStepResult};
//!
//! # fn main() -> Result<()> {
//! let sources = vec![
//!     "Hallo World!",
//!     "This crate provides Rust bindings for CTranslate2."
//! ];
//! let translator = Translator::new("/path/to/model", &Default::default())?;
//! let results = translator.translate_batch(&sources, &Default::default(), None)?;
//! for (r, _) in results{
//!     println!("{}", r);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Supported Models
//! The `ct2rs` crate has been tested and confirmed to work with the following models:
//!
//! - BART
//! - BLOOM
//! - FALCON
//! - Marian-MT
//! - MPT
//! - NLLB
//! - GPT-2
//! - GPT-J
//! - OPT
//! - T5
//! - Whisper
//!
//! Please see the respective
//! [examples](https://github.com/jkawamoto/ctranslate2-rs/tree/main/examples) for each model.
//!
//! # Stream API
//! This crate also offers a streaming API that utilizes callback closures.
//! Please refer to
//! [the example code](https://github.com/jkawamoto/ctranslate2-rs/blob/main/examples/stream.rs)
//! for more information.
//!

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

pub use generator::{GenerationOptions, Generator};
pub use result::GenerationStepResult;
pub use sys::{set_log_level, set_random_seed, BatchType, ComputeType, Config, Device, LogLevel};
pub use tokenizer::Tokenizer;
pub use translator::{TranslationOptions, Translator};
pub use whisper::{Whisper, WhisperOptions};

mod generator;
mod result;
pub mod sys;
mod tokenizer;
pub mod tokenizers;
mod translator;
mod whisper;
