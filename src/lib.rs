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
//!   [Translator](https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html) and
//!   [Generator](https://opennmt.net/CTranslate2/python/ctranslate2.Generator.html) provided by
//!   CTranslate2, specifically [`sys::Translator`] and [`sys::Generator`].
//! * More user-friendly versions of these, [`Translator`] and [`Generator`],
//!   which incorporate tokenizers for easier handling.
//!
//! # Tokenizers
//! Both [`sys::Translator`] and [`sys::Generator`] work with sequences of tokens.
//! To handle human-readable strings, a tokenizer is necessary.
//! The [`Translator`] and [`Generator`] utilize Hugging Face and SentencePiece tokenizers
//! to convert between strings and token sequences.
//! The [`tokenizers::auto::Tokenizer`] automatically determines which tokenizer to use and constructs it
//! appropriately.
//!
//! ## Example:
//! ### [tokenizers::auto::Tokenizer]
//! Here is an example of using [`tokenizers::auto::Tokenizer`] to build a Translator and translate a string:
//!
//! ```no_run
//! # use anyhow::Result;
//! #
//! use ct2rs::{Config, Translator};
//!
//! # fn main() -> Result<()> {
//! // Translator::new creates a translator instance with auto::Tokenizer.
//! let t = Translator::new("/path/to/model", &Config::default())?;
//! let res = t.translate_batch(
//!     &vec!["Hallo World!"],
//!     &Default::default(),
//!     None,
//! )?;
//! for r in res {
//!     println!("{:?}", r);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### [tokenizers::hf::Tokenizer]
//! The following example translates English to German and Japanese using the tokenizer provided by
//! the Hugging Face's [`tokenizers` crate](https://docs.rs/tokenizers/).
//! ```no_run
//! # use anyhow::Result;
//!
//! use ct2rs::{Config, TranslationOptions, Translator};
//! use ct2rs::tokenizers::hf::Tokenizer;
//!
//! # fn main() -> Result<()> {
//! let path = "/path/to/model";
//! let t = Translator::with_tokenizer(&path, Tokenizer::new(&path)?, &Config::default())?;
//! let res = t.translate_batch_with_target_prefix(
//!     &vec![
//!         "Hello world!",
//!         "This library provides Rust bindings for CTranslate2.",
//!     ],
//!     &vec![vec!["deu_Latn"], vec!["jpn_Jpan"]],
//!     &TranslationOptions {
//!         return_scores: true,
//!         ..Default::default()
//!     },
//!     None
//! )?;
//! for r in res {
//!     println!("{}, (score: {:?})", r.0, r.1);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### [tokenizers::sentencepiece::Tokenizer]
//! The following example generates text using the tokenizer provided by
//! [Sentencepiece crate](https://docs.rs/sentencepiece/).
//! ```no_run
//! # use anyhow::Result;
//! use ct2rs::{Config, Device, Generator, GenerationOptions};
//! use ct2rs::tokenizers::sentencepiece::Tokenizer;
//!
//! # fn main() -> Result<()> {
//! let path = "/path/to/model";
//! let g = Generator::with_tokenizer(&path, Tokenizer::new(&path)?, &Config::default())?;
//! let res = g.generate_batch(
//!     &vec!["prompt"],
//!     &GenerationOptions::default(),
//!     None,
//! )?;
//! for r in res {
//!     println!("{:?}", r.0);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Supported Models
//! The `ct2rs` crate has been tested and confirmed to work with the following models:
//!
//! * BART
//! * BLOOM
//! * FALCON
//! * Marian-MT
//! * MPT
//! * NLLB
//! * GPT-2
//! * GPT-J
//! * OPT
//! * T5
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

pub use generator::Generator;
pub use result::GenerationStepResult;
pub use sys::{
    set_log_level, set_random_seed, BatchType, ComputeType, Config, Device, GenerationOptions,
    LogLevel, TranslationOptions,
};
pub use tokenizer::Tokenizer;
pub use translator::Translator;

mod generator;
mod result;
pub mod sys;
mod tokenizer;
pub mod tokenizers;
mod translator;
