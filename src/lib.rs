// lib.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This crate provides Rust bindings for [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2).
//!
//! # Examples
//! The following example translates English to German and Japanese using the tokenizer provided by
//! the Hugging Face's [`tokenizers` crate](https://docs.rs/tokenizers/).
//! ```no_run
//! # use anyhow::Result;
//!
//! use ct2rs::{TranslationOptions, Translator};
//! use ct2rs::config::Config;
//! use ct2rs::tokenizers::Tokenizer;
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
//! The following example generates text using the tokenizer provided by
//! [Sentencepiece crate](https://docs.rs/sentencepiece/).
//! ```no_run
//! # use anyhow::Result;
//! use ct2rs::config::{Config, Device};
//! use ct2rs::{Generator, GenerationOptions};
//! use ct2rs::sentencepiece::Tokenizer;
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
//! Please also see the other sample code available in the
//! [examples directory](https://github.com/jkawamoto/ctranslate2-rs/tree/main/examples).

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::Path;

use anyhow::{anyhow, Result};

use crate::auto::Tokenizer as AutoTokenizer;
pub use crate::config::{set_log_level, set_random_seed};
use crate::config::Config;
pub use crate::generator::GenerationOptions;
pub use crate::translator::TranslationOptions;
pub use crate::types::ffi::GenerationStepResult;

pub mod auto;
pub mod bpe;
pub mod config;
pub mod generator;
pub mod sentencepiece;
pub mod tokenizers;
pub mod translator;
mod types;

/// Defines the necessary functions for a tokenizer.
///
/// This trait provides the core functionality needed to convert strings to sequences of tokens
/// and vice versa. It is essential for text processing tasks such as natural language processing,
/// where text needs to be broken down into manageable pieces or reconstructed from tokenized forms.
///
/// Currently, this crate implements two tokenizers:
/// * [`tokenizers::Tokenizer`]: the tokenizer provided by the Hugging Face's
///   [`tokenizers` crate](https://docs.rs/tokenizers/),
/// * [`sentencepiece::Tokenizer`]: the tokenizer based on
///   [Sentencepiece crate](https://docs.rs/sentencepiece/).
pub trait Tokenizer {
    /// Encodes a given string into a sequence of tokens.
    ///
    /// This function takes a reference to a string and returns a vector of token strings
    /// resulting from the tokenization process.
    ///
    /// # Arguments
    /// * `input` - A reference to the string to be tokenized.
    ///
    /// # Returns
    /// A `Result` containing either the vector of tokens if successful or an error if the tokenization fails.
    fn encode(&self, input: &str) -> Result<Vec<String>>;

    /// Decodes a given sequence of tokens back into a single string.
    ///
    /// This function takes a vector of token strings and reconstructs the original string.
    ///
    /// # Arguments
    /// * `tokens` - A vector of strings representing the tokens to be decoded.
    ///
    /// # Returns
    /// A `Result` containing either the reconstructed string if successful or an error if the decoding fails.
    fn decode(&self, tokens: Vec<String>) -> Result<String>;
}

#[inline]
fn encode_strings<T: Tokenizer, U: AsRef<str>>(
    tokenizer: &T,
    sources: &Vec<U>,
) -> Result<Vec<Vec<String>>> {
    sources
        .into_iter()
        .map(|s| tokenizer.encode(s.as_ref()))
        .collect::<Result<Vec<Vec<String>>>>()
}

/// A text translator with a tokenizer.
pub struct Translator<T: Tokenizer> {
    translator: translator::Translator,
    tokenizer: T,
}

impl Translator<AutoTokenizer> {
    /// Initializes the translator with [`auto::Tokenizer`].
    pub fn new<U: AsRef<Path>>(path: U, config: &Config) -> Result<Self> {
        Self::with_tokenizer(&path, AutoTokenizer::new(&path)?, config)
    }
}

impl<T: Tokenizer> Translator<T> {
    /// Initializes the translator with the given tokenizer.
    pub fn with_tokenizer<U: AsRef<Path>>(path: U, tokenizer: T, config: &Config) -> Result<Self> {
        Ok(Translator {
            translator: translator::Translator::new(path, config)?,
            tokenizer,
        })
    }

    /// Translates a batch of strings.
    pub fn translate_batch<'a, U, V>(
        &self,
        sources: &Vec<U>,
        options: &TranslationOptions<V>,
        callback: Option<&'a mut dyn FnMut(GenerationStepResult) -> bool>,
    ) -> Result<Vec<(String, Option<f32>)>>
    where
        U: AsRef<str>,
        V: AsRef<str>,
    {
        let output = self.translator.translate_batch(
            &encode_strings(&self.tokenizer, sources)?,
            options,
            callback,
        )?;

        let mut res = Vec::new();
        for r in output.into_iter() {
            let score = r.score();
            let h = r
                .hypotheses
                .into_iter()
                .next()
                .ok_or(anyhow!("no results are returned"))?;
            res.push((
                self.tokenizer
                    .decode(h.into_iter().collect())
                    .map_err(|err| anyhow!("failed to decode: {err}"))?,
                score,
            ));
        }
        Ok(res)
    }

    /// Translates a batch of strings using target prefixes.
    pub fn translate_batch_with_target_prefix<'a, U, V, W>(
        &self,
        sources: &Vec<U>,
        target_prefixes: &Vec<Vec<V>>,
        options: &TranslationOptions<W>,
        callback: Option<&'a mut dyn FnMut(GenerationStepResult) -> bool>,
    ) -> Result<Vec<(String, Option<f32>)>>
    where
        U: AsRef<str>,
        V: AsRef<str>,
        W: AsRef<str>,
    {
        let output = self.translator.translate_batch_with_target_prefix(
            &encode_strings(&self.tokenizer, sources)?,
            &target_prefixes,
            options,
            callback,
        )?;

        let mut res = Vec::new();
        for (r, prefix) in output.into_iter().zip(target_prefixes) {
            let score = r.score();
            let h = r
                .hypotheses
                .into_iter()
                .next()
                .ok_or(anyhow!("no results are returned"))?;
            res.push((
                self.tokenizer
                    .decode(h.into_iter().skip(prefix.len()).collect())
                    .map_err(|err| anyhow!("failed to decode: {err}"))?,
                score,
            ));
        }
        Ok(res)
    }

    /// Number of batches in the work queue.
    #[inline]
    pub fn num_queued_batches(&self) -> Result<usize> {
        self.translator.num_queued_batches()
    }

    /// Number of batches in the work queue or currently processed by a worker.
    #[inline]
    pub fn num_active_batches(&self) -> Result<usize> {
        self.translator.num_active_batches()
    }

    /// Number of parallel replicas.
    #[inline]
    pub fn num_replicas(&self) -> Result<usize> {
        self.translator.num_replicas()
    }
}

/// A text generator with a tokenizer.
pub struct Generator<T: Tokenizer> {
    generator: generator::Generator,
    tokenizer: T,
}

impl Generator<AutoTokenizer> {
    /// Initializes the generator with [`auto::Tokenizer`].
    pub fn new<T: AsRef<Path>>(path: T, config: &Config) -> Result<Self> {
        Self::with_tokenizer(&path, AutoTokenizer::new(&path)?, config)
    }
}

impl<T: Tokenizer> Generator<T> {
    /// Initializes the generator with the given tokenizer.
    pub fn with_tokenizer<U: AsRef<Path>>(path: U, tokenizer: T, config: &Config) -> Result<Self> {
        Ok(Generator {
            generator: generator::Generator::new(path, config)?,
            tokenizer,
        })
    }

    /// Generate texts with the given prompts.
    pub fn generate_batch<'a, U, V, W>(
        &self,
        prompts: &Vec<U>,
        options: &GenerationOptions<V, W>,
        callback: Option<&'a mut dyn FnMut(GenerationStepResult) -> bool>,
    ) -> Result<Vec<(Vec<String>, Vec<f32>)>>
    where
        U: AsRef<str>,
        V: AsRef<str>,
        W: AsRef<str>,
    {
        let output = self.generator.generate_batch(
            &encode_strings(&self.tokenizer, prompts)?,
            options,
            callback,
        )?;

        let mut res = Vec::new();
        for r in output.into_iter() {
            let sequence = r
                .sequences
                .into_iter()
                .map(|seq| self.tokenizer.decode(seq))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| anyhow!("failed to decode: {err}"))?;
            let scores = r.scores;
            res.push((sequence, scores))
        }
        Ok(res)
    }

    /// Number of batches in the work queue.
    #[inline]
    pub fn num_queued_batches(&self) -> Result<usize> {
        self.generator.num_queued_batches()
    }

    /// Number of batches in the work queue or currently processed by a worker.
    #[inline]
    pub fn num_active_batches(&self) -> Result<usize> {
        self.generator.num_active_batches()
    }

    /// Number of parallel replicas.
    #[inline]
    pub fn num_replicas(&self) -> Result<usize> {
        self.generator.num_replicas()
    }
}
