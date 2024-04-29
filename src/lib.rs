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
//! The following example translates English to German and Japanese.
//! ```no_run
//! # use anyhow::Result;
//! use ct2rs::config::{Config, Device};
//! use ct2rs::{TranslationOptions, Translator};
//!
//! # fn main() -> Result<()> {
//! let t = Translator::new("/path/to/model", Config::default())?;
//! let res = t.translate_batch_with_target_prefix(
//!     vec![
//!         "Hello world!",
//!         "This library provides Rust bindings for CTranslate2.",
//!     ],
//!     vec![vec!["deu_Latn"], vec!["jpn_Jpan"]],
//!     &TranslationOptions {
//!         return_scores: true,
//!         ..Default::default()
//!     },
//! )?;
//! for r in res {
//!     println!("{}, (score: {:?})", r.0, r.1);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! The following example generates text.
//! ```no_run
//! # use anyhow::Result;
//! use ct2rs::config::{Config, Device};
//! use ct2rs::{Generator, GenerationOptions};
//!
//! # fn main() -> Result<()> {
//! let g = Generator::new("/path/to/model", Config::default())?;
//! let res = g.generate_batch(
//!     vec!["prompt"],
//!     &GenerationOptions::default(),
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

use anyhow::{anyhow, bail, Result};
use tokenizers::{Decoder, EncodeInput, Tokenizer};

use crate::config::Config;
pub use crate::generator::GenerationOptions;
pub use crate::translator::TranslationOptions;

pub mod config;
pub mod generator;
pub mod translator;
mod types;

const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// A text translator with a tokenizer.
pub struct Translator {
    translator: translator::Translator,
    tokenizer: Tokenizer,
}

impl Translator {
    /// Initializes the translator and tokenizer.
    pub fn new<T: AsRef<Path>>(path: T, config: Config) -> Result<Translator> {
        Translator::with_tokenizer(
            &path,
            config,
            Tokenizer::from_file(path.as_ref().join(TOKENIZER_FILENAME))
                .map_err(|err| anyhow!("failed to load a tokenizer: {err}"))?,
        )
    }

    /// Initializes the translator and tokenizer.
    pub fn with_tokenizer<T: AsRef<Path>>(
        path: T,
        config: Config,
        tokenizer: Tokenizer,
    ) -> Result<Translator> {
        Ok(Translator {
            translator: translator::Translator::new(path, config)?,
            tokenizer,
        })
    }

    /// Translates a batch of strings.
    pub fn translate_batch<'a, T, U, V>(
        &self,
        sources: Vec<T>,
        options: &TranslationOptions<V>,
    ) -> Result<Vec<(String, Option<f32>)>>
    where
        T: Into<EncodeInput<'a>>,
        V: AsRef<str>,
    {
        let output = self
            .translator
            .translate_batch(&self.encode(sources)?, options)?;

        let decoder = self.tokenizer.get_decoder().unwrap();
        let mut res = Vec::new();
        for r in output.into_iter() {
            let score = r.score();
            match r.hypotheses.into_iter().next() {
                None => bail!("no results are returned"),
                Some(h) => {
                    res.push((
                        decoder
                            .decode(h.into_iter().collect())
                            .map_err(|err| anyhow!("failed to decode: {err}"))?,
                        score,
                    ));
                }
            }
        }
        Ok(res)
    }

    /// Translates a batch of strings using target prefixes.
    pub fn translate_batch_with_target_prefix<'a, T, U, V>(
        &self,
        sources: Vec<T>,
        target_prefixes: Vec<Vec<U>>,
        options: &TranslationOptions<V>,
    ) -> Result<Vec<(String, Option<f32>)>>
    where
        T: Into<EncodeInput<'a>>,
        U: AsRef<str>,
        V: AsRef<str>,
    {
        let tokens = self.encode(sources)?;
        let output = self.translator.translate_batch_with_target_prefix(
            &tokens,
            &target_prefixes,
            options,
        )?;

        let decoder = self.tokenizer.get_decoder().unwrap();
        let mut res = Vec::new();
        for (r, prefix) in output.into_iter().zip(target_prefixes) {
            let score = r.score();
            match r.hypotheses.into_iter().next() {
                None => bail!("no results are returned"),
                Some(h) => {
                    res.push((
                        decoder
                            .decode(h.into_iter().skip(prefix.len()).collect())
                            .map_err(|err| anyhow!("failed to decode: {err}"))?,
                        score,
                    ));
                }
            }
        }
        Ok(res)
    }

    fn encode<'a, T: Into<EncodeInput<'a>>>(&self, sources: Vec<T>) -> Result<Vec<Vec<String>>> {
        sources
            .into_iter()
            .map(|s| {
                self.tokenizer
                    .encode(s, true)
                    .map(|r| r.get_tokens().to_vec())
                    .map_err(|err| anyhow!("failed to encode the given input: {err}"))
            })
            .collect::<Result<Vec<Vec<String>>>>()
    }
}

/// A text generator with a tokenizer.
pub struct Generator {
    generator: generator::Generator,
    tokenizer: Tokenizer,
}

impl Generator {
    /// Initializes the generator and tokenizer.
    pub fn new<T: AsRef<Path>>(path: T, config: Config) -> Result<Generator> {
        Generator::with_tokenizer(
            &path,
            config,
            Tokenizer::from_file(path.as_ref().join(TOKENIZER_FILENAME))
                .map_err(|err| anyhow!("failed to load a tokenizer: {err}"))?,
        )
    }

    /// Initializes the generator with the given tokenizer.
    pub fn with_tokenizer<T: AsRef<Path>>(
        path: T,
        config: Config,
        tokenizer: Tokenizer,
    ) -> Result<Generator> {
        Ok(Generator {
            generator: generator::Generator::new(path, config)?,
            tokenizer,
        })
    }

    /// Generate texts with the given prompts.
    pub fn generate_batch<'a, T, U, V>(
        &self,
        prompts: Vec<T>,
        options: &GenerationOptions<U, V>,
    ) -> Result<Vec<(Vec<String>, Vec<f32>)>>
    where
        T: Into<EncodeInput<'a>>,
        U: AsRef<str>,
        V: AsRef<str>,
    {
        let tokens = prompts
            .into_iter()
            .map(|s| {
                self.tokenizer
                    .encode(s, false)
                    .map(|r| r.get_tokens().to_vec())
                    .map_err(|err| anyhow!("failed to encode the given input: {err}"))
            })
            .collect::<Result<Vec<Vec<String>>>>()?;

        let output = self.generator.generate_batch(&tokens, options)?;

        let decoder = self.tokenizer.get_decoder().unwrap();
        let mut res = Vec::new();
        for r in output.into_iter() {
            let sequence = r
                .sequences
                .into_iter()
                .map(|seq| decoder.decode(seq))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| anyhow!("failed to decode: {err}"))?;
            let scores = r.scores;
            res.push((sequence, scores))
        }
        Ok(res)
    }
}
