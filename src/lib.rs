// lib.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This crate provides Rust bindings for [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2).
//!
//! Please refer to the crate [ctranslate2-sample](https://github.com/jkawamoto/ctranslate2-rs/tree/main/sample) for the sample code.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use tokenizers::{Decoder, EncodeInput};

use crate::config::{Config, Device};
pub use crate::translator::TranslationOptions;

pub mod config;
pub mod translator;

const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// A text translator with a tokenizer.
pub struct Translator {
    translator: translator::Translator,
    tokenizer: tokenizers::Tokenizer,
}

impl Translator {
    /// Initializes the translator and tokenizer.
    pub fn new<T: AsRef<Path>>(path: T, device: Device, config: Config) -> Result<Translator> {
        Ok(Translator {
            translator: translator::Translator::new(
                path.as_ref().to_str().unwrap(),
                device,
                config,
            )?,
            tokenizer: tokenizers::Tokenizer::from_file(path.as_ref().join(TOKENIZER_FILENAME))
                .map_err(|err| anyhow!("failed to load a tokenizer: {err}"))?,
        })
    }

    /// Translates a batch of strings.
    pub fn translate_batch<'a, T, U, V>(
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
        let tokens = sources
            .into_iter()
            .map(|s| {
                self.tokenizer
                    .encode(s, true)
                    .map(|r| r.get_tokens().to_vec())
                    .map_err(|err| anyhow!("failed to encode the given input: {err}"))
            })
            .collect::<Result<Vec<Vec<String>>>>()?;

        let output = self
            .translator
            .translate_batch(&tokens, &target_prefixes, options)?;

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
}
