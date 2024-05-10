// bpe.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This module provides a tokenizer based on the Byte Pair Encoding (BPE) model.
//!
//! Byte Pair Encoding is a subword tokenization technique that can dynamically adjust
//! the vocabulary based on the provided corpus, often leading to more efficient
//! representation of text data in machine learning models. For a deeper understanding
//! of BPE, refer to the [original paper](https://www.aclweb.org/anthology/P16-1162/).
//!
//! This module allows for the creation of a [`Tokenizer`] structure instance by specifying a path
//! to a directory containing `vocab.json` and `merges.txt`, which are essential for the BPE
//! algorithm.
//! You can also specify optional suffixes for the decoder to fine-tune the behavior of the decoding
//! process. For more details on how to use suffixes with the BPE decoder, please see the
//! documentation for
//! [BPEDecoder](https://docs.rs/tokenizers/latest/tokenizers/decoders/bpe/struct.BPEDecoder.html).
//!
//! ## Usage
//! The tokenizer instances created can be utilized in conjunction with structures like
//! `crate::Translator` and `crate::Generator` for tasks such as translation or text generation.
//!

use std::path::Path;

use anyhow::{anyhow, Result};
use tokenizers::{Decoder, Model};
use tokenizers::decoders::bpe::BPEDecoder;
use tokenizers::models::bpe::BPE;

const VOCAB_FILE: &str = "vocab.json";
const MERGES_FILE: &str = "merges.txt";

pub struct Tokenizer {
    encoder: BPE,
    decoder: BPEDecoder,
}

impl Tokenizer {
    /// Create a tokenizer instance by specifying the path to a directory containing `vocab.json`
    /// and `mergers.txt`.
    pub fn new<T: AsRef<Path>>(path: T, decoder_suffix: Option<String>) -> Result<Self> {
        Self::from_file(
            path.as_ref().join(VOCAB_FILE),
            path.as_ref().join(MERGES_FILE),
            decoder_suffix,
        )
    }

    /// Create a tokenizer instance by specifying the path to `vocab.json` and `mergers.txt`.
    pub fn from_file<T: AsRef<Path>, U: AsRef<Path>>(
        vocab: T,
        merges: U,
        decoder_suffix: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            encoder: BPE::from_file(
                vocab
                    .as_ref()
                    .to_str()
                    .ok_or_else(|| anyhow!("invalid path: {}", vocab.as_ref().display()))?,
                merges
                    .as_ref()
                    .to_str()
                    .ok_or_else(|| anyhow!("invalid path: {}", merges.as_ref().display()))?,
            )
            .build()
            .map_err(|e| anyhow!("failed to build an encoder: {e}"))?,
            decoder: decoder_suffix
                .map(BPEDecoder::new)
                .unwrap_or_else(BPEDecoder::default),
        })
    }
}

impl crate::Tokenizer for Tokenizer {
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
    fn encode(&self, input: &str) -> Result<Vec<String>> {
        let tokens = self
            .encoder
            .tokenize(input)
            .map_err(|e| anyhow!("failed to tokenize input: {e}"))?;
        let mut res = tokens
            .into_iter()
            .map(|token| token.value)
            .collect::<Vec<String>>();
        res.push("</s>".to_string());
        Ok(res)
    }

    /// Decodes a given sequence of tokens back into a single string.
    ///
    /// This function takes a vector of token strings and reconstructs the original string.
    ///
    /// # Arguments
    /// * `tokens` - A vector of strings representing the tokens to be decoded.
    ///
    /// # Returns
    /// A `Result` containing either the reconstructed string if successful or an error if the decoding fails.
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        self.decoder
            .decode(tokens)
            .map_err(|e| anyhow!("failed to decode tokens: {e}"))
    }

    /// Decodes a given sequence of token ids back into a single string.
    ///
    /// This function takes a vector of token ids and reconstructs the original string.
    ///
    /// # Arguments
    /// * `ids` - A vector of u32 integers representing the tokens to be decoded.
    ///
    /// # Returns
    /// A `Result` containing either the reconstructed string if successful or an error if the
    /// decoding fails.
    fn decode_ids(&self, ids: &[u32]) -> Result<String> {
        self.decode(
            ids.into_iter()
                .map(|id| self.encoder.id_to_token(*id))
                .collect::<Option<Vec<String>>>()
                .ok_or_else(|| anyhow!("unknown token ID"))?,
        )
    }
}
