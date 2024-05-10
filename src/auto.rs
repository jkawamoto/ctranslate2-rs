// auto.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This module provides a tokenizer that automatically determines the appropriate tokenizer.
//!
//! It is inspired by the [`Auto Classes`](https://huggingface.co/docs/transformers/model_doc/auto)
//! feature provided by
//! [Hugging Face's Transformers](https://huggingface.co/docs/transformers/),
//! which simplifies the selection process by automatically choosing the correct tokenizer for a
//! given model.
//!
//! ## Usage
//! This tokenizer is particularly useful when working with different types of models where
//! it is not feasible to manually specify the tokenizer each time. It is ideal for scenarios
//! where ease of use and flexibility are more critical than the absolute optimal performance.
//!
//! ```no_run
//! # use anyhow::Result;
//! use ct2rs::auto::Tokenizer as AutoTokenizer;
//! use ct2rs::Tokenizer;
//!
//! # fn main() -> Result<()> {
//! let model_path = "path/to/your/model";
//! let auto_tokenizer = AutoTokenizer::new(model_path)?;
//! let tokenized_output = auto_tokenizer.encode("Example text to tokenize.")?;
//! println!("Tokenized output: {:?}", tokenized_output);
//! # Ok(())
//! # }
//! ```
//!
//! ## Note
//! If you are integrating this module into performance-sensitive applications, it is recommended
//! to evaluate the overhead introduced by dynamic dispatch and consider using a direct tokenizer
//! approach where possible.

use std::path::Path;

use anyhow::{bail, Result};

/// A tokenizer that automatically determines the appropriate tokenizer.
pub struct Tokenizer {
    tokenizer: Box<dyn crate::Tokenizer>,
}

impl Tokenizer {
    /// Create a tokenizer instance by specifying the path to a directory containing model files.
    pub fn new<T: AsRef<Path>>(path: T) -> Result<Self> {
        Ok(Self {
            tokenizer: if let Ok(t) = crate::tokenizers::Tokenizer::new(&path) {
                Box::new(t)
            } else if let Ok(t) = crate::sentencepiece::Tokenizer::new(&path) {
                Box::new(t)
            } else if let Ok(t) = crate::bpe::Tokenizer::new(&path, None) {
                Box::new(t)
            } else {
                bail!("failed to create a tokenizer")
            },
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
    /// A `Result` containing either the vector of tokens if successful or an error if the
    /// tokenization fails.
    fn encode(&self, input: &str) -> Result<Vec<String>> {
        self.tokenizer.encode(input)
    }

    /// Decodes a given sequence of tokens back into a single string.
    ///
    /// This function takes a vector of token strings and reconstructs the original string.
    ///
    /// # Arguments
    /// * `tokens` - A vector of strings representing the tokens to be decoded.
    ///
    /// # Returns
    /// A `Result` containing either the reconstructed string if successful or an error if the
    /// decoding fails.
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        self.tokenizer.decode(tokens)
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
    fn decode_ids(&self, ids: &[u32]) -> Result<String>{
        self.tokenizer.decode_ids(ids)
    }
}
