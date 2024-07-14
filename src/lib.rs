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
//!   CTranslate2, specifically [`translator::Translator`] and [`generator::Generator`].
//! * More user-friendly versions of these, [`Translator`] and [`Generator`],
//!   which incorporate tokenizers for easier handling.
//!
//! # Tokenizers
//! Both [`translator::Translator`] and [`generator::Generator`] work with sequences of tokens.
//! To handle human-readable strings, a tokenizer is necessary.
//! The [`Translator`] and [`Generator`] utilize Hugging Face and SentencePiece tokenizers
//! to convert between strings and token sequences.
//! The [`auto::Tokenizer`] automatically determines which tokenizer to use and constructs it
//! appropriately.
//!
//! ## Example:
//! ### [auto::Tokenizer]
//! Here is an example of using [`auto::Tokenizer`] to build a Translator and translate a string:
//!
//! ```no_run
//! # use anyhow::Result;
//! #
//! use ct2rs::config::Config;
//! use ct2rs::Translator;
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
//! ### [tokenizers::Tokenizer]
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
//! ### [sentencepiece::Tokenizer]
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

use std::path::Path;

use anyhow::{anyhow, Result};

use crate::auto::Tokenizer as AutoTokenizer;
pub use crate::config::{set_log_level, set_random_seed};
use crate::config::Config;
pub use crate::generator::GenerationOptions;
pub use crate::translator::TranslationOptions;

pub mod auto;
pub mod bpe;
pub mod config;
pub mod generator;
pub mod sentencepiece;
pub mod storage_view;
pub mod tokenizers;
pub mod translator;
mod types;
pub mod whisper;

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
    /// A `Result` containing either the vector of tokens if successful or an error if the
    /// tokenization fails.
    fn encode(&self, input: &str) -> Result<Vec<String>>;

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
    fn decode(&self, tokens: Vec<String>) -> Result<String>;
}

#[inline]
fn encode_all<T: Tokenizer, U: AsRef<str>>(
    tokenizer: &T,
    sources: &[U],
) -> Result<Vec<Vec<String>>> {
    sources
        .into_iter()
        .map(|s| tokenizer.encode(s.as_ref()))
        .collect()
}

/// The result for a single generation step.
#[derive(Clone, Debug)]
pub struct GenerationStepResult {
    /// The decoding step.
    pub step: usize,
    /// The batch index.
    pub batch_id: usize,
    /// Index of the hypothesis in the batch.
    pub hypothesis_id: usize,
    /// The generated text.
    pub text: String,
    /// true if return_log_prob was enabled
    pub has_log_prob: bool,
    /// Log probability of the token.
    pub log_prob: f32,
    /// Whether this step is the last step for this batch.
    pub is_last: bool,
}

impl GenerationStepResult {
    fn from_ffi<T: Tokenizer>(r: types::ffi::GenerationStepResult, tokenizer: &T) -> Result<Self> {
        let text = tokenizer.decode(vec![r.token])?;
        Ok(Self {
            step: r.step,
            batch_id: r.batch_id,
            hypothesis_id: r.hypothesis_id,
            text,
            has_log_prob: r.has_log_prob,
            log_prob: r.log_prob,
            is_last: r.is_last,
        })
    }
}

/// A text translator with a tokenizer.
///
/// # Example
/// The following example translates two strings using default settings and outputs each to
/// the standard output.
///
/// ```no_run
/// # use anyhow::Result;
/// #
/// use ct2rs::config::Config;
/// use ct2rs::{Translator, TranslationOptions, GenerationStepResult};
///
/// # fn main() -> Result<()> {
/// let sources = vec![
///     "Hallo World!",
///     "This crate provides Rust bindings for CTranslate2."
/// ];
/// let translator = Translator::new("/path/to/model", &Default::default())?;
/// let results = translator.translate_batch(&sources, &Default::default(), None)?;
/// for (r, _) in results{
///     println!("{}", r);
/// }
/// # Ok(())
/// # }
///```
///
/// The following example translates a single string and uses a callback closure for streaming
/// the output to standard output.
///
///```no_run
/// use std::io::{stdout, Write};
/// use anyhow::Result;
///
/// use ct2rs::config::Config;
/// use ct2rs::{Translator, TranslationOptions, GenerationStepResult};
///
/// # fn main() -> Result<()> {
/// let sources = vec![
///     "Hallo World! This crate provides Rust bindings for CTranslate2."
/// ];
/// let options = TranslationOptions {
///     // beam_size must be 1 to use the stream API.
///     beam_size: 1,
///     ..Default::default()
/// };
/// let mut callback = |step_result: GenerationStepResult| -> Result<()> {
///     print!("{:?}", step_result.text);
///     stdout().flush()?;
///     Ok(())
/// };
/// let translator = Translator::new("/path/to/model", &Config::default())?;
/// let results = translator.translate_batch(&sources, &options, Some(&mut callback))?;
/// # Ok(())
/// # }
/// ```
pub struct Translator<T: Tokenizer> {
    translator: translator::Translator,
    tokenizer: T,
}

impl Translator<AutoTokenizer> {
    /// Initializes the translator with [`auto::Tokenizer`].
    ///
    /// # Arguments
    /// * `path` - A path to the directory containing the language model to be loaded.
    /// * `config` - A reference to a `Config` structure that specifies various settings
    ///   and configurations for the `Translator`.
    ///
    /// # Returns
    /// Returns a `Result` that, if successful, contains the initialized `Translator`. If an error
    /// occurs during initialization, the function will return an error wrapped in the `Result`.
    pub fn new<U: AsRef<Path>>(path: U, config: &Config) -> Result<Self> {
        Self::with_tokenizer(&path, AutoTokenizer::new(&path)?, config)
    }
}

impl<T: Tokenizer> Translator<T> {
    /// Initializes the translator with the given tokenizer.
    ///
    /// # Arguments
    /// * `path` - The path to the directory containing the language model.
    /// * `tokenizer` - An instance of a tokenizer.
    /// * `config` - A reference to a `Config` structure specifying the settings for the
    ///   `Translator`.
    ///
    /// # Returns
    /// Returns a `Result` containing the initialized `Translator`, or an error if initialization
    /// fails.
    ///
    /// # Example
    /// This example demonstrates creating a `Translator` instance with a Sentencepiece tokenizer.
    ///
    /// ```no_run
    /// # use anyhow::Result;
    /// use ct2rs::{TranslationOptions, Translator};
    /// use ct2rs::config::Config;
    /// use ct2rs::sentencepiece::Tokenizer;
    ///
    /// # fn main() -> Result<()> {
    /// let translator = Translator::with_tokenizer(
    ///     "/path/to/model",
    ///     Tokenizer::from_file("/path/to/source.spm", "/path/to/target.spm")?,
    ///     &Config::default()
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    pub fn with_tokenizer<U: AsRef<Path>>(path: U, tokenizer: T, config: &Config) -> Result<Self> {
        Ok(Translator {
            translator: translator::Translator::new(path, config)?,
            tokenizer,
        })
    }

    /// Translates multiple lists of strings in a batch processing manner.
    ///
    /// This function takes a vector of strings and performs batch translation according to the
    /// specified settings in `options`. The results of the batch translation are returned as a
    /// vector. An optional `callback` closure can be provided which is invoked for each new token
    /// generated during the translation process. This allows for step-by-step reception of the
    /// batch translation results. If the callback returns `Err`, it will stop the translation for
    /// that batch. Note that if a callback is provided, `options.beam_size` must be set to `1`.
    ///
    /// # Arguments
    /// * `source` - A vector of strings to be translated.
    /// * `options` - Settings applied to the batch translation process.
    /// * `callback` - An optional mutable reference to a closure that is called for each token
    ///   generation step. The closure takes a `GenerationStepResult` and returns a
    ///   `anyhow::Result<()>`. If it returns `Err`, the translation process for the current batch
    ///   will stop.
    ///
    /// # Returns
    /// Returns a `Result` containing a vector of `TranslationResult` if successful, or an error if
    /// the translation fails.
    ///
    pub fn translate_batch<'a, U, V, W>(
        &self,
        sources: &[U],
        options: &TranslationOptions<V, W>,
        callback: Option<&'a mut dyn FnMut(GenerationStepResult) -> Result<()>>,
    ) -> Result<Vec<(String, Option<f32>)>>
    where
        U: AsRef<str>,
        V: AsRef<str>,
        W: AsRef<str>,
    {
        let output = if let Some(callback) = callback {
            let mut callback_result = Ok(());
            let mut wrapped_callback = |r: types::ffi::GenerationStepResult| -> bool {
                if let Err(e) =
                    GenerationStepResult::from_ffi(r, &self.tokenizer).and_then(|r| callback(r))
                {
                    callback_result = Err(e);
                    return true;
                }
                false
            };
            let output = self.translator.translate_batch(
                &encode_all(&self.tokenizer, sources)?,
                options,
                Some(&mut wrapped_callback),
            )?;
            callback_result?;
            output
        } else {
            self.translator.translate_batch(
                &encode_all(&self.tokenizer, sources)?,
                options,
                None,
            )?
        };

        let mut res = Vec::new();
        for r in output.into_iter() {
            let score = r.score();
            let hypotheses = r
                .hypotheses
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no results are returned"))?;
            res.push((
                self.tokenizer
                    .decode(hypotheses)
                    .map_err(|err| anyhow!("failed to decode: {err}"))?,
                score,
            ));
        }
        Ok(res)
    }

    /// Translates multiple lists of strings with target prefixes in a batch processing manner.
    ///
    /// This function takes a vector of strings and corresponding target prefixes, performing
    /// batch translation according to the specified settings in `options`. An optional `callback`
    /// closure can be provided which is invoked for each new token generated during the translation
    /// process.
    ///
    /// This function is similar to `translate_batch`, with the addition of handling target prefixes
    /// that guide the translation process. For more detailed parameter and option descriptions,
    /// refer to the documentation for [`Translator::translate_batch`].
    ///
    /// # Arguments
    /// * `sources` - A vector of strings translated.
    /// * `target_prefix` - A vector of token lists, each list representing a sequence of target
    ///   prefix tokens that provide a starting point for the translation output.
    /// * `options` - Settings applied to the batch translation process.
    /// * `callback` - An optional mutable reference to a closure that is called for each token
    ///   generation step. The closure takes a `GenerationStepResult` and returns a
    ///   `anyhow::Result<()>`. If it returns `Err`, the translation process for the current batch
    ///   will stop.
    ///
    /// # Returns
    /// Returns a `Result` containing a vector of `TranslationResult` if successful, or an error if
    /// the translation fails.
    pub fn translate_batch_with_target_prefix<'a, U, V, W, E>(
        &self,
        sources: &[U],
        target_prefixes: &Vec<Vec<V>>,
        options: &TranslationOptions<W, E>,
        callback: Option<&'a mut dyn FnMut(GenerationStepResult) -> Result<()>>,
    ) -> Result<Vec<(String, Option<f32>)>>
    where
        U: AsRef<str>,
        V: AsRef<str>,
        W: AsRef<str>,
        E: AsRef<str>,
    {
        let output = if let Some(callback) = callback {
            let mut callback_result = Ok(());
            let mut wrapped_callback = |r: types::ffi::GenerationStepResult| -> bool {
                if let Err(e) =
                    GenerationStepResult::from_ffi(r, &self.tokenizer).and_then(|r| callback(r))
                {
                    callback_result = Err(e);
                    return true;
                }
                false
            };
            let output = self.translator.translate_batch_with_target_prefix(
                &encode_all(&self.tokenizer, sources)?,
                &target_prefixes,
                options,
                Some(&mut wrapped_callback),
            )?;
            callback_result?;
            output
        } else {
            self.translator.translate_batch_with_target_prefix(
                &encode_all(&self.tokenizer, sources)?,
                &target_prefixes,
                options,
                None,
            )?
        };

        let mut res = Vec::new();
        for (r, prefix) in output.into_iter().zip(target_prefixes) {
            let score = r.score();
            let mut hypotheses = r
                .hypotheses
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no results are returned"))?;
            hypotheses.drain(0..prefix.len());

            res.push((
                self.tokenizer
                    .decode(hypotheses)
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
///
/// # Example
/// The following example generates text following two prompts in a batch process,
/// with each result output to the standard output.
///
/// ```no_run
/// # use anyhow::Result;
/// use ct2rs::config::{Config, Device};
/// use ct2rs::{Generator, GenerationOptions};
///
/// # fn main() -> Result<()> {
/// let generator = Generator::new("/path/to/model", &Config::default())?;
/// let res = generator.generate_batch(
///     &vec!["Hello, I am"],
///     &GenerationOptions::default(),
///     None
/// )?;
/// for r in res {
///     println!("{:?}", r);
/// }
/// # Ok(())
/// # }
/// ```
///
/// The following example generates text following a single prompt and outputs it to the standard
/// output using a callback closure for stream processing.
///
/// ```no_run
/// use std::io::{stdout, Write};
/// use anyhow::Result;
///
/// use ct2rs::config::{Config, Device};
/// use ct2rs::{Generator, GenerationOptions};
///
/// # fn main() -> Result<()> {
/// use ct2rs::GenerationStepResult;
/// let generator = Generator::new("/path/to/model", &Config::default())?;
/// let _ = generator.generate_batch(
///     &vec!["Hello, I am"],
///     &GenerationOptions{
///         // beam_size must be 1 to use the stream API.
///         beam_size: 1,
///         ..Default::default()
///     },
///     Some(&mut |step_result: GenerationStepResult| -> Result<()> {
///         print!("{:?}", step_result.text);
///         stdout().flush()?;
///         Ok(())
///     })
/// )?;
/// # Ok(())
/// # }
/// ```
pub struct Generator<T: Tokenizer> {
    generator: generator::Generator,
    tokenizer: T,
}

impl Generator<AutoTokenizer> {
    /// Initializes the generator with [`auto::Tokenizer`].
    ///
    /// # Arguments
    /// * `path` - A path to the directory containing the language model to be loaded.
    /// * `config` - A reference to a `Config` structure that specifies various settings
    ///   and configurations for the `Generator`.
    ///
    /// # Returns
    /// Returns a `Result` that, if successful, contains the initialized `Generator`. If an error
    /// occurs during initialization, the function will return an error wrapped in the `Result`.
    pub fn new<T: AsRef<Path>>(path: T, config: &Config) -> Result<Self> {
        Self::with_tokenizer(&path, AutoTokenizer::new(&path)?, config)
    }
}

impl<T: Tokenizer> Generator<T> {
    /// Initializes the generator with the given tokenizer.
    ///
    /// # Arguments
    /// * `path` - A path to the directory containing the language model to be loaded.
    /// * `tokenizer` - An instance of the tokenizer.
    /// * `config` - A reference to a `Config` structure that specifies various settings
    ///   and configurations for the `Generator`.
    ///
    /// # Returns
    /// Returns a `Result` that, if successful, contains the initialized `Generator`. If an error
    /// occurs during initialization, the function will return an error wrapped in the `Result`.
    ///
    /// # Example
    /// The following example creates a translator instance with the tokenizer provided by
    /// [tokenizers](https://huggingface.co/docs/tokenizers).
    ///
    /// ```no_run
    /// # use anyhow::Result;
    /// use ct2rs::Generator;
    /// use ct2rs::config::Config;
    /// use ct2rs::tokenizers::Tokenizer;
    ///
    /// # fn main() -> Result<()> {
    /// let generator = Generator::with_tokenizer(
    ///     "/path/to/model",
    ///     Tokenizer::from_file("/path/to/tokenizer.json")?,
    ///     &Config::default()
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    pub fn with_tokenizer<U: AsRef<Path>>(path: U, tokenizer: T, config: &Config) -> Result<Self> {
        Ok(Generator {
            generator: generator::Generator::new(path, config)?,
            tokenizer,
        })
    }

    /// Generates texts following the provided batch of start strings.
    ///
    /// This function generates texts sequentially starting from the given `prompts`.
    /// The generation continues according to the
    /// options specified in `options`.
    ///
    /// An optional `callback` closure can be provided which is invoked for each new token
    /// generated during the translation process. This allows for step-by-step reception of the
    /// batch translation results. If the callback returns `Err`, it will stop the translation for
    /// that batch. Note that if a callback is provided, `options.beam_size` must be set to `1`.
    ///
    /// # Arguments
    /// * `prompts` - A vector of prompts. These prompts represent the initial state of the
    ///   generation process.
    /// * `options` - Settings applied to the generation process, such as beam size and other
    ///   generation-specific configurations.
    /// * `callback` - An optional mutable reference to a closure that is called for each token
    ///   generation step. The closure takes a `GenerationStepResult` and returns a
    ///   `anyhow::Result<()>`. If it returns `Err`, the translation process for the current batch
    ///   will stop.
    ///
    /// # Returns
    /// Returns a `Result` containing a vector of `GenerationResult` if successful, encapsulating
    /// the generated sequences for each input start token batch, or an error if the generation
    /// fails.
    pub fn generate_batch<'a, U, V, W, E>(
        &self,
        prompts: &[U],
        options: &GenerationOptions<V, E, W>,
        callback: Option<&'a mut dyn FnMut(GenerationStepResult) -> Result<()>>,
    ) -> Result<Vec<(Vec<String>, Vec<f32>)>>
    where
        U: AsRef<str>,
        V: AsRef<str>,
        W: AsRef<str>,
        E: AsRef<str>,
    {
        let output = if let Some(callback) = callback {
            let mut callback_result = Ok(());
            let mut wrapped_callback = |r: types::ffi::GenerationStepResult| -> bool {
                if let Err(e) =
                    GenerationStepResult::from_ffi(r, &self.tokenizer).and_then(|r| callback(r))
                {
                    callback_result = Err(e);
                    return true;
                }
                false
            };
            let output = self.generator.generate_batch(
                &encode_all(&self.tokenizer, prompts)?,
                options,
                Some(&mut wrapped_callback),
            )?;
            callback_result?;
            output
        } else {
            self.generator
                .generate_batch(&encode_all(&self.tokenizer, prompts)?, options, None)?
        };

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
