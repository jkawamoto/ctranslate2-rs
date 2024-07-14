// whisper.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This module provides a Rust binding to the
//! [`ctranslate2::models::Whisper`](https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html).
//!
//! The main structure provided by this module is the [`Whisper`] structure.
//!
//! In addition to the `Whisper`, this module also offers various supportive structures such
//! as [`WhisperOptions`], [`DetectionResult`], and [`WhisperGenerationResult`].
//!
//! For more detailed information on each structure and its usage, please refer to their respective
//! documentation within this module.

use std::ffi::OsString;
use std::fmt::{Debug, Formatter};
use std::path::Path;

use anyhow::{anyhow, Result};
use cxx::UniquePtr;

use crate::config::Config;
use crate::storage_view::StorageView;
use crate::types::vec_ffi_vecstr;
pub use crate::whisper::ffi::{DetectionResult, WhisperOptions};
use crate::whisper::ffi::VecDetectionResult;

#[cxx::bridge]
mod ffi {
    /// Options for whisper generation.
    /// # Examples
    ///
    /// Example of creating a default `TranslationOptions`:
    ///
    /// ```
    /// use ct2rs::whisper::WhisperOptions;
    ///
    /// let options = WhisperOptions::default();
    /// ```
    #[derive(Clone, Debug)]
    pub struct WhisperOptions {
        /// Beam size to use for beam search (set 1 to run greedy search). (default: 5)
        pub beam_size: usize,
        /// Beam search patience factor, as described in <https://arxiv.org/abs/2204.05424>.
        /// The decoding will continue until beam_size*patience hypotheses are finished.
        /// (default: 1.0)
        pub patience: f32,
        /// Exponential penalty applied to the length during beam search. (default: 1.0)
        pub length_penalty: f32,
        /// Penalty applied to the score of previously generated tokens, as described in
        /// <https://arxiv.org/abs/1909.05858> (set > 1 to penalize). (default: 1.0)
        pub repetition_penalty: f32,
        /// Prevent repetitions of ngrams with this size (set 0 to disable). (default: 0)
        pub no_repeat_ngram_size: usize,
        /// Maximum generation length. (default: 448)
        pub max_length: usize,
        /// Randomly sample from the top K candidates (set 0 to sample from the full distribution).
        /// (default: 1)
        pub sampling_topk: usize,
        /// High temperatures increase randomness. (default: 1.0)
        pub sampling_temperature: f32,
        /// Number of hypotheses to include in the result. (default: 1)
        pub num_hypotheses: usize,
        /// Include scores in the result. (default: false)
        pub return_scores: bool,
        /// Include the probability of the no speech token in the result. (default: false)
        pub return_no_speech_prob: bool,
        /// Maximum index of the first predicted timestamp. (default: 50)
        pub max_initial_timestamp_index: usize,
        /// Suppress blank outputs at the beginning of the sampling. (default: true)
        pub suppress_blank: bool,
        /// List of token IDs to suppress.
        /// -1 will suppress a default set of symbols as defined in the model config.json file.
        /// (default: `[-1]`)
        pub suppress_tokens: Vec<i32>,
    }

    struct WhisperGenerationResult {
        sequences: Vec<VecString>,
        sequences_ids: Vec<VecUSize>,
        scores: Vec<f32>,
        no_speech_prob: f32,
    }

    /// Pair of the detected language and its probability.
    #[derive(PartialEq, Clone, Debug)]
    pub struct DetectionResult {
        /// Token of the language.
        language: String,
        /// Probability of the language.
        probability: f32,
    }

    #[derive(PartialEq, Clone)]
    struct VecDetectionResult {
        v: Vec<DetectionResult>,
    }

    unsafe extern "C++" {
        include!("ct2rs/src/types.rs.h");
        include!("ct2rs/include/whisper.h");

        type VecStr<'a> = crate::types::ffi::VecStr<'a>;
        type VecString = crate::types::ffi::VecString;
        type VecUSize = crate::types::ffi::VecUSize;

        type Config = crate::config::ffi::Config;

        type StorageView = crate::storage_view::ffi::StorageView;

        type Whisper;

        fn whisper(model_path: &str, config: UniquePtr<Config>) -> Result<UniquePtr<Whisper>>;

        fn generate(
            self: &Whisper,
            features: &StorageView,
            prompts: &[VecStr],
            options: &WhisperOptions,
        ) -> Result<Vec<WhisperGenerationResult>>;

        fn detect_language(
            self: &Whisper,
            features: &StorageView,
        ) -> Result<Vec<VecDetectionResult>>;

        fn is_multilingual(self: &Whisper) -> bool;

        fn n_mels(self: &Whisper) -> usize;

        fn num_languages(self: &Whisper) -> usize;

        fn num_queued_batches(self: &Whisper) -> usize;

        fn num_active_batches(self: &Whisper) -> usize;

        fn num_replicas(self: &Whisper) -> usize;
    }
}

impl Default for WhisperOptions {
    fn default() -> Self {
        Self {
            beam_size: 5,
            patience: 1.,
            length_penalty: 1.,
            repetition_penalty: 1.,
            no_repeat_ngram_size: 0,
            max_length: 448,
            sampling_topk: 1,
            sampling_temperature: 1.,
            num_hypotheses: 1,
            return_scores: false,
            return_no_speech_prob: false,
            max_initial_timestamp_index: 50,
            suppress_blank: true,
            suppress_tokens: vec![-1],
        }
    }
}

/// A generation result from the Whisper model.
#[derive(Clone, Debug)]
pub struct WhisperGenerationResult {
    /// Generated sequences of tokens.
    pub sequences: Vec<Vec<String>>,
    /// Generated sequences of token IDs.
    pub sequences_ids: Vec<Vec<usize>>,
    /// Score of each sequence (empty if `return_scores` was disabled).
    pub scores: Vec<f32>,
    /// Probability of the no speech token (0 if `return_no_speech_prob` was disabled).
    pub no_speech_prob: f32,
}

impl WhisperGenerationResult {
    /// Returns the number of sequences.
    #[inline]
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Returns true if this result includes scores.
    #[inline]
    pub fn has_scores(&self) -> bool {
        !self.scores.is_empty()
    }
}

impl From<ffi::WhisperGenerationResult> for WhisperGenerationResult {
    fn from(r: ffi::WhisperGenerationResult) -> Self {
        Self {
            sequences: r.sequences.into_iter().map(Vec::<String>::from).collect(),
            sequences_ids: r
                .sequences_ids
                .into_iter()
                .map(Vec::<usize>::from)
                .collect(),
            scores: r.scores,
            no_speech_prob: r.no_speech_prob,
        }
    }
}

impl Into<Vec<DetectionResult>> for VecDetectionResult {
    fn into(self) -> Vec<DetectionResult> {
        self.v
    }
}

/// A Rust binding to the
/// [`ctranslate2::models::Whisper`](https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html).
pub struct Whisper {
    model: OsString,
    ptr: UniquePtr<ffi::Whisper>,
}

impl Whisper {
    /// Initializes a Whisper model from a converted model.
    pub fn new<T: AsRef<Path>>(model_path: T, config: Config) -> Result<Self> {
        let model_path = model_path.as_ref();
        Ok(Self {
            model: model_path
                .file_name()
                .map(|s| s.to_os_string())
                .unwrap_or_default(),
            ptr: ffi::whisper(
                model_path
                    .to_str()
                    .ok_or_else(|| anyhow!("invalid path: {}", model_path.display()))?,
                config.to_ffi(),
            )?,
        })
    }

    /// Encodes the input features and generates from the given prompt.
    pub fn generate<T: AsRef<str>>(
        &self,
        features: &StorageView,
        prompts: &[Vec<T>],
        opts: &WhisperOptions,
    ) -> Result<Vec<WhisperGenerationResult>> {
        self.ptr
            .generate(features, &vec_ffi_vecstr(prompts), opts)
            .map(|res| res.into_iter().map(WhisperGenerationResult::from).collect())
            .map_err(|e| anyhow!("failed to generate: {e}"))
    }

    /// Returns the probability of each language.
    pub fn detect_language(&self, features: &StorageView) -> Result<Vec<Vec<DetectionResult>>> {
        self.ptr
            .detect_language(features)
            .map(|res| res.into_iter().map(VecDetectionResult::into).collect())
            .map_err(|e| anyhow!("failed to detect language: {e}"))
    }

    /// Returns `true` if this model is multilingual.
    #[inline]
    pub fn is_multilingual(&self) -> bool {
        self.ptr.is_multilingual()
    }

    /// Returns dimension of mel input features.
    #[inline]
    pub fn n_mels(&self) -> usize {
        self.ptr.n_mels()
    }

    /// Returns the number of languages supported.
    #[inline]
    pub fn num_languages(&self) -> usize {
        self.ptr.num_languages()
    }

    /// Number of batches in the work queue.
    #[inline]
    pub fn num_queued_batches(&self) -> usize {
        self.ptr.num_queued_batches()
    }

    /// Number of batches in the work queue or currently processed by a worker.
    #[inline]
    pub fn num_active_batches(&self) -> usize {
        self.ptr.num_active_batches()
    }

    /// Number of parallel replicas.
    #[inline]
    pub fn num_replicas(&self) -> usize {
        self.ptr.num_replicas()
    }
}

impl Debug for Whisper {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Whisper")
            .field("model", &self.model)
            .field("multilingual", &self.is_multilingual())
            .field("mels", &self.n_mels())
            .field("languages", &self.num_languages())
            .field("queued_batches", &self.num_queued_batches())
            .field("active_batches", &self.num_active_batches())
            .field("replicas", &self.num_replicas())
            .finish()
    }
}

unsafe impl Send for Whisper {}
unsafe impl Sync for Whisper {}

#[cfg(test)]
mod tests {
    use crate::whisper::{ffi, WhisperGenerationResult, WhisperOptions};

    #[test]
    fn test_default_options() {
        let opts = WhisperOptions::default();

        assert_eq!(opts.beam_size, 5);
        assert_eq!(opts.patience, 1.);
        assert_eq!(opts.length_penalty, 1.);
        assert_eq!(opts.repetition_penalty, 1.);
        assert_eq!(opts.no_repeat_ngram_size, 0);
        assert_eq!(opts.max_length, 448);
        assert_eq!(opts.sampling_topk, 1);
        assert_eq!(opts.sampling_temperature, 1.);
        assert_eq!(opts.num_hypotheses, 1);
        assert!(!opts.return_scores);
        assert!(!opts.return_no_speech_prob);
        assert_eq!(opts.max_initial_timestamp_index, 50);
        assert!(opts.suppress_blank);
        assert_eq!(opts.suppress_tokens, vec![-1]);
    }

    #[test]
    fn test_generation_result() {
        let sequences = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
        ];
        let sequences_ids = vec![vec![1, 2], vec![5, 6, 7]];
        let scores = vec![9., 8., 7.];
        let no_speech_prob = 10.;

        let res: WhisperGenerationResult = ffi::WhisperGenerationResult {
            sequences: sequences
                .iter()
                .map(|v| ffi::VecString::from(v.clone()))
                .collect(),
            sequences_ids: sequences_ids
                .iter()
                .map(|v| ffi::VecUSize::from(v.clone()))
                .collect(),
            scores: scores.clone(),
            no_speech_prob,
        }
        .into();

        assert_eq!(res.sequences, sequences);
        assert_eq!(res.sequences_ids, sequences_ids);
        assert_eq!(res.scores, scores);
        assert_eq!(res.no_speech_prob, no_speech_prob);
        assert_eq!(res.num_sequences(), sequences.len());
        assert!(res.has_scores());
    }

    #[test]
    fn test_empty_result() {
        let res: WhisperGenerationResult = ffi::WhisperGenerationResult {
            sequences: vec![],
            sequences_ids: vec![],
            scores: vec![],
            no_speech_prob: 0.,
        }
        .into();

        assert!(res.sequences.is_empty());
        assert!(res.sequences_ids.is_empty());
        assert!(res.scores.is_empty());
        assert_eq!(res.no_speech_prob, 0.);
        assert_eq!(res.num_sequences(), 0);
        assert!(!res.has_scores());
    }
}
