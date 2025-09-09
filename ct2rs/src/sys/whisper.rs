// whisper.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This module provides a Rust binding to the
//! [`ctranslate2::models::Whisper`](https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html).

use std::ffi::OsString;
use std::fmt::{Debug, Formatter};
use std::path::Path;

use anyhow::{anyhow, Result};
use cxx::UniquePtr;

use super::{
    config, storage_view, vec_ffi_vecstr, Config, StorageView, VecStr, VecString, VecUSize,
};

use self::ffi::VecDetectionResult;
pub use self::ffi::{DetectionResult, WhisperOptions};

#[cxx::bridge]
mod ffi {
    /// Options for whisper generation.
    ///
    /// # Examples
    ///
    /// Example of creating a default `WhisperOptions`:
    ///
    /// ```
    /// use ct2rs::sys::WhisperOptions;
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
        /// Include log probs of each token in the result. (default: false)
        pub return_logits_vocab: bool,
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
        include!("ct2rs/include/whisper.h");
        include!("ct2rs/src/sys/types.rs.h");

        type VecStr<'a> = super::VecStr<'a>;
        type VecString = super::VecString;
        type VecUSize = super::VecUSize;

        type Config = super::config::ffi::Config;

        type StorageView = super::storage_view::ffi::StorageView;

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
            return_logits_vocab: false,
            return_no_speech_prob: false,
            max_initial_timestamp_index: 50,
            suppress_blank: true,
            suppress_tokens: vec![-1],
        }
    }
}

/// A generation result from the Whisper model.
///
/// This struct is a Rust binding to the
/// [`ctranslate2.models.WhisperGenerationResult`](https://opennmt.net/CTranslate2/python/ctranslate2.models.WhisperGenerationResult.html).
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

impl From<VecDetectionResult> for Vec<DetectionResult> {
    fn from(value: VecDetectionResult) -> Self {
        value.v
    }
}

/// Implements the Whisper speech recognition model published by OpenAI.
///
/// This struct is a Rust binding to the
/// [`ctranslate2::models::Whisper`](https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html).
///
/// # Example
/// ```no_run
/// # use anyhow::Result;
/// #
/// use ct2rs::sys::{Config, StorageView, Whisper};
///
/// # fn main() -> Result<()>{
/// let whisper = Whisper::new("/path/to/model", Config::default())?;
///
/// let batch_size = 1;
/// let n_mels = whisper.n_mels();
/// let chunk_length = 3000;
///
/// // Calculate Mel spectrogram of the source audio and store it in `mel_spectrogram`.
/// // The length of the vector should be `batch_size` x `n_mels` x `chunk_length`.
/// let mut mel_spectrogram = vec![];
///
/// let storage_view = StorageView::new(
///     &[batch_size, n_mels, chunk_length],
///     &mut mel_spectrogram,
///     Default::default()
/// )?;
///
/// // Detect language.
/// let lang = whisper.detect_language(&storage_view)?;
///
/// // Transcribe.
/// let res = whisper.generate(
///     &storage_view,
///     &[vec![
///         "<|startoftranscript|>",
///         &lang[0][0].language,
///         "<|transcribe|>",
///         "<|notimestamps|>",
///     ]],
///     &Default::default(),
/// )?;
/// # Ok(())
/// # }
/// ```
pub struct Whisper {
    model: OsString,
    ptr: UniquePtr<ffi::Whisper>,
}

impl Whisper {
    /// Creates and initializes an instance of `Whisper`.
    ///
    /// This function constructs a new `Whisper` by loading a language model from the specified
    /// `model_path` and applying the provided `config` settings.
    ///
    /// # Arguments
    /// * `model_path` - A path to the directory containing the language model to be loaded.
    /// * `config` - A reference to a [`Config`] structure that specifies various settings
    ///   and configurations for the `Whisper`.
    ///
    /// # Returns
    /// Returns a `Result` that, if successful, contains the initialized `Whisper`. If an error
    /// occurs during initialization, the function will return an error wrapped in the `Result`.
    ///
    /// # Example
    /// ```no_run
    /// # use anyhow::Result;
    /// #
    /// use ct2rs::sys::{Config, Whisper};
    ///
    /// # fn main() -> Result<()> {
    /// let whisper = Whisper::new("/path/to/model", Config::default())?;
    /// # Ok(())
    /// # }
    /// ```
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
    ///
    /// # Arguments
    /// * `features` – A [`StorageView`] consisting of Mel spectrogram of the audio,
    ///   as a float array with shape `[batch_size, n_mels, chunk_length]`.
    ///   [`n_mels`][Whisper::n_mels] method gives the expected `n_mels` in the shape.
    /// * `prompts` – Batch of initial string tokens.
    /// * `options` - Settings.
    ///
    /// # Returns
    /// Returns a `Result` containing a vector of [`WhisperGenerationResult`] if successful,
    /// or an error if the translation fails.
    pub fn generate<T: AsRef<str>>(
        &self,
        features: &StorageView,
        prompts: &[Vec<T>],
        options: &WhisperOptions,
    ) -> Result<Vec<WhisperGenerationResult>> {
        self.ptr
            .generate(features, &vec_ffi_vecstr(prompts), options)
            .map(|res| res.into_iter().map(WhisperGenerationResult::from).collect())
            .map_err(|e| anyhow!("failed to generate: {e}"))
    }

    /// Returns the probability of each language.
    ///
    /// # Arguments
    /// * `features` – [`StorageView`] consisting of Mel spectrogram of the audio, as a float array
    ///   with shape `[batch_size, n_mels, chunk_length]`.
    ///
    /// # Returns
    /// For each batch, a list of [`DetectionResult`] ordered from best to worst probability.
    /// This result is wrapped by `Result`.
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

// Releasing `UniquePtr<Whisper>` invokes joining threads.
// However, on Windows, this causes a deadlock.
// As a workaround, it is bypassed here.
// See also https://github.com/jkawamoto/ctranslate2-rs/issues/64
#[cfg(target_os = "windows")]
impl Drop for Whisper {
    fn drop(&mut self) {
        let ptr = std::mem::replace(&mut self.ptr, UniquePtr::null());
        unsafe {
            std::ptr::drop_in_place(ptr.into_raw());
        }
    }
}

unsafe impl Send for ffi::Whisper {}
unsafe impl Sync for ffi::Whisper {}

#[cfg(test)]
mod tests {
    use super::{ffi, WhisperGenerationResult, WhisperOptions};

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
        assert!(!opts.return_logits_vocab);
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

    #[cfg(feature = "hub")]
    mod hub {
        use crate::download_model;
        use crate::sys::Whisper;

        const MODEL_ID: &str = "jkawamoto/whisper-tiny-ct2";

        #[test]
        #[ignore]
        fn test_whisper_debug() {
            let model_path = download_model(MODEL_ID).unwrap();

            let whisper = Whisper::new(&model_path, Default::default()).unwrap();
            assert!(format!("{:?}", whisper)
                .contains(model_path.file_name().unwrap().to_str().unwrap()));
        }
    }
}
