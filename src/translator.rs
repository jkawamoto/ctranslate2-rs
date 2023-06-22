// translator.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use cxx::UniquePtr;

use crate::config::{BatchType, ComputeType, Config, Device};

#[cxx::bridge]
mod ffi {
    struct VecString {
        v: Vec<String>,
    }

    struct VecStr<'a> {
        v: Vec<&'a str>,
    }

    enum ComputeType {
        Default,
        Auto,
        Float32,
        Int8,
        Int8Float16,
        Int16,
        Float16,
    }

    struct TranslatorConfig {
        compute_type: ComputeType,
        device_indices: Vec<i32>,
        num_threads_per_replica: usize,
        max_queued_batches: i64,
        cpu_core_offset: i32,
    }

    enum BatchType {
        Examples,
        Tokens,
    }

    struct TranslationOptions<'a> {
        beam_size: usize,
        patience: f32,
        length_penalty: f32,
        coverage_penalty: f32,
        repetition_penalty: f32,
        no_repeat_ngram_size: usize,
        disable_unk: bool,
        suppress_sequences: Vec<VecStr<'a>>,
        prefix_bias_beta: f32,
        // end_token,
        return_end_token: bool,
        max_input_length: usize,
        max_decoding_length: usize,
        min_decoding_length: usize,
        sampling_topk: usize,
        sampling_topp: f32,
        sampling_temperature: f32,
        use_vmap: bool,
        num_hypotheses: usize,
        return_scores: bool,
        return_attention: bool,
        return_alternatives: bool,
        min_alternative_expansion_prob: f32,
        replace_unknowns: bool,
        // callback,
        max_batch_size: usize,
        batch_type: BatchType,
    }

    struct TranslationResult {
        hypotheses: Vec<VecString>,
        scores: Vec<f32>,
        // attention: Vec<Vec<Vec<f32>>>,
    }

    unsafe extern "C++" {
        include!("ctranslate2/include/translator.h");

        type Translator;

        fn new_translator(
            model_path: &str,
            cuda: bool,
            config: TranslatorConfig,
        ) -> Result<UniquePtr<Translator>>;

        fn translate_batch(
            self: &Translator,
            source: Vec<VecStr>,
            target_prefix: Vec<VecStr>,
            options: TranslationOptions,
        ) -> Result<Vec<TranslationResult>>;
    }
}

#[derive(Debug)]
pub struct TranslationOptions<T: AsRef<str>> {
    /// Beam size to use for beam search (set 1 to run greedy search).
    pub beam_size: usize,
    /// Beam search patience factor, as described in https://arxiv.org/abs/2204.05424.
    /// The decoding will continue until beam_size*patience hypotheses are finished.
    pub patience: f32,
    /// Exponential penalty applied to the length during beam search.
    /// The scores are normalized with:
    ///   hypothesis_score /= (hypothesis_length ** length_penalty)
    pub length_penalty: f32,
    /// Coverage penalty weight applied during beam search.
    pub coverage_penalty: f32,
    /// Penalty applied to the score of previously generated tokens, as described in
    /// https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
    pub repetition_penalty: f32,
    /// Prevent repetitions of ngrams with this size (set 0 to disable).
    pub no_repeat_ngram_size: usize,
    /// Disable the generation of the unknown token.
    pub disable_unk: bool,
    /// Disable the generation of some sequences of tokens.
    pub suppress_sequences: Vec<Vec<T>>,
    /// Biases decoding towards a given prefix, see https://arxiv.org/abs/1912.03393 --section 4.2
    /// Only activates biased-decoding when beta is in range (0, 1) and SearchStrategy is set to BeamSearch.
    /// The closer beta is to 1, the stronger the bias is towards the given prefix.
    ///
    /// If beta <= 0 and a non-empty prefix is given, then the prefix will be used as a
    /// hard-prefix rather than a soft, biased-prefix.
    pub prefix_bias_beta: f32,
    // Stop the decoding on one of these tokens (defaults to the model EOS token).
    // end_token,
    /// Include the end token in the result.
    pub return_end_token: bool,
    /// Truncate the inputs after this many tokens (set 0 to disable truncation).
    pub max_input_length: usize,
    /// Decoding length constraints.
    pub max_decoding_length: usize,
    /// Decoding length constraints.
    pub min_decoding_length: usize,
    /// Randomly sample from the top K candidates (set 0 to sample from the full output distribution).
    pub sampling_topk: usize,
    /// Keep the most probable tokens whose cumulative probability exceeds this value.
    pub sampling_topp: f32,
    /// High temperature increase randomness.
    pub sampling_temperature: f32,
    /// Allow using the vocabulary map included in the model directory, if it exists.
    pub use_vmap: bool,
    /// Number of hypotheses to store in the TranslationResult class.
    pub num_hypotheses: usize,
    /// Store scores in the TranslationResult class.
    pub return_scores: bool,
    /// Store attention vectors in the TranslationResult class.
    pub return_attention: bool,
    /// Return alternatives at the first unconstrained decoding position. This is typically
    /// used with a target prefix to provide alternatives at a specifc location in the
    /// translation.
    pub return_alternatives: bool,
    /// Minimum probability to expand an alternative.
    pub min_alternative_expansion_prob: f32,
    /// Replace unknown target tokens by the original source token with the highest attention.
    pub replace_unknowns: bool,
    /// Function to call for each generated token in greedy search.
    // Returns true indicate the current generation is considered finished thus can be stopped early.
    // callback,
    pub max_batch_size: usize,
    pub batch_type: BatchType,
}

impl Default for TranslationOptions<String> {
    fn default() -> Self {
        Self {
            beam_size: 2,
            patience: 1.,
            length_penalty: 1.,
            coverage_penalty: 0.,
            repetition_penalty: 1.,
            no_repeat_ngram_size: 0,
            disable_unk: false,
            suppress_sequences: vec![],
            prefix_bias_beta: 0.,
            return_end_token: false,
            max_input_length: 1024,
            max_decoding_length: 256,
            min_decoding_length: 1,
            sampling_topk: 1,
            sampling_topp: 1.,
            sampling_temperature: 1.,
            use_vmap: false,
            num_hypotheses: 1,
            return_scores: false,
            return_attention: false,
            return_alternatives: false,
            min_alternative_expansion_prob: 0.,
            replace_unknowns: false,
            max_batch_size: 0,
            batch_type: BatchType::default(),
        }
    }
}

impl<T: AsRef<str>> TranslationOptions<T> {
    fn to_ffi(&self) -> ffi::TranslationOptions {
        ffi::TranslationOptions {
            beam_size: self.beam_size,
            patience: self.patience,
            length_penalty: self.length_penalty,
            coverage_penalty: self.coverage_penalty,
            repetition_penalty: self.repetition_penalty,
            no_repeat_ngram_size: self.no_repeat_ngram_size,
            disable_unk: self.disable_unk,
            suppress_sequences: vec_ffi_vecstr(self.suppress_sequences.as_ref()),
            prefix_bias_beta: self.prefix_bias_beta,
            return_end_token: self.return_end_token,
            max_input_length: self.max_input_length,
            max_decoding_length: self.max_decoding_length,
            min_decoding_length: self.min_decoding_length,
            sampling_topk: self.sampling_topk,
            sampling_topp: self.sampling_topp,
            sampling_temperature: self.sampling_temperature,
            use_vmap: self.use_vmap,
            num_hypotheses: self.num_hypotheses,
            return_scores: self.return_scores,
            return_attention: self.return_attention,
            return_alternatives: self.return_alternatives,
            min_alternative_expansion_prob: self.min_alternative_expansion_prob,
            replace_unknowns: self.replace_unknowns,
            max_batch_size: self.max_batch_size,
            batch_type: match self.batch_type {
                BatchType::Examples => ffi::BatchType::Examples,
                BatchType::Tokens => ffi::BatchType::Tokens,
            },
        }
    }
}

pub struct Translator {
    ptr: UniquePtr<ffi::Translator>,
}

impl Translator {
    pub fn new<T: AsRef<str>>(
        model_path: T,
        device: Device,
        config: Config,
    ) -> anyhow::Result<Translator> {
        Ok(Translator {
            ptr: ffi::new_translator(
                model_path.as_ref(),
                match device {
                    Device::CPU => false,
                    Device::CUDA => true,
                },
                ffi::TranslatorConfig {
                    compute_type: match config.compute_type {
                        ComputeType::Default => ffi::ComputeType::Default,
                        ComputeType::Auto => ffi::ComputeType::Auto,
                        ComputeType::Float32 => ffi::ComputeType::Float32,
                        ComputeType::Int8 => ffi::ComputeType::Int8,
                        ComputeType::Int8Float16 => ffi::ComputeType::Int8Float16,
                        ComputeType::Int16 => ffi::ComputeType::Int16,
                        ComputeType::Float16 => ffi::ComputeType::Float16,
                    },
                    device_indices: config.device_indices,
                    num_threads_per_replica: config.num_threads_per_replica,
                    max_queued_batches: config.max_queued_batches,
                    cpu_core_offset: config.cpu_core_offset,
                },
            )?,
        })
    }

    pub fn translate_batch<T, U, V>(
        &self,
        source: &[Vec<T>],
        target_prefix: &[Vec<U>],
        options: &TranslationOptions<V>,
    ) -> anyhow::Result<Vec<TranslationResult>>
    where
        T: AsRef<str>,
        U: AsRef<str>,
        V: AsRef<str>,
    {
        Ok(self
            .ptr
            .translate_batch(
                vec_ffi_vecstr(source),
                vec_ffi_vecstr(target_prefix),
                options.to_ffi(),
            )?
            .into_iter()
            .map(TranslationResult::from)
            .collect())
    }
}

#[derive(Debug)]
pub struct TranslationResult {
    pub hypotheses: Vec<Vec<String>>,
    pub scores: Vec<f32>,
}

impl From<ffi::TranslationResult> for TranslationResult {
    fn from(r: ffi::TranslationResult) -> Self {
        Self {
            hypotheses: r.hypotheses.into_iter().map(|h| h.v).collect(),
            scores: r.scores,
        }
    }
}

impl TranslationResult {
    pub fn output(&self) -> Option<&Vec<String>> {
        self.hypotheses.first()
    }

    pub fn score(&self) -> Option<f32> {
        self.scores.first().copied()
    }

    pub fn num_hypotheses(&self) -> usize {
        self.hypotheses.len()
    }

    pub fn has_scores(&self) -> bool {
        !self.scores.is_empty()
    }
}

#[inline]
fn vec_ffi_vecstr<T: AsRef<str>>(src: &[Vec<T>]) -> Vec<ffi::VecStr> {
    src.iter()
        .map(|v| ffi::VecStr {
            v: v.iter().map(|s| s.as_ref()).collect(),
        })
        .collect()
}
