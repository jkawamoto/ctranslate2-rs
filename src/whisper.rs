// whisper.py.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This module provides a speach transcriber.

use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis, stack};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use serde::Deserialize;

use super::{Config, sys, Tokenizer};
pub use super::sys::WhisperOptions;
use super::tokenizers::hf;

const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor_config.json";

/// A speach transcriber using the Whisper speech recognition model published by OpenAI.
///
/// # Example
/// ```no_run
/// use ct2rs::Whisper;
///
/// # fn main() -> anyhow::Result<()>{ ///
/// let whisper = Whisper::new("/path/to/model", Default::default())?;
///
/// let sampling_rate = whisper.sampling_rate();
/// // Sample the source audio at the sampling rates shown above.
/// // Each sample must be normalized to the range [-1, 1].
/// let samples = vec![];
///
/// let res = whisper.generate(&samples, None, false, &Default::default())?;
/// for r in res {
///     println!("{}", r);
/// }
/// # Ok(())
/// # }
/// ```
pub struct Whisper {
    whisper: sys::Whisper,
    tokenizer: hf::Tokenizer,
    config: PreprocessorConfig,
}

impl Whisper {
    /// Initializes the transcriber.
    ///
    /// # Arguments
    /// * `path` - A path to the directory containing the language model to be loaded.
    /// * `config` - A [`Config`] structure that specifies various settings
    ///   and configurations for the `Whisper`.
    ///
    /// # Returns
    /// Returns a `Result` that, if successful, contains the initialized `Whisper`. If an error
    /// occurs during initialization, the function will return an error wrapped in the `Result`.
    pub fn new<T: AsRef<Path>>(model_path: T, config: Config) -> Result<Self> {
        Ok(Self {
            whisper: sys::Whisper::new(&model_path, config)?,
            tokenizer: hf::Tokenizer::new(&model_path)?,
            config: PreprocessorConfig::read(model_path.as_ref().join(PREPROCESSOR_CONFIG_FILE))?,
        })
    }

    /// Transcribe the given samples.
    ///
    /// # Arguments
    /// * `samples` - Samples of the source audio. They must be sampled at the sampling rate
    ///   returned by [`sampling_rate`][Whisper::sampling_rate] method and normalized to the range
    ///   `[-1, 1]`. If the samples are longer than the maximum number of samples returned by
    ///   [`n_samples`][Whisper::n_samples] method, they will be processed in segments.
    /// * `language` - An optional language setting. It transcribes assuming the specified language.
    ///   If `None`, it uses Whisper's language detection.
    /// * `timestamp` - If `true`, the output will include timestamps.
    /// * `options` - Settings.
    ///
    /// # Returns
    /// Returns a `Result` containing a vector of [`WhisperGenerationResult`] if successful,
    /// or an error if the translation fails.
    pub fn generate(
        &self,
        samples: &[f32],
        language: Option<&str>,
        timestamp: bool,
        options: &WhisperOptions,
    ) -> Result<Vec<String>> {
        let mut mel_spectrogram_vec = vec![];
        for chunk in samples.chunks(self.config.n_samples) {
            let stft = if chunk.len() < self.config.n_samples {
                let mut padded_chunk = Vec::with_capacity(self.config.n_samples);
                padded_chunk.extend_from_slice(chunk);
                padded_chunk
                    .extend(std::iter::repeat(0.0).take(self.config.n_samples - chunk.len()));

                stft(&padded_chunk, self.config.n_fft, self.config.hop_length)
            } else {
                stft(chunk, self.config.n_fft, self.config.hop_length)
            };

            // Compute Mel Spectrogram
            mel_spectrogram_vec.push(mel_spectrogram(&stft, &self.config.mel_filters));
        }

        let mut mel_spectrogram = stack(
            Axis(0),
            &mel_spectrogram_vec
                .iter()
                .map(|a| a.view())
                .collect::<Vec<_>>(),
        )?;
        if !mel_spectrogram.is_standard_layout() {
            mel_spectrogram = mel_spectrogram.as_standard_layout().into_owned()
        }

        let shape = mel_spectrogram.shape().to_vec();
        let storage_view = sys::StorageView::new(
            &shape,
            mel_spectrogram.as_slice_mut().unwrap(),
            Default::default(),
        )?;

        // Detect language.
        let lang_token = match language {
            Some(lang) => {
                format!("<|{}|>", lang)
            }
            None => {
                let detection_result = self.whisper.detect_language(&storage_view)?;
                detection_result
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("failed to detect language"))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("failed to detect language"))?
                    .language
            }
        };

        // Transcribe.
        let mut prompt = vec!["<|startoftranscript|>", &lang_token, "<|transcribe|>"];
        if !timestamp {
            prompt.push("<|notimestamps|>");
        }
        self.whisper
            .generate(
                &storage_view,
                &vec![prompt; mel_spectrogram_vec.len()],
                options,
            )?
            .into_iter()
            .map(|res| {
                let r = res
                    .sequences
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("failed to transcribe samples"))?;
                self.tokenizer.decode(r)
            })
            .collect()
    }

    /// Returns the expected sampling rate.
    pub fn sampling_rate(&self) -> usize {
        self.config.sampling_rate
    }

    /// Max number of samples per batch.
    pub fn n_samples(&self) -> usize {
        self.config.n_samples
    }

    /// Returns `true` if this model is multilingual.
    #[inline]
    pub fn is_multilingual(&self) -> bool {
        self.whisper.is_multilingual()
    }

    /// Returns the number of languages supported.
    #[inline]
    pub fn num_languages(&self) -> usize {
        self.whisper.num_languages()
    }

    /// Number of batches in the work queue.
    #[inline]
    pub fn num_queued_batches(&self) -> usize {
        self.whisper.num_queued_batches()
    }

    /// Number of batches in the work queue or currently processed by a worker.
    #[inline]
    pub fn num_active_batches(&self) -> usize {
        self.whisper.num_active_batches()
    }

    /// Number of parallel replicas.
    #[inline]
    pub fn num_replicas(&self) -> usize {
        self.whisper.num_replicas()
    }
}

impl Debug for Whisper {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.whisper)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct PreprocessorConfig {
    chunk_length: usize,
    feature_extractor_type: String,
    feature_size: usize,
    hop_length: usize,
    n_fft: usize,
    n_samples: usize,
    nb_max_frames: usize,
    padding_side: String,
    padding_value: f32,
    processor_class: String,
    return_attention_mask: bool,
    sampling_rate: usize,
    mel_filters: Array2<f32>,
}

impl PreprocessorConfig {
    fn read<T: AsRef<Path>>(path: T) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        #[derive(Deserialize)]
        struct PreprocessorConfigAux {
            chunk_length: usize,
            feature_extractor_type: String,
            feature_size: usize,
            hop_length: usize,
            n_fft: usize,
            n_samples: usize,
            nb_max_frames: usize,
            padding_side: String,
            padding_value: f32,
            processor_class: String,
            return_attention_mask: bool,
            sampling_rate: usize,
            mel_filters: Vec<Vec<f32>>,
        }
        let aux: PreprocessorConfigAux = serde_json::from_reader(reader)?;

        let rows = aux.mel_filters.len();
        let cols = aux
            .mel_filters
            .first()
            .map(|row| row.len())
            .unwrap_or_default();

        Ok(Self {
            chunk_length: aux.chunk_length,
            feature_extractor_type: aux.feature_extractor_type,
            feature_size: aux.feature_size,
            hop_length: aux.hop_length,
            n_fft: aux.n_fft,
            n_samples: aux.n_samples,
            nb_max_frames: aux.nb_max_frames,
            padding_side: aux.padding_side,
            padding_value: aux.padding_value,
            processor_class: aux.processor_class,
            return_attention_mask: aux.return_attention_mask,
            sampling_rate: aux.sampling_rate,
            mel_filters: Array2::from_shape_vec(
                (rows, cols),
                aux.mel_filters.into_iter().flatten().collect(),
            )?,
        })
    }
}

fn stft(samples: &[f32], n_fft: usize, hop_length: usize) -> Array2<Complex<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let n_frames = (samples.len() - 1) / hop_length + 1;
    let mut stft = Array2::zeros((n_fft / 2 + 1, n_frames));

    let mut padded_samples = samples.to_vec();
    padded_samples.extend(vec![0.0; n_fft]);

    for (i, frame) in padded_samples
        .windows(n_fft)
        .step_by(hop_length)
        .take(n_frames)
        .enumerate()
    {
        let mut fft_input: Vec<Complex<f32>> =
            frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut fft_input);
        for (j, value) in fft_input.iter().take(n_fft / 2 + 1).enumerate() {
            stft[[j, i]] = *value;
        }
    }

    stft
}

fn mel_spectrogram(stft: &Array2<Complex<f32>>, mel_filter_bank: &Array2<f32>) -> Array2<f32> {
    let spectrum = stft.mapv(|x| x.norm_sqr());

    let res = mel_filter_bank.dot(&spectrum).mapv(|x| x.log10());
    let global_max = res.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    res.mapv(|x| (x.max(global_max - 8.0) + 4.0) / 4.0)
}

#[cfg(test)]
mod tests {
    use crate::Whisper;

    #[test]
    #[ignore]
    fn test_generator_debug() {
        let w = Whisper::new("data/whisper-tiny-ct2", Default::default()).unwrap();
        assert!(format!("{:?}", w).contains("whisper-tiny-ct2"));
    }
}
