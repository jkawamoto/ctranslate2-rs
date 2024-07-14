// whisper.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! Transcribe a WAV file using Whisper models.
//!
//! In this example, we will use
//! the [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) model
//! to transcribe a WAV file.
//! The original Python version of the code can be found in the
//! [CTranslate2 documentation](https://opennmt.net/CTranslate2/guides/transformers.html#whisper).
//!
//! First, convert the model files with the following command:
//!
//! ```bash
//! pip install -U ctranslate2 huggingface_hub torch transformers
//!
//! ct2-transformers-converter --model openai/whisper-tiny --output_dir whisper-tiny-ct2 \
//!     --copy_files preprocessor_config.json tokenizer.json
//! ```
//!
//! Then, execute the sample code below with the following command:
//!
//! ```bash
//! cargo run --example whisper -- ./whisper-tiny-ct2 audio.wav
//! ```
//!

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time;

use anyhow::Result;
use clap::Parser;
use hound::WavReader;
use ndarray::{Array2, Ix3};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use serde::Deserialize;

use ct2rs::sys::{StorageView, Whisper};
use ct2rs::{auto, Tokenizer};

const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor_config.json";

/// Transcribe a file using Whisper models.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the directory that contains model.bin.
    model_dir: PathBuf,
    /// Path to the WAVE file.
    audio_file: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = PreprocessorConfig::read(args.model_dir.join(PREPROCESSOR_CONFIG_FILE))?;

    let mut samples = read_audio(args.audio_file, cfg.sampling_rate)?;
    if samples.len() < cfg.n_samples {
        samples.append(&mut vec![0f32; cfg.n_samples - samples.len()]);
    } else {
        samples.truncate(cfg.n_samples);
    }

    // Compute STFT
    let stft = stft(&samples, cfg.n_fft, cfg.hop_length);

    // Compute Mel Spectrogram
    let mel_spectrogram = mel_spectrogram(&stft, &cfg.mel_filters);

    let shape = mel_spectrogram.shape();
    let new_shape = Ix3(1, shape[0], shape[1]);

    let mut mel_spectrogram = mel_spectrogram.into_shape(new_shape)?;
    if !mel_spectrogram.is_standard_layout() {
        mel_spectrogram = mel_spectrogram.as_standard_layout().into_owned()
    }

    let shape = mel_spectrogram.shape().to_vec();
    let storage_view = StorageView::new(
        &shape,
        mel_spectrogram.as_slice_mut().unwrap(),
        Default::default(),
    )?;

    // Load the model.
    let model = Whisper::new(&args.model_dir, Default::default()).unwrap();
    let tokenizer = auto::Tokenizer::new(&args.model_dir)?;

    let now = time::Instant::now();

    // Detect language.
    let lang = model.detect_language(&storage_view)?;
    println!("Detected language: {:?}", lang[0][0]);

    // Transcribe.
    let res = model.generate(
        &storage_view,
        &[vec![
            "<|startoftranscript|>",
            &lang[0][0].language,
            "<|transcribe|>",
            "<|notimestamps|>",
        ]],
        &Default::default(),
    )?;

    let elapsed = now.elapsed();

    match res.into_iter().next() {
        None => println!("Empty result"),
        Some(r) => {
            for v in r.sequences.into_iter() {
                println!("{:?}", tokenizer.decode(v));
            }
        }
    }
    println!("Time taken: {:?}", elapsed);

    Ok(())
}

fn read_audio<T: AsRef<Path>>(path: T, sample_rate: usize) -> Result<Vec<f32>> {
    // Should use a better resampling algorithm.
    fn resample(samples: Vec<f32>, src_rate: usize, target_rate: usize) -> Vec<f32> {
        samples
            .into_iter()
            .step_by(src_rate / target_rate)
            .collect()
    }

    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let max = 2_i32.pow((spec.bits_per_sample - 1) as u32) as f32;
    let samples = reader
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / max)
        .collect::<Vec<f32>>();

    if spec.channels == 1 {
        return Ok(resample(samples, spec.sample_rate as usize, sample_rate));
    }

    let mut mono = vec![];
    for chunk in samples.chunks(2) {
        if chunk.len() == 2 {
            mono.push((chunk[0] + chunk[1]) / 2.);
        }
    }

    Ok(resample(mono, spec.sample_rate as usize, sample_rate))
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

    let mut res = mel_filter_bank.dot(&spectrum).mapv(|x| x.log10());
    let global_max = res.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    res.mapv_inplace(|x| x.max(global_max - 8.0));
    res.mapv_inplace(|x| (x + 4.0) / 4.0);

    res
}

#[allow(dead_code)]
#[derive(Debug)]
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
