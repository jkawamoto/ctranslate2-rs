// translator.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use cxx::UniquePtr;

use crate::config::{ComputeType, Config, Device};

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
        ) -> Result<Vec<TranslationResult>>;
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

    pub fn translate_batch<T, U>(
        &self,
        source: &[Vec<T>],
        target_prefix: &[Vec<U>],
    ) -> anyhow::Result<Vec<TranslationResult>>
    where
        T: AsRef<str>,
        U: AsRef<str>,
    {
        Ok(self
            .ptr
            .translate_batch(vec_ffi_vecstr(source), vec_ffi_vecstr(target_prefix))?
            .into_iter()
            .map(|r| TranslationResult {
                hypotheses: r.hypotheses.into_iter().map(|h| h.v).collect(),
                scores: r.scores,
            })
            .collect())
    }
}

#[derive(Debug)]
pub struct TranslationResult {
    pub hypotheses: Vec<Vec<String>>,
    pub scores: Vec<f32>,
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
