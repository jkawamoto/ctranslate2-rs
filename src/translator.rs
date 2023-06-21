// translator.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use cxx::UniquePtr;

#[cxx::bridge]
mod ffi {
    struct VecString {
        v: Vec<String>,
    }

    struct VecStr<'a> {
        v: Vec<&'a str>,
    }

    struct TranslationResult {
        hypotheses: Vec<VecString>,
        scores: Vec<f32>,
        // attention: Vec<Vec<Vec<f32>>>,
    }

    unsafe extern "C++" {
        include!("ctranslate2/include/translator.h");

        type Translator;

        fn new_translator(model_path: &str) -> Result<UniquePtr<Translator>>;

        fn translate_batch(
            &self,
            source: Vec<VecStr>,
            target_prefix: Vec<VecStr>,
        ) -> Result<Vec<TranslationResult>>;
    }
}

pub struct Translator {
    ptr: UniquePtr<ffi::Translator>,
}

impl Translator {
    pub fn new<T: AsRef<str>>(model_path: T) -> anyhow::Result<Translator> {
        Ok(Translator {
            ptr: ffi::new_translator(model_path.as_ref())?,
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
