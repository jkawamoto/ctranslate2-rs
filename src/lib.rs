// lib.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use anyhow::{anyhow, bail, Error, Result};
use tokenizers::{Decoder, EncodeInput, Tokenizer};

use ctranslate2_sys::{release_translation_result, TranslationResult};

use crate::array::{StringArrayAux, StringArrayPtr};

mod array;

const TOKENIZER_FILENAME: &str = "tokenizer.json";

struct TranslationResultPtr(*const TranslationResult, StringArrayPtr, Option<f32>);

impl TryFrom<*const TranslationResult> for TranslationResultPtr {
    type Error = Error;

    fn try_from(ptr: *const TranslationResult) -> Result<Self> {
        if ptr.is_null() {
            bail!("empty result");
        }
        unsafe {
            if (*ptr).hypothesis.is_null() {
                bail!("empty hypothesis");
            }
            let score = if (*ptr).score.is_null() {
                None
            } else {
                Some(*(*ptr).score)
            };

            Ok(Self(ptr, StringArrayPtr::from((*ptr).hypothesis), score))
        }
    }
}

impl TranslationResultPtr {
    fn hypothesis(&self) -> &StringArrayPtr {
        &self.1
    }
    fn score(&self) -> Option<f32> {
        self.2
    }
}

impl Drop for TranslationResultPtr {
    fn drop(&mut self) {
        unsafe {
            release_translation_result(self.0);
        }
    }
}

pub struct Translator {
    translator: ctranslate2_sys::Translator,
    tokenizer: Tokenizer,
}

impl Translator {
    pub fn new<T: AsRef<Path>>(path: T) -> Result<Translator> {
        let tokenizer = path.as_ref().join(TOKENIZER_FILENAME);
        let model_path = CString::new(path.as_ref().as_os_str().as_bytes())
            .expect("failed to convert the given path to CString");
        let translator = unsafe { ctranslate2_sys::Translator::new(model_path.as_ptr()) };

        Ok(Translator {
            translator,
            tokenizer: Tokenizer::from_file(tokenizer)
                .map_err(|err| anyhow!("failed to load a tokenizer: {err}"))?,
        })
    }

    pub fn translate<'a, T, U>(
        &self,
        source: T,
        target_prefix: Vec<U>,
    ) -> Result<(String, Option<f32>)>
    where
        T: Into<EncodeInput<'a>>,
        U: Into<Vec<u8>>,
    {
        let source = StringArrayAux::from(
            self.tokenizer
                .encode(source, true)
                .map_err(|err| anyhow!("failed to encode the given input: {err}"))?
                .get_tokens()
                .to_vec(),
        );
        let target_prefix = StringArrayAux::from(target_prefix);
        let res = TranslationResultPtr::try_from(unsafe {
            self.translator
                .translate(source.as_ptr(), target_prefix.as_ptr())
        })?;
        Ok((
            self.tokenizer
                .get_decoder()
                .unwrap()
                .decode(
                    res.hypothesis()
                        .to_vec()?
                        .into_iter()
                        .skip(target_prefix.len())
                        .collect::<Vec<_>>(),
                )
                .map_err(|err| anyhow!("failed to decode: {err}"))?,
            res.score(),
        ))
    }
}

impl Drop for Translator {
    fn drop(&mut self) {
        unsafe {
            self.translator.destruct();
        }
    }
}
