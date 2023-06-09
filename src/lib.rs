// lib.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::ffi::{c_char, c_int, CStr, CString};
use std::path::Path;

use anyhow::{anyhow, Result};
use tokenizers::{Decoder, EncodeInput, Tokenizer};

struct StringArray {
    chars: Vec<*mut c_char>,
}

impl StringArray {
    fn new<T: Into<Vec<u8>>>(strings: Vec<T>) -> StringArray {
        let chars: Vec<_> = strings
            .into_iter()
            .map(|s| {
                CString::new(s)
                    .expect("failed to convert a string to CString")
                    .into_raw()
            })
            .collect();
        StringArray { chars }
    }

    fn to_string_array(&self) -> ctranslate2_sys::StringArray {
        ctranslate2_sys::StringArray {
            strings: self.chars.as_ptr(),
            length: self.chars.len() as c_int,
        }
    }
}

impl Drop for StringArray {
    fn drop(&mut self) {
        for c in &self.chars {
            unsafe {
                drop(CString::from_raw(*c));
            }
        }
    }
}

pub struct Translator {
    translator: ctranslate2_sys::Translator,
    tokenizer: Tokenizer,
}

impl Translator {
    pub fn new<T: Into<Vec<u8>>, U: AsRef<Path>>(model: T, tokenizer: U) -> Result<Translator> {
        let model_path = CString::new(model).expect("failed to convert the given path to CString");
        let translator = unsafe { ctranslate2_sys::Translator::new(model_path.as_ptr()) };

        Ok(Translator {
            translator,
            tokenizer: Tokenizer::from_file(tokenizer)
                .map_err(|err| anyhow!("failed to load a tokenizer: {err}"))?,
        })
    }

    pub fn translate<'s, T, U>(&mut self, source: T, target_prefix: Vec<U>) -> Result<String>
        where
            T: Into<EncodeInput<'s>>,
            U: Into<Vec<u8>>,
    {
        let source = StringArray::new(
            self.tokenizer
                .encode(source, true)
                .map_err(|err| anyhow!("failed to encode the given input: {err}"))?
                .get_tokens()
                .to_vec(),
        );
        let target_prefix = StringArray::new(target_prefix);
        let results = unsafe {
            self.translator
                .translate(source.to_string_array(), target_prefix.to_string_array())
        };

        let string_ptrs =
            unsafe { std::slice::from_raw_parts(results.strings, results.length as usize) };

        let mut res = vec![];
        for i in 1..results.length {
            let c_string = unsafe { CStr::from_ptr(string_ptrs[i as usize]) };
            res.push(c_string.to_string_lossy().into_owned());
        }

        let res_string = self.tokenizer.get_decoder().unwrap().decode(res).unwrap();
        unsafe {
            ctranslate2_sys::release_string_array(results);
        };

        Ok(res_string)
    }
}

impl Drop for Translator {
    fn drop(&mut self) {
        unsafe {
            self.translator.destruct();
        }
    }
}
