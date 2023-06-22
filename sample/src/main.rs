// main.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::env;

use anyhow::{anyhow, Result};

use ctranslate2::config::{Config, Device};
use ctranslate2::{TranslationOptions, Translator};

fn main() -> Result<()> {
    let path = env::args()
        .nth(1)
        .ok_or(anyhow!("no model path is given"))?;
    let t = Translator::new(path, Device::CPU, Config::default())?;
    let res = t.translate_batch(
        vec![
            "Hello world!",
            "This library provides Rust bindings for CTranslate2.",
        ],
        vec![vec!["deu_Latn"], vec!["jpn_Jpan"]],
        &TranslationOptions {
            return_scores: true,
            ..Default::default()
        },
    )?;
    for r in res {
        println!("{}, (score: {:?})", r.0, r.1);
    }

    Ok(())
}
