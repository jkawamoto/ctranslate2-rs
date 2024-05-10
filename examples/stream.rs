// stream.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! Translate a file using Marian-MT models with the Stream API.
//!
//! In this example, we will use
//! the [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian) model
//! to perform a translation from English to German with the stream API.
//!
//! First, convert the model files with the following command:
//!
//! ```bash
//! pip install -U ctranslate2 huggingface_hub torch transformers
//!
//! ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de --output_dir opus-mt-en-de \
//!     --copy_files source.spm target.spm
//! ```
//!
//! Create a file named `prompt.txt`, write the sentence you want to translate into it,
//! and save the file.
//! Then, execute the sample code below with the following command:
//!
//! ```bash
//! cargo run --example stream -- ./opus-mt-en-de
//! ```
//!

use std::fs::File;
use std::io::{BufRead, BufReader, stdout, Write};

use anyhow::Result;
use clap::Parser;

use ct2rs::{GenerationStepResult, TranslationOptions, Translator};
use ct2rs::config::{Config, Device};

/// Translate a file using Marian-MT model with the Stream API.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the file contains prompts.
    #[arg(short, long, value_name = "FILE", default_value = "prompt.txt")]
    prompt: String,
    /// Use CUDA.
    #[arg(short, long)]
    cuda: bool,
    /// Path to the directory that contains model.bin.
    path: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = if args.cuda {
        Config {
            device: Device::CUDA,
            device_indices: vec![0],
            ..Config::default()
        }
    } else {
        Config::default()
    };

    let t = Translator::new(&args.path, &cfg)?;
    let source =
        BufReader::new(File::open(args.prompt)?)
            .lines()
            .fold(Ok(String::new()), |acc, line| {
                acc.and_then(|mut acc| {
                    line.map(|l| {
                        acc.push_str(&l);
                        acc
                    })
                })
            })?;

    let _ = t.translate_batch(
        &vec![source],
        &TranslationOptions {
            // beam_size must be 1 to use the stream API.
            beam_size: 1,
            ..Default::default()
        },
        // Each time a new token is generated, the following callback closure is called.
        // In this example, it writes to the standard output sequentially.
        Some(&mut |r: GenerationStepResult| -> bool {
            print!("{}", r.text);
            let _ = stdout().flush();
            false
        }),
    )?;

    Ok(())
}
