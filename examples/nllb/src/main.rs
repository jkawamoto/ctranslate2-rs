// main.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, stdout, Write};

use anyhow::Result;
use clap::Parser;

use ct2rs::config::Config;
use ct2rs::Translator;

/// Translate a file using NLLB.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the output file. If not specified, output to stdout.
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,
    /// Path to the file contains prompts.
    #[arg(short, long, value_name = "FILE", default_value = "prompt.txt")]
    prompt: String,
    /// Target language.
    #[arg(short, long, value_name = "LANG", default_value = "jpn_Jpan")]
    target: String,
    /// Path to the directory that contains model.bin.
    path: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t = Translator::new(args.path, Config::default())?;

    let sources = BufReader::new(File::open(args.prompt)?)
        .lines()
        .collect::<Result<Vec<String>, io::Error>>()?;
    let target_prefixes = vec![vec![args.target]; sources.len()];

    let res = t.translate_batch(sources, target_prefixes, &Default::default())?;
    let mut out: BufWriter<Box<dyn Write>> = BufWriter::new(match args.output {
        None => Box::new(stdout()),
        Some(p) => Box::new(File::create(p)?),
    });
    for (r, _) in res {
        writeln!(out, "{r}")?;
    }

    Ok(())
}
