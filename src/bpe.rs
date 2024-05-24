// bpe.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This module provides a tokenizer based on the Byte Pair Encoding (BPE) model.
//!
//! Byte Pair Encoding is a subword tokenization technique that can dynamically adjust
//! the vocabulary based on the provided corpus, often leading to more efficient
//! representation of text data in machine learning models. For a deeper understanding
//! of BPE, refer to the [original paper](https://www.aclweb.org/anthology/P16-1162/).
//!
//! This module allows for the creation of a [`Tokenizer`] structure instance by specifying a path
//! to a directory containing `vocab.json` and `merges.txt`, which are essential for the BPE
//! algorithm.
//! You can also specify optional suffixes for the decoder to fine-tune the behavior of the decoding
//! process. For more details on how to use suffixes with the BPE decoder, please see the
//! documentation for
//! [BPEDecoder](https://docs.rs/tokenizers/latest/tokenizers/decoders/bpe/struct.BPEDecoder.html).
//!
//! ## Usage
//! The tokenizer instances created can be utilized in conjunction with structures like
//! `crate::Translator` and `crate::Generator` for tasks such as translation or text generation.
//!

use std::path::Path;

use anyhow::{anyhow, Result};
use tokenizers::decoders::bpe::BPEDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::processors::roberta::RobertaProcessing;
use tokenizers::Tokenizer as HFTokenizer;

use crate::tokenizers::Tokenizer;

const VOCAB_FILE: &str = "vocab.json";
const MERGES_FILE: &str = "merges.txt";

/// Create a tokenizer instance by specifying the path to a directory containing `vocab.json`
/// and `mergers.txt`.
pub fn new<T: AsRef<Path>>(path: T, decoder_suffix: Option<String>) -> Result<Tokenizer> {
    from_file(
        path.as_ref().join(VOCAB_FILE),
        path.as_ref().join(MERGES_FILE),
        decoder_suffix,
    )
}

/// Create a tokenizer instance by specifying the path to `vocab.json` and `mergers.txt`.
pub fn from_file<T: AsRef<Path>, U: AsRef<Path>>(
    vocab: T,
    merges: U,
    decoder_suffix: Option<String>,
) -> Result<Tokenizer> {
    let mut res = Tokenizer::from(HFTokenizer::new(
        BPE::from_file(
            vocab
                .as_ref()
                .to_str()
                .ok_or_else(|| anyhow!("invalid path: {}", vocab.as_ref().display()))?,
            merges
                .as_ref()
                .to_str()
                .ok_or_else(|| anyhow!("invalid path: {}", merges.as_ref().display()))?,
        )
        .build()
        .map_err(|err| anyhow!("failed to build a tokenizer: {err}"))?,
    ));

    res.with_decoder(match decoder_suffix {
        None => BPEDecoder::default(),
        Some(s) => BPEDecoder::new(s),
    })
    .with_post_processor(
        RobertaProcessing::new(("</s>".to_string(), 2), ("<s>".to_string(), 0)).trim_offsets(true),
    );

    Ok(res)
}
