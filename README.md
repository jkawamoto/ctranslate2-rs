# ctranslate2-rs

[![Latest version](https://img.shields.io/crates/v/ct2rs.svg)](https://crates.io/crates/ct2rs)
[![docs.rs](https://img.shields.io/docsrs/ct2rs)](https://docs.rs/ct2rs)
[![GitHub License](https://img.shields.io/github/license/jkawamoto/ctranslate2-rs)](https://github.com/jkawamoto/ctranslate2-rs/blob/main/LICENSE)
[![Build](https://github.com/jkawamoto/ctranslate2-rs/actions/workflows/build.yaml/badge.svg)](https://github.com/jkawamoto/ctranslate2-rs/actions/workflows/build.yaml)

This library provides Rust bindings for [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2).

## Usage

Add this crate to your `Cargo.toml` with selecting the backends you want to use as the features:

```toml
[dependencies]
ct2rs = { version = "0.9.7", features = ["cuda", "dnnl", "mkl"] }
```

Or you can use platform-specific default features by using the `ct2rs-platform` crate:

```toml
[dependencies]
ct2rs = { version = "0.9.7", package = "ct2rs-platform" }
```

If you want [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) model support,
you need to enable the `whisper` feature.

See below for more details about the available features.

### Prerequisites

The installation of [CMake](https://cmake.org/) is required to compile the library.

### Additional notes for Windows

Setting the environment variable `RUSTFLAGS=-C target-feature=+crt-static` might be required.

## Features

### Backend Futures

- `cuda`: Enables CUDA support
- `cudnn`: Enables cuDNN support

The above features require setting the `CUDA_TOOLKIT_ROOT_DIR` environment variable appropriately.

- `mkl`: Enables [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) support
- `openblas`: Enables [OpenBLAS](https://www.openblas.net/) support
- `dnnl`: Enables [oneDNN](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html) support
- `ruy`: Enables [Ruy](https://github.com/google/ruy) support
- `accelerate`: Enables [Apple Accelerate](https://developer.apple.com/documentation/accelerate) support (macOS only)
- `openmp-runtime-comp`: Enables OpenMP runtime support
- `openmp-runtime-intel`: Enables OpenMP runtime support for Intel compilers
- `msse4_1`: Enables MSSE4.1 support

Multiple features can be enabled at the same time.

By default, the `ruy` feature is enabled.

If you want to use platform-specific default features, use the `ct2rs-platform` crate.

### GPU-Specific Features

- `flash-attention`:
  Enables [Flash Attention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention)
- `tensor-parallel`:
  Enables [Tensor Parallelism](https://huggingface.co/docs/text-generation-inference/conceptual/tensor_parallelism)
- `cuda-dynamic-loading`: Enables dynamic loading of CUDA libraries at runtime instead of static linking (requires
  CUDA >= 11)
- `cuda-small-binary`: Reduces binary size by compressing device code

### Tokenizer Features

- `sentencepiece`: Enables [SentencePiece](https://github.com/google/sentencepiece) tokenizer support
- `tokenizers`: Enables HuggingFace's [Tokenizers](https://github.com/huggingface/tokenizers) library support
- `all-tokenizers`: Enables both `sentencepiece` and `tokenizers` support

### Additional Features

- `whisper`: Enables [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) model support
- `hub`: Enables [HuggingFace Hub](https://huggingface.co/docs/hub) integration

### Platform Specific Features

When `ct2rs-platform` is used, the following features are automatically selected based on the platform:

- Windows: `openmp-runtime-intel`, `dnnl`, `cuda`, `cudnn`, `cuda-dynamic-loading`, `mkl`
- Intel MacOS: `dnnl`, `mkl`
- Apple Silicon MacOS: `accelerate`, `ruy`
- Linux (non-ARM): `dnnl`, `openmp-runtime-comp`, `cuda`, `cudnn`, `cuda-dynamic-loading`, `mkl`, `tensor-parallel`,
  `msse4_1`
- Linux (ARM): `openmp-runtime-comp`, `openblas`, `ruy`

## Supported Models

The ct2rs crate has been tested and confirmed to work with the following models:

- BART
- BLOOM
- FALCON
- Marian-MT
- MPT
- NLLB
- GPT-2
- GPT-J
- OPT
- T5
- Whisper

Please see the respective
[examples](ct2rs/examples)
for each model.

## Stream API

This crate also offers a streaming API that utilizes callback closures.
Please refer to the [example code](ct2rs/examples/stream.rs)
for more information.

## Model Conversion for CTranslate2

To use model files with CTranslate2, they must first be converted.
Below is an example of how to convert the `nllb-200-distilled-600M` model:

```shell-session
pip install ctranslate2 huggingface_hub torch transformers
ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-200-distilled-600M \
    --copy_files tokenizer.json
```

For more details, please refer to
the [CTranslate2's docs](https://opennmt.net/CTranslate2/guides/transformers.html#nllb).

## License

This application is released under the MIT License. For details, see the [LICENSE](LICENSE) file.
