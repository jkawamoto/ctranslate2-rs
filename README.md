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
ct2rs = { version = "0.9.13", features = ["cuda", "dnnl", "mkl"] }
```

Or you can use platform-specific default features by using the `ct2rs-platform` crate:

```toml
[dependencies]
ct2rs = { version = "0.9.13", package = "ct2rs-platform" }
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
- `system-mkl`: Enables [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) support
  using the system-installed MKL library instead of downloading and building it
- `openblas`: Enables [OpenBLAS](https://www.openblas.net/) support (OpenBLAS needs to be installed manually
  via [vcpkg](https://vcpkg.io) on Windows)
- `dnnl`: Enables [oneDNN](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html) support
- `ruy`: Enables [Ruy](https://github.com/google/ruy) support
- `accelerate`: Enables [Apple Accelerate](https://developer.apple.com/documentation/accelerate) support (macOS only)
- `openmp-runtime-comp`: Enables OpenMP runtime support
- `openmp-runtime-intel`: Enables OpenMP runtime support for Intel compilers

Multiple features can be enabled at the same time.

To enable Streaming SIMD Extensions 4.1 (SSE4.1), add the `-C target-feature=+sse4.1` flag to `RUSTFLAGS` environment
variable.

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
- `system`: Skip compiling CTranslate2 and use the system's pre-installed shared library instead (requires setting the
  appropriate environment variables to locate the CTranslate2 shared library)

### Platform Specific Features

When `ct2rs-platform` is used, the following features are automatically selected based on the platform:

- Windows: `openmp-runtime-intel`, `dnnl`, `cuda`, `cudnn`, `cuda-dynamic-loading`, `mkl`
- Intel MacOS: `dnnl`, `mkl`
- Apple Silicon MacOS: `accelerate`, `ruy`
- Linux (non-ARM): `dnnl`, `openmp-runtime-comp`, `cuda`, `cudnn`, `cuda-dynamic-loading`, `mkl`, `tensor-parallel`
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

[CTranslate2](https://github.com/OpenNMT/CTranslate2) (which this crate builds and links to)
is licensed under MIT License.
If you redistribute binaries or source that include CTranslate2,
you must comply with the MIT terms, including preserving license and notice files.
For details, see CTranslate2’s upstream LICENSE/NOTICE.

The published crate may build and/or redistribute artifacts originating from the following third-party projects.
Each component remains under its own license, which applies in addition to this crate’s MIT license.
When redistributing your software,
ensure that you include the required attributions and license texts for any components
that were used by your build configuration and target platform.
The exact set of components used can depend on enabled features, target architecture, and toolchain.

The published crate also includes the following libraries:

BSD-2-Clause:

- [cpuinfo](https://github.com/pytorch/cpuinfo)

BSD-3-Clause:

- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [GoogleTest](https://github.com/google/googletest)
- [cub](https://github.com/NVIDIA/cub)

Apache License 2.0:

- [cpu_features](https://github.com/google/cpu_features)
- [The ruy matrix multiplication library](https://github.com/google/ruy)
- [Thrust: Code at the speed of light](https://github.com/NVIDIA/thrust)

MIT License:

- [cxxopts](https://github.com/jarro2783/cxxopts)
- [spdlog](https://github.com/gabime/spdlog)
