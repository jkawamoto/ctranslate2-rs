# ctranslate2-rs

[![Latest version](https://img.shields.io/crates/v/ct2rs.svg)](https://crates.io/crates/ct2rs)
[![Build](https://github.com/jkawamoto/ctranslate2-rs/actions/workflows/build.yaml/badge.svg)](https://github.com/jkawamoto/ctranslate2-rs/actions/workflows/build.yaml)

This library provides Rust bindings for [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2).
At this time, it has only been tested and confirmed to work on macOS and Linux.
Windows support is available experimentally,
but it has not been thoroughly tested and may have limitations or require additional configuration.

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

Please see the respective
[examples](https://github.com/jkawamoto/ctranslate2-rs/tree/main/examples)
for each model.

## Stream API

This crate also offers a streaming API that utilizes callback closures.
Please refer to the [example code](https://github.com/jkawamoto/ctranslate2-rs/blob/main/examples/stream.rs)
for more information.

## Compilation

If you plan to use GPU acceleration, CUDA and cuDNN are available.
Please enable the `cuda` or `cudnn` feature and set the `CUDA_TOOLKIT_ROOT_DIR` environment variable appropriately.

Several backends are available for use:
[OpenBLAS](https://www.openblas.net/),
[Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html),
[Ruy](https://github.com/google/ruy),
and [Apple Accelerate](https://developer.apple.com/documentation/accelerate).

- **OpenBLAS**: To use OpenBLAS, enable the `openblas` feature and add the path to the directory
  containing `libopenblas.a` to the `LIBRARY_PATH` environment variable.
- **Intel MKL**: To use Intel MKL, enable the `mkl` feature and set the path to the Intel libraries in the `MKLROOT`
  environment variable (default is `/opt/intel`).
- **Ruy**: To use Ruy, enable the `ruy` feature.
- **Apple Accelerate**: Available only on macOS, enable the `accelerate` feature to use Apple Accelerate.

The installation of CMake is required to compile the library.

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
