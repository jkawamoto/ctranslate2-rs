# ctranslate2-rs

[![Latest version](https://img.shields.io/crates/v/ct2rs.svg)](https://crates.io/crates/ct2rs)
[![Rust library](https://github.com/jkawamoto/ctranslate2-rs/actions/workflows/ci.yaml/badge.svg)](https://github.com/jkawamoto/ctranslate2-rs/actions/workflows/ci.yaml)

This library provides Rust bindings for [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2).
At this time, it has only been tested and confirmed to work on macOS and Linux.
Windows support is available experimentally, 
but it has not been thoroughly tested and may have limitations or require additional configuration.

# Compilation

Several libraries are available for use:
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

If no feature is specified:

- On macOS, Apple Accelerate will be used by default.
- On Linux and Windows, Ruy will be used by default.

The installation of CMake is required to compile the library.

Additional notes for Windows:
it is necessary to add `RUSTFLAGS=-C target-feature=+crt-static` to the environment variables for compilation.

## Model Conversion for CTranslate2

To use model files with CTranslate2, they must first be converted.
Below is an example of how to convert the `nllb-200-distilled-600M` model:

```shell-session
pip install ctranslate2
ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-200-distilled-600M --copy_files tokenizer.json
```

For more details, please refer to
the [CTranslate2's docs](https://opennmt.net/CTranslate2/guides/transformers.html#nllb).

## Example of text translation

The following example translates English to German and Japanese using the previously converted
model `nllb-200-distilled-600M`.

```rust
use anyhow::Result;
use ct2rs::config::{Config, Device};
use ct2rs::{TranslationOptions, Translator};

fn main() -> Result<()> {
    let path = "/path/to/nllb-200-distilled-600M";
    let t = Translator::new(path, Config::default())?;
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
```

### Output

```
Hallo Welt!<unk>, (score: Some(-0.5597002))
このライブラリでは,CTranslate2 の Rust バインディングが提供されています., (score: Some(-0.56321025))
```

## Example of text generation

```rust
fn main() -> Result<()> {
    let path = "/path/to/model";
    let g = Generator::new(path, Config::default())?;
    let res = g.generate_batch(
        vec!["prompt"],
        &GenerationOptions::default(),
    )?;
    for r in res {
        println!("{:?}", r.0);
    }

    Ok(())
}
```

## License

This application is released under the MIT License. For details, see the [LICENSE](LICENSE) file.
