# ctranslate2-rs
[![Rust library](https://github.com/jkawamoto/ctranslate2-rs/actions/workflows/ci.yaml/badge.svg)](https://github.com/jkawamoto/ctranslate2-rs/actions/workflows/ci.yaml)

This library provides Rust bindings for [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2).
At this time, it has only been tested and confirmed to work on Intel Mac and Linux.

## Compilation
On Linux, [OpenBLAS](https://www.openblas.net/) is required.
Please add the path to the directory containing `libopenblas.a` to `LIBRARY_PATH` environment variable.

## About the Model
The model files need to be converted for CTranslate2.
For instance, the following command will convert `nllb-200-distilled-600M`:

```shell-session
pip install ctranslate2
ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-200-distilled-600M --copy_files tokenizer.json
```

Please do not forget to copy `tokenizer.json`.
For more details, please refer to the [CTranslate2's docs](https://opennmt.net/CTranslate2/guides/transformers.html#nllb).

## Example of Execution
The following example translates English to German using the previously converted model `nllb-200-distilled-600M`.

```rust
use anyhow::Result;
use ctranslate2::config::{Config, Device};
use ctranslate2::Translator;

fn main() -> Result<()> {
    let path = "/path/to/nllb-200-distilled-600M";
    let t = Translator::new(path, Device::CPU, Config::default())?;
    let res = t.translate_batch(
        vec![
            "Hello world!",
            "This library provides Rust bindings for CTranslate2.",
        ],
        vec![vec!["deu_Latn"], vec!["jpn_Jpan"]],
    )?;
    for r in res {
        println!("{}, (score: {:?})", r.0, r.1);
    }


    Ok(())
}
```

### Output
```
Hallo Welt!<unk>, (score: None)
このライブラリでは,CTranslate2 の Rust バインディングが提供されています., (score: None)
```

## License

This application is released under the MIT License. For details, see the [LICENSE](LICENSE) file.
