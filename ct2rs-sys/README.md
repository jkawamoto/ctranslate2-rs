# ct2rs-sys

Low-level Rust bindings for [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2).
This crate owns the CTranslate2 build and link configuration used by `ct2rs`.

```toml
[dependencies]
ct2rs-sys = "0.10.3"
```

The bindings are exposed at the crate root, for example `ct2rs_sys::Translator` and
`ct2rs_sys::Config`. Backend and system-library features are configured on this crate.
See the [main project README](../README.md) for feature descriptions and platform notes.
