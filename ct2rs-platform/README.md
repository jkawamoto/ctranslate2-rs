# ct2rs-platform

This crate defines platform-specific default feature sets.
The default feature set is based on [these scripts](https://github.com/OpenNMT/CTranslate2/tree/master/python/tools).

## Usage

Add this crate as the `package` argument of the `ct2rs` crate in your `Cargo.toml`:

```toml
[dependencies]
ct2rs = { package = "ct2rs-platform", version = "0.9.11" }
```

See the [ct2rs crate](../README.md) for more information.
