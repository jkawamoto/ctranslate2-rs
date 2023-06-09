// build.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=CTranslate2");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.cpp");

    let ctranslate2 = cmake::build("CTranslate2");
    println!(
        "cargo:rustc-link-search={}",
        ctranslate2.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=ctranslate2");

    if let Ok(path) = env::var("DYLD_LIBRARY_PATH") {
        for s in path.split(';') {
            println!("cargo:rustc-link-search={s}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=iomp5");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args([
            "-x",
            "c++",
            "-std=c++17",
            "-I",
            "CTranslate2/include",
            "-I",
            "CTranslate2/third_party/spdlog/include",
        ])
        .derive_copy(false)
        .generate()
        .expect("Unable to generate bindings");
    bindings
        .write_to_file("src/bindings.rs")
        .expect("Unable to write bindings.rs");

    let mut builder = cc::Build::new();
    builder
        .cpp(true)
        .flag("-std=c++17")
        .file("wrapper.cpp")
        .include("CTranslate2/include")
        .compile("wrapper");
}
