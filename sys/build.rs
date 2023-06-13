// build.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use cmake::Config;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=CTranslate2");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.cpp");

    let openblas = std::env::var("OPENBLAS_LIBRARY").expect("OPENBLAS_LIBRARY is not set");
    println!("cargo:rustc-link-search={openblas}");
    println!("cargo:rustc-link-lib=static=openblas");

    let libomp = std::env::var("OMP_LIBRARY").expect("OMP_LIBRARY is not set");
    println!("cargo:rustc-link-search={libomp}");
    println!("cargo:rustc-link-lib=static=omp");

    let ctranslate2 = Config::new("CTranslate2")
        .define("BUILD_CLI", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WITH_MKL", "OFF")
        .define("WITH_OPENBLAS", "ON")
        .define(
            "CMAKE_PREFIX_PATH",
            Path::new(&openblas)
                .parent()
                .expect("OPENBLAS_LIBRARY has a wrong value"),
        )
        .build();
    println!(
        "cargo:rustc-link-search={}",
        ctranslate2.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=ctranslate2");
    println!(
        "cargo:rustc-link-search={}",
        ctranslate2.join("build/third_party/cpu_features").display()
    );
    println!("cargo:rustc-link-lib=static=cpu_features");

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
