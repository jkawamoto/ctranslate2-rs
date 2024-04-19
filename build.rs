// build.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/translator.rs");
    println!("cargo:rerun-if-changed=src/translator.cpp");
    println!("cargo:rerun-if-changed=src/generator.rs");
    println!("cargo:rerun-if-changed=src/generator.cpp");
    println!("cargo:rerun-if-changed=include/convert.h");
    println!("cargo:rerun-if-changed=include/translator.h");
    println!("cargo:rerun-if-changed=include/generator.h");
    println!("cargo:rerun-if-changed=CTranslate2");
    println!("cargo:rerun-if-env-changed=LIBRARY_PATH");
    if let Ok(library_path) = env::var("LIBRARY_PATH") {
        library_path.split(':').for_each(|v| {
            println!("cargo:rustc-link-search={}", v);
        });
    }

    let mut cmake = Config::new("CTranslate2");
    cmake
        .define("BUILD_CLI", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("OPENMP_RUNTIME", "NONE");

    if cfg!(feature = "mkl") {
        cmake.define("WITH_MKL", "ON");
    } else {
        cmake.define("WITH_MKL", "OFF");
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            cmake.define("WITH_ACCELERATE", "OFF");
            cmake.define("WITH_ACCELERATE", "ON");
        } else if cfg!(target_os = "linux") {
            cmake.define("WITH_OPENBLAS", "ON");
            println!("cargo:rustc-link-lib=static=openblas");
        }
    }

    let ctranslate2 = cmake.build();
    println!(
        "cargo:rustc-link-search={}",
        ctranslate2.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=ctranslate2");
    if cfg!(target_arch = "x86_64") {
        println!(
            "cargo:rustc-link-search={}",
            ctranslate2.join("build/third_party/cpu_features").display()
        );
        println!("cargo:rustc-link-lib=static=cpu_features");
    }

    cxx_build::bridges(vec!["src/translator.rs", "src/generator.rs"])
        .file("src/translator.cpp")
        .file("src/generator.cpp")
        .flag_if_supported("-std=c++17")
        .include("CTranslate2/include")
        .compile("ct2rs");
}
