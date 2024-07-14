// build.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::env;
use std::path::{Path, PathBuf};

use cmake::Config;
use walkdir::WalkDir;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/types.rs");
    println!("cargo:rerun-if-changed=src/config.rs");
    println!("cargo:rerun-if-changed=src/translator.rs");
    println!("cargo:rerun-if-changed=src/translator.cpp");
    println!("cargo:rerun-if-changed=src/generator.rs");
    println!("cargo:rerun-if-changed=src/generator.cpp");
    println!("cargo:rerun-if-changed=src/storage_view.rs");
    println!("cargo:rerun-if-changed=src/whisper.rs");
    println!("cargo:rerun-if-changed=src/whisper.cpp");
    println!("cargo:rerun-if-changed=include/types.h");
    println!("cargo:rerun-if-changed=include/config.h");
    println!("cargo:rerun-if-changed=include/translator.h");
    println!("cargo:rerun-if-changed=include/generator.h");
    println!("cargo:rerun-if-changed=include/storage_view.h");
    println!("cargo:rerun-if-changed=include/whisper.h");
    println!("cargo:rerun-if-changed=CTranslate2");
    println!("cargo:rerun-if-env-changed=LIBRARY_PATH");
    if let Ok(library_path) = env::var("LIBRARY_PATH") {
        library_path
            .split(':')
            .filter(|v| !v.is_empty())
            .for_each(|v| {
                println!("cargo:rustc-link-search={}", v);
            });
    }

    let mut cmake = Config::new("CTranslate2");
    cmake
        .define("BUILD_CLI", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WITH_MKL", "OFF")
        .define("OPENMP_RUNTIME", "NONE");
    if cfg!(target_os = "windows") {
        let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
        if !rustflags.contains("target-feature=+crt-static") {
            println!("cargo:warning=For Windows compilation, set `RUSTFLAGS=-C target-feature=+crt-static`.");
        }

        println!("cargo::rustc-link-arg=/FORCE:MULTIPLE");
        cmake.profile("Release").cxxflag("/EHsc").static_crt(true);
    }

    if cfg!(feature = "cuda") {
        let cuda = cuda_root().expect("CUDA_TOOLKIT_ROOT_DIR is not specified");
        cmake.define("WITH_CUDA", "ON");
        cmake.define("CUDA_TOOLKIT_ROOT_DIR", &cuda);
        link_libraries(cuda.join("lib"));
        link_libraries(cuda.join("lib64"));
        if cfg!(feature = "cudnn") {
            cmake.define("WITH_CUDNN", "ON");
        }
    }
    if cfg!(feature = "mkl") {
        cmake.define("WITH_MKL", "ON");
    }
    if cfg!(feature = "openblas") {
        cmake.define("WITH_OPENBLAS", "ON");
        println!("cargo:rustc-link-lib=static=openblas");
    }
    if cfg!(feature = "ruy") {
        cmake.define("WITH_RUY", "ON");
    }
    if cfg!(all(feature = "accelerate", target_os = "macos")) {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        cmake.define("WITH_ACCELERATE", "ON");
    }

    let ctranslate2 = cmake.build();
    link_libraries(ctranslate2.join("build"));

    cxx_build::bridges(vec![
        "src/types.rs",
        "src/config.rs",
        "src/translator.rs",
        "src/generator.rs",
        "src/storage_view.rs",
        "src/whisper.rs",
    ])
    .file("src/translator.cpp")
    .file("src/generator.cpp")
    .file("src/whisper.cpp")
    .include("CTranslate2/include")
    .std("c++17")
    .static_crt(cfg!(target_os = "windows"))
    .flag_if_supported("/EHsc")
    .compile("ct2rs");
}

#[cfg(not(target_os = "windows"))]
fn is_library(name: &&str) -> bool {
    name.starts_with("lib") && name.ends_with(".a")
}

#[cfg(not(target_os = "windows"))]
fn library_name(name: &str) -> &str {
    &name[3..name.len() - 2]
}

#[cfg(target_os = "windows")]
fn is_library(name: &&str) -> bool {
    name.ends_with(".lib")
}

#[cfg(target_os = "windows")]
fn library_name(name: &str) -> &str {
    &name[0..name.len() - 4]
}

fn link_libraries<T: AsRef<Path>>(root: T) {
    let mut current_dir = None;
    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file() {
            path.file_name()
                .and_then(|name| name.to_str())
                .filter(is_library)
                .iter()
                .for_each(|name| {
                    let parent = path.parent();
                    if parent != current_dir.as_deref() {
                        let dir = parent.unwrap();
                        println!("cargo:rustc-link-search={}", dir.display());
                        current_dir = Some(dir.to_path_buf())
                    }
                    println!("cargo:rustc-link-lib=static={}", library_name(name));
                });
        }
    }
}

// The function below was derived and modified from the `cudarc` crate.
// Original source: https://github.com/coreylowman/cudarc/blob/main/build.rs
//
// Copyright (c) 2024 Corey Lowman
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
fn cuda_root() -> Option<PathBuf> {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let roots = roots.into_iter().map(Into::into);
    env_vars
        .chain(roots)
        .map(Into::<PathBuf>::into)
        .find(|path| path.join("include").join("cuda.h").is_file())
}
