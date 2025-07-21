// build.rs
//
// Copyright (c) 2023-2025 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::env;
use std::path::{Path, PathBuf};

use cmake::Config;
use walkdir::WalkDir;

#[cfg(not(target_os = "windows"))]
const PATH_SEPARATOR: char = ':';

#[cfg(target_os = "windows")]
const PATH_SEPARATOR: char = ';';

fn add_search_paths(key: &str) {
    println!("cargo:rerun-if-env-changed={}", key);
    if let Ok(library_path) = env::var(key) {
        library_path
            .split(PATH_SEPARATOR)
            .filter(|v| !v.is_empty())
            .for_each(|v| {
                println!("cargo:rustc-link-search={}", v);
            });
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/sys");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=CTranslate2");
    add_search_paths("LIBRARY_PATH");
    add_search_paths("CMAKE_LIBRARY_PATH");

    let mut cmake = Config::new("CTranslate2");
    let windows = cfg!(target_os = "windows");
    let macos = cfg!(target_os = "macos");
    let aarch64 = cfg!(target_arch = "aarch64");

    let mut cuda = cfg!(feature = "cuda");
    let mut cudnn = cfg!(feature = "cudnn");
    let mut cuda_dynamic_loading = cfg!(feature = "cuda-dynamic-loading");
    let mut mkl = cfg!(feature = "mkl");
    let mut openblas = cfg!(feature = "openblas");
    let mut ruy = cfg!(feature = "ruy");
    let mut accelarate = cfg!(feature = "accelerate");
    let mut tensor_parallel = cfg!(feature = "tensor-parallel");
    let mut dnnl = cfg!(feature = "dnnl");
    let mut openmp_comp = cfg!(feature = "openmp-runtime-comp");
    let flash_attention = cfg!(feature = "flash-attention");
    if cfg!(feature = "os-defaults") {
        match (windows, macos, aarch64) {
            (true, false, false) => {
                dnnl = true;
                cuda = true;
                cudnn = true;
                cuda_dynamic_loading = true;
                mkl = true;
            }
            (false, true, true) => {
                accelarate = true;
                ruy = true;
            }
            (false, true, false) => {
                dnnl = true;
                mkl = true;
            }
            (false, false, true) => {
                openmp_comp = true;
                openblas = true;
                ruy = true;
            }
            (false, false, false) => {
                dnnl = true;
                openmp_comp = true;
                cudnn = true;
                cuda = true;
                cuda_dynamic_loading = true;
                mkl = true;
                tensor_parallel = true;
            }
            _ => {}
        }
    }
    cmake
        .define("BUILD_CLI", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WITH_MKL", "OFF")
        .define("OPENMP_RUNTIME", "NONE")
        .define("CMAKE_POLICY_VERSION_MINIMUM", "3.5");
    if windows {
        let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
        if !rustflags.contains("target-feature=+crt-static") {
            println!("cargo:warning=For Windows compilation, set `RUSTFLAGS=-C target-feature=+crt-static`.");
        }

        println!("cargo::rustc-link-arg=/FORCE:MULTIPLE");
        cmake.profile("Release").cxxflag("/EHsc").static_crt(true);
    }

    if cuda {
        let cuda = cuda_root().expect("CUDA_TOOLKIT_ROOT_DIR is not specified");
        cmake.define("WITH_CUDA", "ON");
        cmake.define("CUDA_TOOLKIT_ROOT_DIR", &cuda);
        println!("cargo:rustc-link-search={}", cuda.join("lib").display());
        println!("cargo:rustc-link-search={}", cuda.join("lib64").display());
        println!("cargo:rustc-link-search={}", cuda.join("lib/x64").display());
        println!("cargo:rustc-link-lib=static=cudart_static");
        if cudnn {
            cmake.define("WITH_CUDNN", "ON");
            println!("cargo:rustc-link-lib=cudnn");
        }
        if cuda_dynamic_loading {
            cmake.define("CUDA_DYNAMIC_LOADING", "ON");
        } else {
            if windows {
                println!("cargo:rustc-link-lib=static=cublas");
                println!("cargo:rustc-link-lib=static=cublasLt");
            } else {
                println!("cargo:rustc-link-lib=static=cublas_static");
                println!("cargo:rustc-link-lib=static=cublasLt_static");
                println!("cargo:rustc-link-lib=static=culibos");
            }
        }
    }
    if macos && aarch64 {
        cmake.define("CMAKE_OSX_ARCHITECTURES", "arm64");
    }

    if mkl {
        cmake.define("WITH_MKL", "ON");
    }
    if openblas {
        println!("cargo:rustc-link-lib=static=openblas");
        cmake.define("WITH_OPENBLAS", "ON");
    }
    if ruy {
        cmake.define("WITH_RUY", "ON");
    }
    if accelarate {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        cmake.define("WITH_ACCELERATE", "ON");
    }
    if tensor_parallel {
        cmake.define("WITH_TENSOR_PARALLEL", "ON");
    }
    if dnnl {
        println!("cargo:rustc-link-lib=static=dnnl");
        cmake.define("WITH_DNNL", "ON");
    }
    if openmp_comp {
        println!("cargo:rustc-link-lib=gomp");
        cmake.define("OPENMP_RUNTIME", "COMP");
    }
    if flash_attention {
        cmake.define("WITH_FLASH_ATTN", "ON");
    }

    let ctranslate2 = cmake.build();
    link_libraries(ctranslate2.join("build"));

    cxx_build::bridges([
        "src/sys/types.rs",
        "src/sys/config.rs",
        "src/sys/scoring.rs",
        "src/sys/translator.rs",
        "src/sys/generator.rs",
        "src/sys/storage_view.rs",
        "src/sys/whisper.rs",
    ])
    .file("src/sys/translator.cpp")
    .file("src/sys/generator.cpp")
    .file("src/sys/whisper.cpp")
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
