// build.rs
//
// Copyright (c) 2023-2025 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::env;
use std::path::Path;

use cmake::Config;

fn add_search_paths(key: &str) {
    println!("cargo:rerun-if-env-changed={}", key);
    if let Ok(library_path) = env::var(key) {
        env::split_paths(&library_path)
            .filter(|v| v.exists())
            .for_each(|v| {
                println!("cargo:rustc-link-search={}", v.display());
            });
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/sys");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=CTranslate2");
    add_search_paths("LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=CMAKE_INCLUDE_PATH");
    add_search_paths("CMAKE_LIBRARY_PATH");

    let mut cmake = Config::new("CTranslate2");
    match env::var("CMAKE_PARALLEL") {
        Ok(job_n) => {
            cmake.build_arg("-j").build_arg(job_n);
        }
        Err(env::VarError::NotPresent) => (),
        Err(err) => panic!("CMAKE_PARALLEL format error: {:?}", err),
    }
    let os = if cfg!(target_os = "windows") {
        ctranslate2_src_build_support::Os::Win
    } else if cfg!(target_os = "macos") {
        ctranslate2_src_build_support::Os::Mac
    } else if cfg!(target_os = "linux") {
        ctranslate2_src_build_support::Os::Linux
    } else {
        ctranslate2_src_build_support::Os::Unknown
    };

    let aarch64 = cfg!(target_arch = "aarch64");

    let cuda = cfg!(feature = "cuda");
    let cudnn = cfg!(feature = "cudnn");
    let cuda_dynamic_loading = cfg!(feature = "cuda-dynamic-loading");
    let cuda_small_binary = cfg!(feature = "cuda-small-binary");
    let mkl = cfg!(feature = "mkl");
    let openblas = cfg!(feature = "openblas");
    let ruy = cfg!(feature = "ruy");
    let accelarate = cfg!(feature = "accelerate");
    let tensor_parallel = cfg!(feature = "tensor-parallel");
    let dnnl = cfg!(feature = "dnnl");
    let mut openmp_comp: bool = cfg!(feature = "openmp-runtime-comp");
    let openmp_intel = cfg!(feature = "openmp-runtime-intel");
    let msse4_1 = cfg!(feature = "msse4_1");
    let vendor = cfg!(feature = "vendor");
    if !openmp_intel && !openmp_comp && dnnl {
        if os == ctranslate2_src_build_support::Os::Linux {
            openmp_comp = true;
        }
    }
    let flash_attention = cfg!(feature = "flash-attention");
    ctranslate2_src_build_support::main((
        os,
        aarch64,
        cuda,
        cudnn,
        cuda_dynamic_loading,
        mkl,
        openblas,
        ruy,
        accelarate,
        tensor_parallel,
        dnnl,
        openmp_comp,
        openmp_intel,
        msse4_1,
        flash_attention,
        cuda_small_binary,
        false,
        vendor,
        false,
        false,
        Some(Path::new("CTranslate2")),
    ));

    cxx_build::bridges([
        "src/sys/types.rs",
        "src/sys/config.rs",
        "src/sys/model_memory_reader.rs",
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
    .flag_if_supported("/EHsc")
    .compile("ct2rs");
}
