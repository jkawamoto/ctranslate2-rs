// build.rs
//
// Copyright (c) 2023-2025 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};

use cmake::Config;
use flate2::Compression;
use tar::Builder;
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

fn download(url: &str, out: &Path) -> u16 {
    let response = ureq::get(url).call().expect("Failed to send request");
    let status = response.status().as_u16();

    if response.status() != 200 {
        return status;
    }

    let mut body = response.into_body();
    let reader = body.as_reader();

    let mut gz = flate2::read::GzDecoder::new(reader);

    let mut archive = tar::Archive::new(&mut gz);
    archive.unpack(&out).expect("Failed to extract archive");
    status
}

fn build_dnnl() {
    let out_dir = if let Ok(dir) = env::var("CARGO_TARGET_DIR") {
        PathBuf::from(dir)
    } else {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        PathBuf::from(manifest_dir).join("target")
    }
    .join("dnnl");
    let dnnl_version = "3.1.1";
    let dnnl_archive = format!("v{}.tar.gz", dnnl_version);
    let dnnl_url = format!(
        "https://github.com/oneapi-src/oneDNN/archive/refs/tags/{}",
        dnnl_archive
    );

    let source_dir = out_dir.join(format!("oneDNN-{}", dnnl_version));

    if !source_dir.exists() {
        if download(&dnnl_url, &out_dir) != 200 {
            panic!("Failed to download oneDNN");
        }
    }

    let dst = Config::new(source_dir)
        .define("ONEDNN_LIBRARY_TYPE", "STATIC")
        .define("ONEDNN_BUILD_EXAMPLES", "OFF")
        .define("ONEDNN_BUILD_TESTS", "OFF")
        .define("ONEDNN_ENABLE_WORKLOAD", "INFERENCE")
        .define("ONEDNN_ENABLE_PRIMITIVE", "CONVOLUTION;REORDER")
        .define("ONEDNN_BUILD_GRAPH", "OFF")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=dnnl");
    println!("cargo:include={}/include", dst.display());
}

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
enum Os {
    Win,
    Mac,
    Linux,
    Unknown,
}

fn load_vendor(os: Os, aarch64: bool) -> Option<PathBuf> {
    let url = format!(
        "https://github.com/frederik-uni/ctranslate2-rs/releases/download/ctranslate2-05.08.2025/{}-{}.tar.gz",
        match os {
            Os::Win => "windows",
            Os::Mac => "macos",
            Os::Linux => "linux",
            Os::Unknown => return None,
        },
        match aarch64 {
            true => "arm64",
            false => "x86_64",
        }
    );
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_dir = out_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("ctranslate2-vendor");
    let dyn_dir = out_dir.join("dyn");

    println!("cargo:rustc-link-search=native={}", dyn_dir.display());
    if download(&url, &out_dir) != 200 {
        return None;
    }
    match (os, aarch64) {
        (Os::Win, false) => {
            println!("cargo:rustc-link-lib=iomp5md");
            Some(out_dir.to_path_buf())
        }
        (Os::Mac, true) => {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            Some(out_dir.to_path_buf())
        }
        (Os::Linux, true) => {
            println!("cargo:rustc-link-lib=gomp");
            Some(out_dir.to_path_buf())
        }
        (Os::Mac, false) => {
            println!("cargo:rustc-link-lib=iomp5");
            Some(out_dir.to_path_buf())
        }
        (Os::Linux, false) => {
            println!("cargo:rustc-link-lib=cudnn");
            println!("cargo:rustc-link-lib=gomp");
            Some(out_dir.to_path_buf())
        }
        _ => None,
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/sys");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=CTranslate2");
    let mut found = None;
    let aarch64 = cfg!(target_arch = "aarch64");
    let os = if cfg!(target_os = "windows") {
        Os::Win
    } else if cfg!(target_os = "macos") {
        Os::Mac
    } else if cfg!(target_os = "linux") {
        Os::Linux
    } else {
        Os::Unknown
    };
    if cfg!(feature = "vendored") {
        found = load_vendor(os, aarch64);
    }
    let lib_path = if found.is_none() {
        add_search_paths("LIBRARY_PATH");
        add_search_paths("CMAKE_LIBRARY_PATH");

        let mut cmake = Config::new("CTranslate2");
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
        let mut openmp_intel = cfg!(feature = "openmp-runtime-intel");
        let mut msse4_1 = cfg!(feature = "msse4_1");
        if !openmp_intel && !openmp_comp && dnnl {
            if os == Os::Linux {
                openmp_comp = true;
            }
        }
        let flash_attention = cfg!(feature = "flash-attention");
        if cfg!(feature = "os-defaults") {
            match (os, aarch64) {
                (Os::Win, false) => {
                    openmp_intel = false;
                    openmp_comp = false;
                    dnnl = true;
                    cuda = true;
                    cudnn = true;
                    cuda_dynamic_loading = true;
                    mkl = true;
                    ruy = false;
                    accelarate = false;
                    openblas = false;
                }
                (Os::Mac, true) => {
                    openmp_intel = false;
                    openmp_comp = false;
                    dnnl = false;
                    mkl = false;
                    cuda = false;
                    cudnn = false;
                    cuda_dynamic_loading = false;
                    ruy = true;
                    accelarate = true;
                    openblas = false;
                }
                (Os::Mac, false) => {
                    openmp_intel = true;
                    openmp_comp = false;
                    dnnl = true;
                    mkl = true;
                    cuda = false;
                    cudnn = false;
                    cuda_dynamic_loading = false;
                    ruy = false;
                    accelarate = false;
                    openblas = false;
                }
                (Os::Linux, true) => {
                    openmp_intel = false;
                    openmp_comp = true;
                    dnnl = false;
                    mkl = false;
                    cuda = false;
                    cudnn = false;
                    cuda_dynamic_loading = false;
                    ruy = true;
                    accelarate = false;
                    openblas = true;
                }
                (Os::Linux, false) => {
                    openmp_intel = false;
                    openmp_comp = true;
                    dnnl = true;
                    mkl = true;
                    cuda = true;
                    cudnn = true;
                    cuda_dynamic_loading = true;
                    ruy = false;
                    accelarate = false;
                    openblas = false;

                    tensor_parallel = true;
                    msse4_1 = true;
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
        if os == Os::Win {
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
            cmake.define("CUDA_ARCH_LIST", "Common");
            if cfg!(feature = "cuda-small-binary") {
                cmake.define("CUDA_NVCC_FLAGS", "-Xfatbin=-compress-all");
            }

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
                if os == Os::Win {
                    println!("cargo:rustc-link-lib=static=cublas");
                    println!("cargo:rustc-link-lib=static=cublasLt");
                } else {
                    println!("cargo:rustc-link-lib=static=cublas_static");
                    println!("cargo:rustc-link-lib=static=cublasLt_static");
                    println!("cargo:rustc-link-lib=static=culibos");
                }
            }
        }
        if os == Os::Mac && aarch64 {
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
        if msse4_1 {
            cmake.define("CMAKE_CXX_FLAGS", "-msse4.1");
        }
        if dnnl {
            build_dnnl();
            cmake.define("WITH_DNNL", "ON");
        }
        if openmp_comp {
            println!("cargo:rustc-link-lib=gomp");
            cmake.define("OPENMP_RUNTIME", "COMP");
        } else if openmp_intel {
            if os == Os::Win {
                println!("cargo:rustc-link-lib=dylib=iomp5md");
            } else {
                println!("cargo:rustc-link-lib=iomp5");
            }
            cmake.define("OPENMP_RUNTIME", "INTEL");
        }
        if flash_attention {
            cmake.define("WITH_FLASH_ATTN", "ON");
        }

        let ctranslate2 = cmake.build();
        ctranslate2.join("build")
    } else {
        found.unwrap()
    };

    let modules = link_libraries(&lib_path);
    if cfg!(feature = "_vendor") {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let out_dir = out_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap();

        let tar_gz = File::create(out_dir.join("vendored.tar.gz")).unwrap();
        let enc = flate2::write::GzEncoder::new(tar_gz, Compression::default());
        let mut tar = Builder::new(enc);

        tar.append_dir_all("include", lib_path.join("../include"))
            .unwrap();

        for module in modules {
            let mut file = File::open(&module).unwrap();
            let name = module.file_name().unwrap().to_str().unwrap();
            tar.append_file(format!("lib/{}", name), &mut file).unwrap();
        }

        tar.finish().unwrap();
    }

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

fn link_libraries<T: AsRef<Path>>(root: T) -> Vec<PathBuf> {
    let mut current_dir = None;
    let mut libs = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file() {
            if let Some(file_name) = path
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| is_library(name))
            {
                let parent = path.parent().unwrap();
                if Some(parent) != current_dir.as_deref() {
                    println!("cargo:rustc-link-search={}", parent.display());
                    current_dir = Some(parent.to_path_buf());
                }
                libs.push(path.to_path_buf());

                let lib_name = library_name(file_name);
                println!("cargo:rustc-link-lib=static={}", lib_name);
            }
        }
    }
    libs
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
