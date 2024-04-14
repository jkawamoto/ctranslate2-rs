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

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let mut cmake = Config::new("CTranslate2");
    cmake
        .define("BUILD_CLI", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WITH_MKL", "OFF")
        .define("OPENMP_RUNTIME", "NONE");

    match target_os.as_str() {
        "macos" => {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            cmake.define("WITH_ACCELERATE", "ON");
        }
        "linux" => {
            link_static_library("openblas");
            cmake.define("WITH_OPENBLAS", "ON");
        }
        _ => {}
    }

    let ctranslate2 = cmake.build();
    println!(
        "cargo:rustc-link-search={}",
        ctranslate2.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=ctranslate2");

    match target_arch.as_str() {
        "x86_64" => {
            println!(
                "cargo:rustc-link-search={}",
                ctranslate2.join("build/third_party/cpu_features").display()
            );
            println!("cargo:rustc-link-lib=static=cpu_features");
        }
        _ => {}
    }

    cxx_build::bridges(vec!["src/translator.rs", "src/generator.rs"])
        .file("src/translator.cpp")
        .file("src/generator.cpp")
        .flag_if_supported("-std=c++17")
        .include("CTranslate2/include")
        .compile("ct2rs");
}

fn link_static_library<T: std::fmt::Display>(name: T) -> bool {
    let libname = format!("lib{name}.a");
    if find_system_library(&libname) {
        println!("cargo:rustc-link-lib=static={name}");
        return true;
    } else if let Some(p) = find_library(&libname) {
        println!("cargo:rustc-link-search={}", p.display());
        println!("cargo:rustc-link-lib=static={name}");
        return true;
    }
    false
}

fn find_library<T: AsRef<Path>>(name: T) -> Option<PathBuf> {
    if let Ok(p) = env::var("LIBRARY_PATH") {
        if let Some(path) = p
            .split(':')
            .map(Path::new)
            .find(|p| p.join(name.as_ref()).exists())
        {
            return Some(PathBuf::from(path));
        }
    }
    None
}

fn find_system_library<T: AsRef<Path>>(name: T) -> bool {
    let default_paths = vec![".", "/lib", "/usr/lib", "/usr/local/lib"];
    default_paths
        .into_iter()
        .map(|s| Path::new(s).join(name.as_ref()))
        .any(|p| p.exists())
}
