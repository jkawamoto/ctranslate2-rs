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
    println!("cargo:rerun-if-changed=include/convert.h");
    println!("cargo:rerun-if-changed=include/config.h");
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
        .define("WITH_MKL", "OFF")
        .define("OPENMP_RUNTIME", "NONE");

    if cfg!(feature = "mkl") {
        cmake.define("WITH_MKL", "ON");
    } else if cfg!(feature = "openblas") {
        cmake.define("WITH_OPENBLAS", "ON");
        println!("cargo:rustc-link-lib=static=openblas");
    } else if cfg!(feature = "ruy") || cfg!(target_os = "linux") {
        cmake.define("WITH_RUY", "ON");
    } else if cfg!(feature = "accelerate" ) || cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        cmake.define("WITH_ACCELERATE", "ON");
    }

    let ctranslate2 = cmake.build();
    link_libraries(ctranslate2.join("build"));

    cxx_build::bridges(vec![
        "src/types.rs", "src/config.rs", "src/translator.rs", "src/generator.rs"])
        .file("src/translator.cpp")
        .file("src/generator.cpp")
        .std("c++17")
        .include("CTranslate2/include")
        .compile("ct2rs");
}

fn link_libraries<T: AsRef<Path>>(root: T) {
    let mut current_dir = None;
    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file() {
            path.file_name()
                .and_then(|name| name.to_str())
                .filter(|name| name.starts_with("lib") && name.ends_with(".a"))
                .iter()
                .for_each(|name| {
                    let parent = path.parent();
                    if parent != current_dir.as_ref().map(|p: &PathBuf| p.as_path()) {
                        let dir = parent.unwrap();
                        println!("cargo:rustc-link-search={}", dir.display());
                        current_dir = Some(dir.to_path_buf())
                    }
                    println!("cargo:rustc-link-lib=static={}", &name[3..name.len() - 2]);
                });
        }
    }
}
