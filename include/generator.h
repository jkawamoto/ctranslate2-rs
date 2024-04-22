// generator.h
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#pragma once

#include "rust/cxx.h"

#include "config.h"

#include <ctranslate2/generator.h>
#include <memory>

struct VecStr;
struct GenerationOptions;
struct GenerationResult;

class Generator {
private:
  std::unique_ptr<ctranslate2::Generator> impl;

public:
  Generator(std::unique_ptr<ctranslate2::Generator> impl) : impl(std::move(impl)) {}

  rust::Vec<GenerationResult> generate_batch(rust::Vec<VecStr> start_tokens,
                                             GenerationOptions options) const;
};

inline std::unique_ptr<Generator> generator(
    rust::Str model_path,
    std::unique_ptr<Config> config
){
    return std::make_unique<Generator>(std::make_unique<ctranslate2::Generator>(
        static_cast<std::string>(model_path),
        config->device,
        config->compute_type,
        std::vector<int>(config->device_indices.begin(), config->device_indices.end()),
        config->tensor_parallel,
        *config->replica_pool_config
    ));
}