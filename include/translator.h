// translator.h
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#pragma once

#include "rust/cxx.h"

#include "config.h"

#include <ctranslate2/translator.h>
#include <memory>

struct VecStr;
struct TranslationOptions;
struct TranslationResult;

class Translator {
private:
    std::unique_ptr<ctranslate2::Translator> impl;

public:
    Translator(std::unique_ptr<ctranslate2::Translator> impl)
        : impl(std::move(impl)) { }

    rust::Vec<TranslationResult>
    translate_batch(rust::Vec<VecStr> source, rust::Vec<VecStr> target_prefix, TranslationOptions options) const;
};

inline std::unique_ptr<Translator> translator(
    rust::Str model_path,
    std::unique_ptr<Config> config
) {
    return std::make_unique<Translator>(std::make_unique<ctranslate2::Translator>(
        static_cast<std::string>(model_path),
        config->device,
        config->compute_type,
        std::vector<int>(config->device_indices.begin(), config->device_indices.end()),
        config->tensor_parallel,
        *config->replica_pool_config
    ));
}
