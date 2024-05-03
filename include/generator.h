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
struct GenerationStepResult;

class Generator {
private:
    std::unique_ptr<ctranslate2::Generator> impl;

public:
    Generator(std::unique_ptr<ctranslate2::Generator> impl)
        : impl(std::move(impl)) { }

    rust::Vec<GenerationResult> generate_batch(
        const rust::Vec<VecStr>& start_tokens,
        const GenerationOptions& options,
        bool has_callback,
        rust::Fn<bool(GenerationStepResult)> callback
    ) const;

    size_t num_queued_batches() const {
        return this->impl->num_queued_batches();
    }

    size_t num_active_batches() const {
        return this->impl->num_active_batches();
    }

    size_t num_replicas() const {
        return this->impl->num_replicas();
    }
};

inline std::unique_ptr<Generator> generator(
    rust::Str model_path,
    std::unique_ptr<Config> config
) {
    return std::make_unique<Generator>(std::make_unique<ctranslate2::Generator>(
        static_cast<std::string>(model_path),
        config->device,
        config->compute_type,
        std::vector<int>(config->device_indices.begin(), config->device_indices.end()),
        config->tensor_parallel,
        *config->replica_pool_config
    ));
}
