// whisper.h
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#pragma once

#include <memory>

#include <ctranslate2/models/whisper.h>

#include "rust/cxx.h"

#include "config.h"
#include "model_memory_reader.h"

using ctranslate2::StorageView;

struct VecStr;
struct VecDetectionResult;
struct WhisperOptions;
struct WhisperGenerationResult;

class Whisper {
private:
    std::unique_ptr<ctranslate2::models::Whisper> impl;

public:
    Whisper(std::unique_ptr<ctranslate2::models::Whisper> impl)
        : impl(std::move(impl)) { }

    rust::Vec<WhisperGenerationResult>
    generate(const StorageView& features, const rust::Slice<const VecStr> prompts, const WhisperOptions& options) const;

    rust::Vec<VecDetectionResult>
    detect_language(const StorageView& features) const;

    inline bool is_multilingual() const {
        return impl->is_multilingual();
    }

    inline size_t n_mels() const {
        return impl->n_mels();
    }

    inline size_t num_languages() const {
        return impl->num_languages();
    }

    inline size_t num_queued_batches() const {
        return impl->num_queued_batches();
    }

    inline size_t num_active_batches() const {
        return impl->num_active_batches();
    }

    inline size_t num_replicas() const {
        return impl->num_replicas();
    }
};

inline std::unique_ptr<Whisper> whisper(
    rust::Str model_path,
    std::unique_ptr<Config> config
) {
    return std::make_unique<Whisper>(
        std::make_unique<ctranslate2::models::Whisper>(
            static_cast<std::string>(model_path),
            config->device,
            config->compute_type,
            std::vector<int>(config->device_indices.begin(), config->device_indices.end()),
            config->tensor_parallel,
            *config->replica_pool_config
        )
    );
}

inline std::unique_ptr<Whisper> whisper_from_memory(
    ModelMemoryReader &model_reader,
    std::unique_ptr<Config> config
) {
    ctranslate2::models::ModelLoader model_loader = 
    ctranslate2::models::ModelLoader(model_reader.get_impl());
    model_loader.device = config->device;
    model_loader.compute_type = config->compute_type;
    model_loader.device_indices = std::vector<int>(config->device_indices.begin(), config->device_indices.end());
    model_loader.tensor_parallel = config->tensor_parallel;
    
    return std::make_unique<Whisper>(
        std::make_unique<ctranslate2::models::Whisper>(
            model_loader,
            *config->replica_pool_config
        )
    );
}
