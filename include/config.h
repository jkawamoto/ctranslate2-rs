// config.h
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#pragma once

#include "rust/cxx.h"

#include <memory>

#include <ctranslate2/replica_pool.h>

using ctranslate2::Device;
using ctranslate2::ComputeType;
using ctranslate2::ReplicaPoolConfig;

inline std::unique_ptr<ReplicaPoolConfig> replica_pool_config(
    size_t num_threads_per_replica,
    long max_queued_batches,
    int cpu_core_offset
){
    return std::make_unique<ReplicaPoolConfig>(ReplicaPoolConfig{
        num_threads_per_replica,
        max_queued_batches,
        cpu_core_offset,
    });
}

struct Config {
    Device device;
    ComputeType compute_type;
    rust::Slice<const int> device_indices;
    bool tensor_parallel;
    std::unique_ptr<ReplicaPoolConfig> replica_pool_config;
};

inline std::unique_ptr<Config> config(
    Device device,
    ComputeType compute_type,
    rust::Slice<const int> device_indices,
    bool tensor_parallel,
    std::unique_ptr<ReplicaPoolConfig> replica_pool_config
){
    return std::make_unique<Config>(Config{
        device,
        compute_type,
        device_indices,
        tensor_parallel,
        std::move(replica_pool_config),
    });
}
