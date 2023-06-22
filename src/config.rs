// config.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

/// Device to use.
#[derive(Debug)]
pub enum Device {
    CPU,
    CUDA,
}

/// Model computation type or a dictionary mapping a device name to the computation type.
#[derive(Debug, Default)]
pub enum ComputeType {
    #[default]
    Default,
    Auto,
    Float32,
    Int8,
    Int8Float16,
    Int16,
    Float16,
}

#[derive(Debug)]
pub struct Config {
    /// Model computation type or a dictionary mapping a device name to the computation type.
    pub compute_type: ComputeType,
    /// Device IDs where to place this generator on.
    pub device_indices: Vec<i32>,
    pub num_threads_per_replica: usize,
    pub max_queued_batches: i64,
    pub cpu_core_offset: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            compute_type: Default::default(),
            device_indices: vec![0],
            num_threads_per_replica: 0,
            max_queued_batches: 0,
            cpu_core_offset: -1,
        }
    }
}

#[derive(Debug, Default)]
pub enum BatchType {
    #[default]
    Examples,
    Tokens,
}
