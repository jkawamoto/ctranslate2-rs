// config.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! Configs and associated enums.

/// Represents the computing device to be used.
#[derive(Debug)]
pub enum Device {
    CPU,
    CUDA,
}

/// Model computation type.
///
/// The default setting for this enum is `Default`.
///
/// See also [Quantization](https://opennmt.net/CTranslate2/quantization.html#quantize-on-model-loading).
///
/// # Examples
///
/// Example of creating a default `ComputeType`:
///
/// ```
/// use ct2rs::config::ComputeType;
///
/// let compute_type = ComputeType::default();
/// assert_eq!(compute_type, ComputeType::Default);
/// ```
///
#[derive(PartialEq, Eq, Default, Debug)]
pub enum ComputeType {
    /// Keep the same quantization that was used during model conversion.
    #[default]
    Default,
    /// Use the fastest computation type that is supported on this system and device.
    Auto,
    Float32,
    Int8,
    Int8Float32,
    Int8Float16,
    Int8BFloat16,
    Int16,
    Float16,
    BFloat16,
}

/// The `Config` structure holds the configuration settings for CTranslator2.
///
/// # Examples
///
/// Example of creating a default `Config`:
///
/// ```
/// use ct2rs::config::{ComputeType, Config};
///
/// let config = Config::default();
/// assert_eq!(config.compute_type, ComputeType::default());
/// assert_eq!(config.device_indices, vec![0]);
/// assert_eq!(config.tensor_parallel, false);
/// assert_eq!(config.num_threads_per_replica, 0);
/// assert_eq!(config.max_queued_batches, 0);
/// assert_eq!(config.cpu_core_offset, -1);
/// ```
#[derive(PartialEq, Eq, Debug)]
pub struct Config {
    /// Model computation type.
    pub compute_type: ComputeType,
    /// Device IDs where to place this generator on.
    pub device_indices: Vec<i32>,
    /// Run model with tensor parallel mode.
    pub tensor_parallel: bool,
    pub num_threads_per_replica: usize,
    /// Maximum numbers of batches in the queue (-1 for unlimited, 0 for an automatic value).
    /// When the queue is full, future requests will block until a free slot is available.
    pub max_queued_batches: i64,
    pub cpu_core_offset: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            compute_type: Default::default(),
            device_indices: vec![0],
            tensor_parallel: false,
            num_threads_per_replica: 0,
            max_queued_batches: 0,
            cpu_core_offset: -1,
        }
    }
}

/// Specifies how the `max_batch_size` should be calculated,
/// whether by the number of "examples" or "tokens".
///
/// The default setting for this enum is `Examples`,
/// meaning that the batch size will typically be calculated based on the number of individual
/// examples unless specified otherwise.
///
/// # Examples
///
/// Example of creating a default `BatchType`:
///
/// ```
/// use ct2rs::config::BatchType;
///
/// let batch_type = BatchType::default();
/// assert_eq!(batch_type, BatchType::Examples);
/// ```
#[derive(PartialEq, Eq, Default, Debug)]
pub enum BatchType {
    /// When selected, `max_batch_size` represents the number of individual examples.
    /// This is the default behavior.
    #[default]
    Examples,
    /// When selected, `max_batch_size` corresponds to the total number of tokens across all
    /// examples.
    Tokens,
}
