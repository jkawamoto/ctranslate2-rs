// config.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! Configs and associated enums.

use cxx::UniquePtr;

pub use ffi::BatchType;
pub use ffi::ComputeType;
pub use ffi::Device;

#[cxx::bridge]
pub(crate) mod ffi {
    /// Represents the computing device to be used.
    ///
    /// This enum can take one of the following two values:
    /// - `CPU`
    /// - `CUDA`
    ///
    /// The default setting for this enum is `CPU`.
    ///
    /// # Examples
    ///
    /// Example of creating a default `Device`:
    ///
    /// ```
    /// use ct2rs::config::Device;
    ///
    /// let device = Device::default();
    /// assert_eq!(device, Device::CPU);
    /// ```
    ///
    #[derive(Debug)]
    #[repr(i32)]
    enum Device {
        CPU,
        CUDA,
    }


    /// Model computation type.
    ///
    /// This enum can take one of the following values:
    /// - `DEFAULT`: Keeps the same quantization that was used during model conversion.
    ///   This is the default setting.
    /// - `AUTO`: Uses the fastest computation type that is supported on this system and device.
    /// - `FLOAT32`: Utilizes 32-bit floating-point precision.
    /// - `INT8`: Uses 8-bit integer precision.
    /// - `INT8_FLOAT32`: Combines 8-bit integer quantization with 32-bit floating-point
    ///   computation.
    /// - `INT8_FLOAT16`: Combines 8-bit integer quantization with 16-bit floating-point
    ///   computation.
    /// - `INT8_BFLOAT16`: Combines 8-bit integer quantization with Brain Floating Point (16-bit)
    ///   computation.
    /// - `INT16`: Uses 16-bit integer precision.
    /// - `FLOAT16`: Utilizes 16-bit floating-point precision (half precision).
    /// - `BFLOAT16`: Uses Brain Floating Point (16-bit) precision.
    ///
    /// The default setting for this enum is `DEFAULT`, meaning that unless specified otherwise,
    /// the computation will proceed with the same quantization level as was used during the model's
    /// conversion.
    ///
    /// See also:
    /// [Quantization](https://opennmt.net/CTranslate2/quantization.html#quantize-on-model-loading)
    /// for more details on how quantization affects computation and how it can be applied during
    /// model loading.
    ///
    /// # Examples
    ///
    /// Example of creating a default `ComputeType`:
    ///
    /// ```
    /// use ct2rs::config::ComputeType;
    ///
    /// let compute_type = ComputeType::default();
    /// assert_eq!(compute_type, ComputeType::DEFAULT);
    /// ```
    ///
    #[derive(Debug)]
    #[repr(i32)]
    enum ComputeType {
        DEFAULT,
        AUTO,
        FLOAT32,
        INT8,
        INT8_FLOAT32,
        INT8_FLOAT16,
        INT8_BFLOAT16,
        INT16,
        FLOAT16,
        BFLOAT16,
    }


    /// Specifies how the `max_batch_size` should be calculated,
    /// whether by the number of "examples" or "tokens".
    ///
    /// This enum can take one of the following two values:
    /// - `Examples`: The batch size is calculated based on the number of individual examples.
    /// - `Tokens`: The batch size is calculated based on the total number of tokens across all
    ///    examples.
    ///
    /// The default setting for this enum is `Examples`.
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
    #[derive(Debug)]
    #[repr(i32)]
    enum BatchType {
        Examples,
        Tokens,
    }


    unsafe extern "C++" {
        include!("ct2rs/include/config.h");

        type Device;
        type ComputeType;
        type ReplicaPoolConfig;
        pub type BatchType;

        fn replica_pool_config(
            num_threads_per_replica: usize,
            max_queued_batches: i32,
            cpu_core_offset: i32,
        ) -> UniquePtr<ReplicaPoolConfig>;

        pub type Config;

        fn config(
            device: Device,
            compute_type: ComputeType,
            device_indices: &[i32],
            tensor_parallel: bool,
            replica_pool_config: UniquePtr<ReplicaPoolConfig>,
        ) -> UniquePtr<Config>;
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::CPU
    }
}

impl Default for ComputeType {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Default for BatchType {
    fn default() -> Self {
        Self::Examples
    }
}

/// The `Config` structure holds the configuration settings for CTranslator2.
///
/// # Examples
///
/// Example of creating a default `Config`:
///
/// ```
/// use ct2rs::config::{ComputeType, Config, Device};
///
/// let config = Config::default();
/// assert_eq!(config.device, Device::default());
/// assert_eq!(config.compute_type, ComputeType::default());
/// assert_eq!(config.device_indices, vec![0]);
/// assert_eq!(config.tensor_parallel, false);
/// assert_eq!(config.num_threads_per_replica, 0);
/// assert_eq!(config.max_queued_batches, 0);
/// assert_eq!(config.cpu_core_offset, -1);
/// ```
#[derive(PartialEq, Eq, Debug)]
pub struct Config {
    /// Device to use.
    pub device: Device,
    /// Model computation type.
    pub compute_type: ComputeType,
    /// Device IDs where to place this generator on.
    pub device_indices: Vec<i32>,
    /// Run model with tensor parallel mode.
    pub tensor_parallel: bool,
    pub num_threads_per_replica: usize,
    /// Maximum numbers of batches in the queue (-1 for unlimited, 0 for an automatic value).
    /// When the queue is full, future requests will block until a free slot is available.
    pub max_queued_batches: i32,
    pub cpu_core_offset: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            device: Default::default(),
            compute_type: Default::default(),
            device_indices: vec![0],
            tensor_parallel: false,
            num_threads_per_replica: 0,
            max_queued_batches: 0,
            cpu_core_offset: -1,
        }
    }
}

impl Config {
    pub(crate) fn to_ffi(&self) -> UniquePtr<ffi::Config> {
        ffi::config(
            self.device,
            self.compute_type,
            self.device_indices.as_slice(),
            false,
            ffi::replica_pool_config(
                self.num_threads_per_replica,
                self.max_queued_batches,
                self.cpu_core_offset,
            ),
        )
    }
}


#[cfg(test)]
mod tests {
    use crate::config::Config;

    #[test]
    fn test_config_to_ffi() {
        let config = Config::default();
        let res = config.to_ffi();

        assert!(!res.is_null());
    }
}
