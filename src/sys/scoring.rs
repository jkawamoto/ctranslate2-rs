// scoring.rs
//
// Copyright (c) 2023-2025 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use super::BatchType;
pub use ffi::{ScoringOptions, ScoringResult};

#[cxx::bridge]
pub(crate) mod ffi {

    #[derive(Clone, Debug)]
    pub struct ScoringOptions {
        /// Truncate the inputs after this many tokens (set 0 to disable truncation).
        pub max_input_length: usize,
        pub offset: i64,

        max_batch_size: usize,
        batch_type: BatchType,
    }

    #[derive(Clone, Debug)]
    pub struct ScoringResult {
        pub tokens: Vec<String>,
        pub tokens_score: Vec<f32>,
    }

    struct _dummy {
        _vec_scoring_result: Vec<ScoringResult>,
    }

    unsafe extern "C++" {
        include!("ct2rs/include/config.h");

        type BatchType = super::BatchType;
    }
}

impl Default for ScoringOptions {
    fn default() -> Self {
        Self {
            max_input_length: 1024,
            offset: 0,
            max_batch_size: 0,
            batch_type: Default::default(),
        }
    }
}

impl ScoringResult {
    pub fn cumulated_score(&self) -> f32 {
        self.tokens_score.iter().sum()
    }

    pub fn normalized_score(&self) -> f32 {
        let num_tokens = self.tokens_score.len();
        if num_tokens == 0 {
            return 0.0;
        }
        self.cumulated_score() / num_tokens as f32
    }
}

#[cfg(test)]
mod tests {
    use crate::sys::scoring::ScoringResult;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_scoring_result() {
        let res = ScoringResult {
            tokens: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            tokens_score: vec![1.0, 2.0, 3.0],
        };
        assert!((res.cumulated_score() - 6.0).abs() < EPSILON);
        assert!((res.normalized_score() - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_empty_scoring_result() {
        let res = ScoringResult {
            tokens: vec![],
            tokens_score: vec![],
        };

        assert_eq!(res.cumulated_score(), 0.0);
        assert_eq!(res.normalized_score(), 0.0);
    }
}
