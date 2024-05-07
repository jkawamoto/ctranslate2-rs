// types.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This module defines common structures.

#[cxx::bridge]
pub(crate) mod ffi {
    /// The result for a single generation step.
    #[derive(Clone, Debug)]
    pub struct GenerationStepResult{
        /// The decoding step.
        pub step: usize,
        /// The batch index.
        pub batch_id: usize,
        /// ID of the generated token.
        pub token_id: usize,
        /// Index of the hypothesis in the batch.
        pub hypothesis_id: usize,
        /// String value of the generated token.
        pub token: String,
        /// true if return_log_prob was enabled
        pub has_log_prob: bool,
        /// Log probability of the token.
        pub log_prob: f32,
        /// Whether this step is the last decoding step for this batch.
        pub is_last: bool,
    }

    pub struct VecString {
        v: Vec<String>,
    }

    pub struct VecStr<'a> {
        v: Vec<&'a str>,
    }

    pub struct VecUSize {
        v: Vec<usize>,
    }

    struct _dummy<'a> {
        _vec_string: Vec<VecString>,
        _vec_str: Vec<VecStr<'a>>,
        _vec_usize: Vec<VecUSize>,
    }
}

#[inline]
pub(crate) fn vec_ffi_vecstr<T: AsRef<str>>(src: &Vec<Vec<T>>) -> Vec<ffi::VecStr> {
    src.iter()
        .map(|v| ffi::VecStr {
            v: v.iter().map(|s| s.as_ref()).collect(),
        })
        .collect()
}

pub(crate) fn noop_callback(_r: ffi::GenerationStepResult) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use crate::types::vec_ffi_vecstr;

    #[test]
    fn str_vectors() {
        let data = vec![vec!["a", "b", "c"], vec!["1", "2"]];
        let res = vec_ffi_vecstr(&data);

        assert_eq!(res.len(), data.len());
        for (i, list) in data.iter().enumerate() {
            let v = &res.get(i).unwrap().v;
            assert_eq!(v.len(), list.len());
            for (j, s) in list.iter().enumerate() {
                assert_eq!(v.get(j).unwrap(), s);
            }
        }
    }

    #[test]
    fn empty_inner_vectors() {
        let data: Vec<Vec<&str>> = vec![vec![], vec![]];
        let res = vec_ffi_vecstr(&data);

        assert_eq!(res.len(), data.len());
        for item in res.iter() {
            assert_eq!(item.v.len(), 0);
        }
    }

    #[test]
    fn empty_vectors() {
        let data: Vec<Vec<&str>> = vec![];
        let res = vec_ffi_vecstr(&data);

        assert_eq!(res.len(), 0);
    }
}
