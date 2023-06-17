// array.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::ffi::{c_char, c_ulong, CStr, CString};

use anyhow::Result;

use ctranslate2_sys::{release_string_array, StringArray};

pub struct StringArrayAux(Vec<CString>, Vec<*const c_char>, StringArray);

impl<T: Into<Vec<u8>>> From<Vec<T>> for StringArrayAux {
    fn from(strings: Vec<T>) -> Self {
        let cstrings: Vec<_> = strings
            .into_iter()
            .map(|s| CString::new(s).expect("failed to convert a string to CString"))
            .collect();
        let ptrs: Vec<_> = cstrings.iter().map(|c| c.as_ptr()).collect();
        let a = StringArray {
            strings: ptrs.as_ptr(),
            length: ptrs.len() as c_ulong,
        };
        Self(cstrings, ptrs, a)
    }
}

impl StringArrayAux {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn as_ptr(&self) -> *const StringArray {
        &self.2 as *const StringArray
    }
}

pub struct StringArrayPtr(*const StringArray);

impl From<*const StringArray> for StringArrayPtr {
    fn from(ptr: *const StringArray) -> Self {
        Self(ptr)
    }
}

impl StringArrayPtr {
    pub fn to_vec(&self) -> Result<Vec<String>> {
        let mut res = Vec::new();
        for s in unsafe {
            std::slice::from_raw_parts((*self.0).strings, (*self.0).length as usize)
                .iter()
                .map(|s| CStr::from_ptr(*s))
        } {
            res.push(s.to_str()?.to_string());
        }
        Ok(res)
    }
}

impl Drop for StringArrayPtr {
    fn drop(&mut self) {
        unsafe { release_string_array(self.0) };
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use crate::array::StringArrayAux;

    #[test]
    fn test_string_array_aux() {
        let data = vec!["a", "aa", "aaa"];
        let strings = StringArrayAux::from(data.clone());

        assert_eq!(strings.0.len(), data.len());
        for (s, expect) in strings.0.iter().zip(&data) {
            assert_eq!(s.to_bytes(), expect.as_bytes());
        }

        let p = strings.as_ptr();
        unsafe {
            assert_eq!((*p).length as usize, data.len());
            for (v, expect) in std::slice::from_raw_parts((*p).strings, data.len())
                .iter()
                .map(|s| CStr::from_ptr(*s))
                .zip(data)
            {
                assert_eq!(v.to_str().unwrap(), expect);
            }
        }
    }

    // #[test]
    // fn test_string_array_ptr() {
    //     let data = vec!["b", "bb", "bbb"];
    //     let arr = StringArrayAux::from(data.clone());
    //     let cstrs = StringArrayPtr::from(arr.as_ptr());
    //
    //     for (s, expect) in cstrs.to_vec().unwrap().iter().zip(data) {
    //         assert_eq!(s.as_str(), expect);
    //     }
    // }
}
