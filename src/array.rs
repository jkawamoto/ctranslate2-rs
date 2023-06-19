// array.rs
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

use std::ffi::{c_char, c_ulong, CStr, CString};
use std::slice::Iter;

use anyhow::Result;

use ctranslate2_sys::{release_string_array, StringArray, StringArrayArray};

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

pub struct StringArrayArrayAux(
    Vec<StringArrayAux>,
    Vec<*const StringArray>,
    StringArrayArray,
);

impl<T: Into<Vec<u8>>> From<Vec<Vec<T>>> for StringArrayArrayAux {
    fn from(values: Vec<Vec<T>>) -> Self {
        let auxes: Vec<_> = values.into_iter().map(StringArrayAux::from).collect();
        let arrays: Vec<_> = auxes.iter().map(|aux| aux.as_ptr()).collect();
        let a = StringArrayArray {
            arrays: arrays.as_ptr(),
            length: auxes.len() as c_ulong,
        };
        Self(auxes, arrays, a)
    }
}

impl StringArrayArrayAux {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn as_ptr(&self) -> *const StringArrayArray {
        &self.2 as *const StringArrayArray
    }

    pub fn iter(&self) -> Iter<'_, StringArrayAux> {
        self.0.iter()
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

    use ctranslate2_sys::StringArray;

    use crate::array::{StringArrayArrayAux, StringArrayAux};

    fn assert_eq_string_array_aux(aux: &StringArrayAux, expect: &Vec<&str>) {
        assert_eq!(aux.0.len(), expect.len());
        for (s, expect) in aux.0.iter().zip(expect) {
            assert_eq!(s.to_bytes(), expect.as_bytes());
        }
    }

    unsafe fn assert_eq_string_array_ptr(p: *const StringArray, expect: &Vec<&str>) {
        assert_eq!((*p).length as usize, expect.len());
        for (v, expect) in std::slice::from_raw_parts((*p).strings, expect.len())
            .iter()
            .map(|s| CStr::from_ptr(*s))
            .zip(expect)
        {
            assert_eq!(v.to_str().unwrap(), *expect);
        }
    }

    #[test]
    fn test_string_array_aux() {
        let data = vec!["a", "aa", "aaa"];
        let strings = StringArrayAux::from(data.clone());

        assert_eq!(strings.len(), data.len());
        assert_eq_string_array_aux(&strings, &data);
        unsafe {
            assert_eq_string_array_ptr(strings.as_ptr(), &data);
        }
    }

    #[test]
    fn test_string_array_array_aux() {
        let data = vec![vec!["a"], vec!["bb", "b"], vec!["ccc", "cc", "c"]];
        let aux = StringArrayArrayAux::from(data.clone());

        assert_eq!(aux.len(), data.len());
        for (a, v) in aux.0.iter().zip(&data) {
            assert_eq_string_array_aux(a, v);
        }

        unsafe {
            assert_eq!((*aux.as_ptr()).length as usize, data.len());
            for (a, v) in std::slice::from_raw_parts((*aux.as_ptr()).arrays, data.len())
                .iter()
                .zip(&data)
            {
                assert_eq_string_array_ptr(*a, v);
            }
        }

        for (a, d) in aux.iter().zip(data.iter()) {
            assert_eq!(a.len(), d.len());
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
