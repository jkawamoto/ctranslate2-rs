// wrapper.h
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

struct StringArray {
  char const *const *strings;
  unsigned long length;
};

void release_string_array(StringArray const *a);

class Translator {
private:
  void *impl;

public:
  Translator(char const *model_path);
  ~Translator();

  StringArray const *translate(const StringArray &source,
                               const StringArray &target_prefix) const;
};
