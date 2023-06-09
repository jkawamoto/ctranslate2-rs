// wrapper.h
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

struct StringArray {
  char *const *strings;
  int length;
};

void release_string_array(struct StringArray a);

class Translator {
private:
  void *impl;

public:
  Translator(const char *model_path);
  ~Translator();

  StringArray translate(const struct StringArray source,
                        const struct StringArray target_prefix);
};
