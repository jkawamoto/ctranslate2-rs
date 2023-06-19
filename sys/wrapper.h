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

struct StringArrayArray {
  StringArray const *const *arrays;
  unsigned long length;
};

struct TranslationResult {
  StringArray const *hypothesis;
  float const *score;
};

void release_translation_result(TranslationResult const *r);

struct TranslationResultArray {
  TranslationResult const *const *arrays;
  unsigned long length;
};

void release_translation_result_array(TranslationResultArray const *r);

class Translator {
private:
  void *impl;

public:
  Translator(char const *model_path);
  ~Translator();

  //  TranslationResult const *translate(const StringArray &source,
  //                                     const StringArray &target_prefix)
  //                                     const;

  TranslationResultArray const *
  translate_batch(const StringArrayArray &source,
                  const StringArrayArray &target_prefix) const;
};
