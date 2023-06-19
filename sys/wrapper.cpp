// wrapper.cpp
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#include <ctranslate2/translator.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "wrapper.h"

using std::string;
using std::vector;

// Helper functions for StringArray.
inline vector<string> to_vec(const StringArray &a) {
  vector<string> res;
  for (unsigned long i = 0; i != a.length; ++i) {
    res.push_back(string(a.strings[i]));
  }
  return res;
}

inline StringArray *from_vec(const vector<string> &vec) {
  char **ss = new char *[vec.size()];
  for (std::vector<std::string>::size_type i = 0; i != vec.size(); ++i) {
    const string &s = vec[i];
    ss[i] = new char[s.size() + 1];
    std::char_traits<char>::copy(ss[i], s.c_str(), s.size() + 1);
  }
  return new StringArray{ss, vec.size()};
}

void release_string_array(StringArray const *a) {
  for (unsigned long i = 0; i != a->length; ++i) {
    delete[] a->strings[i];
  }
  delete[] a->strings;
  delete a;
}

// Helper functions for StringArrayArray.
inline vector<vector<string>> to_vec(const StringArrayArray &a) {
  vector<vector<string>> res;
  for (unsigned long i = 0; i != a.length; ++i) {
    res.push_back(to_vec(*a.arrays[i]));
  }
  return res;
}

// Helper functions for TranslationResult.
void release_translation_result(TranslationResult const *r) {
  // Note: r->hypothesis should be released by the user using
  // release_string_array separately.
  if (r->score != nullptr) {
    delete r->score;
  }
  delete r;
}

// Helper functions for TranslationResultArray.
void release_translation_result_array(TranslationResultArray const *r) {
  // Note: r->arrays should be released by the user using
  // release_translation_result separately.
  delete[] r->arrays;
  delete r;
}

Translator::Translator(char const *model_path) {
  const auto model = ctranslate2::models::Model::load(std::string(model_path));
  const ctranslate2::models::ModelLoader model_loader(model_path);
  this->impl = new ctranslate2::Translator(model_loader);
}

Translator::~Translator() {
  delete static_cast<ctranslate2::Translator *>(this->impl);
}

// TranslationResult const *
// Translator::translate(const StringArray &source,
//                       const StringArray &target_prefix) const {
//   vector<vector<string>> ss;
//   ss.push_back(to_vec(source));
//
//   vector<vector<string>> tps;
//   tps.push_back(to_vec(target_prefix));
//
//   const auto res = static_cast<ctranslate2::Translator *>(this->impl)
//                        ->translate_batch(ss, tps);
//   if (res.empty()) {
//     return nullptr;
//   }
//
//   const auto &res0 = res[0];
//   return new TranslationResult{
//       res0.num_hypotheses() != 0 ? from_vec(res0.output()) : nullptr,
//       res0.has_scores() ? new float(res0.score()) : nullptr,
//   };
// }

TranslationResultArray const *
Translator::translate_batch(const StringArrayArray &source,
                            const StringArrayArray &target_prefix) const {
  const auto res = static_cast<ctranslate2::Translator *>(this->impl)
                       ->translate_batch(to_vec(source), to_vec(target_prefix));
  if (res.empty()) {
    return nullptr;
  }

  const auto arrays = new TranslationResult *[res.size()];
  for (auto i = res.begin(); i != res.end(); ++i) {
    arrays[std::distance(res.begin(), i)] = new TranslationResult{
        i->num_hypotheses() != 0 ? from_vec(i->output()) : nullptr,
        i->has_scores() ? new float(i->score()) : nullptr,
    };
  }

  return new TranslationResultArray{arrays, res.size()};
}
