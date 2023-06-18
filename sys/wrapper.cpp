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

void release_string_array(StringArray const *a) {
  for (unsigned long i = 0; i != a->length; ++i) {
    delete[] a->strings[i];
  }
  delete[] a->strings;
  delete a;
}

void release_translation_result(TranslationResult const *r) {
  // Note: r->hypothesis should be released by the user using
  // release_string_array separately.
  if (r->score != nullptr) {
    delete r->score;
  }
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

inline vector<string> string_array_to_string_vec(const StringArray &a) {
  vector<string> res;
  for (unsigned long i = 0; i != a.length; ++i) {
    res.push_back(string(a.strings[i]));
  }
  return res;
}

inline StringArray *string_vec_to_string_array(const vector<string> &vec) {
  char **ss = new char *[vec.size()];
  for (std::vector<std::string>::size_type i = 0; i != vec.size(); ++i) {
    const string &s = vec[i];
    ss[i] = new char[s.size() + 1];
    std::char_traits<char>::copy(ss[i], s.c_str(), s.size() + 1);
  }
  return new StringArray{ss, vec.size()};
}

TranslationResult const *
Translator::translate(const StringArray &source,
                      const StringArray &target_prefix) const {
  vector<vector<string>> ss;
  ss.push_back(string_array_to_string_vec(source));

  vector<vector<string>> tps;
  tps.push_back(string_array_to_string_vec(target_prefix));

  const auto res = static_cast<ctranslate2::Translator *>(this->impl)
                       ->translate_batch(ss, tps);
  if (res.empty()) {
    return nullptr;
  }

  const auto &res0 = res[0];
  return new TranslationResult{
      res0.num_hypotheses() != 0 ? string_vec_to_string_array(res0.output())
                                 : nullptr,
      res0.has_scores() ? new float(res0.score()) : nullptr};
}
