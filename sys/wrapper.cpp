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

Translator::Translator(char const *model_path) {
  const auto model = ctranslate2::models::Model::load(std::string(model_path));
  const ctranslate2::models::ModelLoader model_loader(model_path);
  this->impl = new ctranslate2::Translator(model_loader);
}

Translator::~Translator() {
  delete static_cast<ctranslate2::Translator *>(this->impl);
}

StringArray const *
Translator::translate(const StringArray &sources,
                      const StringArray &target_prefix) const {
  vector<string> s, tp;
  for (unsigned long i = 0; i != sources.length; ++i) {
    s.push_back(string(sources.strings[i]));
  }
  for (unsigned long i = 0; i != target_prefix.length; ++i) {
    tp.push_back(string(target_prefix.strings[i]));
  }

  vector<vector<string>> ss, tps;
  ss.push_back(s);
  tps.push_back(tp);

  const auto output = static_cast<ctranslate2::Translator *>(this->impl)
                          ->translate_batch(ss, tps)[0]
                          .output();
  char **strings = new char *[output.size()];

  for (std::vector<std::string>::size_type i = 0; i != output.size(); ++i) {
    const string &s = output[i];
    strings[i] = new char[s.size() + 1];
    std::char_traits<char>::copy(strings[i], s.c_str(), s.size() + 1);
  }

  return new StringArray{strings, output.size()};
}
