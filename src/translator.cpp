// translator.cpp
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#include "ctranslate2/include/translator.h"
#include "ctranslate2/include/convert.h"
#include "ctranslate2/src/translator.rs.h"

using rust::Str;
using rust::String;
using rust::Vec;
using std::string;
using std::vector;

Vec<TranslationResult>
Translator::translate_batch(Vec<VecStr> source,
                            Vec<VecStr> target_prefix) const {
  const auto batch_result =
      this->impl->translate_batch(from_rust(source), from_rust(target_prefix));
  Vec<TranslationResult> res;
  for (const auto &item : batch_result) {
    res.push_back(TranslationResult{
        to_rust(item.hypotheses), to_rust(item.scores),
        //        to_rust(item.attention),
    });
  }
  return res;
}

std::unique_ptr<Translator> new_translator(const Str model_path) {
  const ctranslate2::models::ModelLoader model_loader(
      static_cast<string>(model_path));
  return std::make_unique<Translator>(
      std::make_shared<ctranslate2::Translator>(model_loader));
}
