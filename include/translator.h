// translator.h
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#pragma once

#include "rust/cxx.h"

#include <ctranslate2/translator.h>
#include <memory>

struct VecStr;
struct TranslatorConfig;
struct TranslationResult;

class Translator {
private:
  std::shared_ptr<ctranslate2::Translator> impl;

public:
  Translator(std::shared_ptr<ctranslate2::Translator> impl) : impl(impl) {}

  rust::Vec<TranslationResult>
  translate_batch(rust::Vec<VecStr> source,
                  rust::Vec<VecStr> target_prefix) const;
};

std::unique_ptr<Translator> new_translator(rust::Str model_path, bool cuda,
                                           TranslatorConfig config);
