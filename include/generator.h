// generator.h
//
// Copyright (c) 2023 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#pragma once

#include "rust/cxx.h"

#include <ctranslate2/generator.h>
#include <memory>

struct GenVecStr;
struct GeneratorConfig;
struct GenerationOptions;
struct GenerationResult;

class Generator {
private:
  std::shared_ptr<ctranslate2::Generator> impl;

public:
  Generator(std::shared_ptr<ctranslate2::Generator> impl) : impl(impl) {}

  rust::Vec<GenerationResult> generate_batch(rust::Vec<GenVecStr> start_tokens,
                                             GenerationOptions options) const;
};

std::unique_ptr<Generator> new_generator(rust::Str model_path, bool cuda,
                                         GeneratorConfig config);
