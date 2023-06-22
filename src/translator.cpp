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

std::unique_ptr<Translator> new_translator(const Str model_path,
                                           const bool cuda,
                                           const TranslatorConfig config) {
  ctranslate2::ComputeType compute_type;
  switch (config.compute_type) {
  case ComputeType::Default:
    compute_type = ctranslate2::ComputeType::DEFAULT;
    break;
  case ComputeType::Auto:
    compute_type = ctranslate2::ComputeType::AUTO;
    break;
  case ComputeType::Float32:
    compute_type = ctranslate2::ComputeType::FLOAT32;
    break;
  case ComputeType::Int8:
    compute_type = ctranslate2::ComputeType::INT8;
    break;
  case ComputeType::Int8Float16:
    compute_type = ctranslate2::ComputeType::INT8_FLOAT16;
    break;
  case ComputeType::Int16:
    compute_type = ctranslate2::ComputeType::INT16;
    break;
  case ComputeType::Float16:
    compute_type = ctranslate2::ComputeType::FLOAT16;
    break;
  };

  return std::make_unique<Translator>(std::make_shared<ctranslate2::Translator>(
      static_cast<string>(model_path),
      cuda ? ctranslate2::Device::CUDA : ctranslate2::Device::CPU, compute_type,
      from_rust(config.device_indices),
      ctranslate2::ReplicaPoolConfig{config.num_threads_per_replica,
                                     config.max_queued_batches,
                                     config.cpu_core_offset}));
}
