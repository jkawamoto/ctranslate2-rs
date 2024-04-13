// generator.cpp
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#include "ct2rs/include/generator.h"
#include "ct2rs/include/convert.h"
#include "ct2rs/src/generator.rs.h"

using rust::Str;
using rust::Vec;
using std::string;
using std::vector;

Vec<GenerationResult>
Generator::generate_batch(Vec<GenVecStr> start_tokens,
                          GenerationOptions options) const {

  ctranslate2::BatchType batch_type;
  switch (options.batch_type) {
  case GenerationBatchType::Examples:
    batch_type = ctranslate2::BatchType::Examples;
    break;
  case GenerationBatchType::Tokens:
    batch_type = ctranslate2::BatchType::Tokens;
    break;
  }

  auto futures = this->impl->generate_batch_async(
      from_rust(start_tokens),
      ctranslate2::GenerationOptions{options.beam_size,
                                     options.patience,
                                     options.length_penalty,
                                     options.repetition_penalty,
                                     options.no_repeat_ngram_size,
                                     options.disable_unk,
                                     from_rust(options.suppress_sequences),
                                     {},
                                     options.return_end_token,
                                     options.max_length,
                                     options.min_length,
                                     options.sampling_topk,
                                     options.sampling_topp,
                                     options.sampling_temperature,
                                     options.num_hypotheses,
                                     options.return_scores,
                                     options.return_alternatives,
                                     options.min_alternative_expansion_prob,
                                     from_rust(options.static_prompt),
                                     options.cache_static_prompt,
                                     options.include_prompt_in_result,
                                     nullptr},
      options.max_batch_size, batch_type);

  Vec<GenerationResult> res;
  for (auto &future : futures) {
    const auto &r = future.get();
    res.push_back(GenerationResult{to_rust<GenVecString>(r.sequences),
                                   to_rust<GenVecUSize>(r.sequences_ids),
                                   to_rust(r.scores)});
  }

  return res;
}

std::unique_ptr<Generator> new_generator(const Str model_path, const bool cuda,
                                         const GeneratorConfig config) {
  ctranslate2::ComputeType compute_type;
  switch (config.compute_type) {
  case GenComputeType::Default:
    compute_type = ctranslate2::ComputeType::DEFAULT;
    break;
  case GenComputeType::Auto:
    compute_type = ctranslate2::ComputeType::AUTO;
    break;
  case GenComputeType::Float32:
    compute_type = ctranslate2::ComputeType::FLOAT32;
    break;
  case GenComputeType::Int8:
    compute_type = ctranslate2::ComputeType::INT8;
    break;
  case GenComputeType::Int8Float16:
    compute_type = ctranslate2::ComputeType::INT8_FLOAT16;
    break;
  case GenComputeType::Int16:
    compute_type = ctranslate2::ComputeType::INT16;
    break;
  case GenComputeType::Float16:
    compute_type = ctranslate2::ComputeType::FLOAT16;
    break;
  };

  return std::make_unique<Generator>(std::make_shared<ctranslate2::Generator>(
      static_cast<string>(model_path),
      cuda ? ctranslate2::Device::CUDA : ctranslate2::Device::CPU, compute_type,
      from_rust(config.device_indices),
      ctranslate2::ReplicaPoolConfig{config.num_threads_per_replica,
                                     config.max_queued_batches,
                                     config.cpu_core_offset}));
}
