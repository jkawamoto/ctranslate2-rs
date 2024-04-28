// translator.cpp
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#include "ct2rs/include/translator.h"

#include "ct2rs/include/convert.h"
#include "ct2rs/src/translator.rs.h"

using rust::Str;
using rust::String;
using rust::Vec;
using std::string;
using std::vector;

inline ctranslate2::TranslationOptions convert_options(const TranslationOptions& options) {
    return ctranslate2::TranslationOptions {
        options.beam_size,
        options.patience,
        options.length_penalty,
        options.coverage_penalty,
        options.repetition_penalty,
        options.no_repeat_ngram_size,
        options.disable_unk,
        from_rust(options.suppress_sequences),
        options.prefix_bias_beta,
        {},
        options.return_end_token,
        options.max_input_length,
        options.max_decoding_length,
        options.min_decoding_length,
        options.sampling_topk,
        options.sampling_topp,
        options.sampling_temperature,
        options.use_vmap,
        options.num_hypotheses,
        options.return_scores,
        options.return_attention,
        options.return_alternatives,
        options.min_alternative_expansion_prob,
        options.replace_unknowns,
        nullptr,
    };
}

inline Vec<TranslationResult> convert_results(const std::vector<ctranslate2::TranslationResult> results) {
    Vec<TranslationResult> res;
    for (const auto& item : results) {
        res.push_back(TranslationResult {
            to_rust<VecString>(item.hypotheses),
            to_rust(item.scores),
            // to_rust(item.attention),
        });
    }
    return res;
}

Vec<TranslationResult> Translator::translate_batch(
    Vec<VecStr> source,
    TranslationOptions options
) const {
    return convert_results(this->impl->translate_batch(
        from_rust(source),
        convert_options(options),
        options.max_batch_size,
        options.batch_type
    ));
}

Vec<TranslationResult> Translator::translate_batch_with_target_prefix(
    Vec<VecStr> source,
    Vec<VecStr> target_prefix,
    TranslationOptions options
) const {
    return convert_results(this->impl->translate_batch(
        from_rust(source),
        from_rust(target_prefix),
        convert_options(options),
        options.max_batch_size,
        options.batch_type
    ));
}
