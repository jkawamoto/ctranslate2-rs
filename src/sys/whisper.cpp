// whisper.cpp
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#include <utility>

#include "ct2rs/src/sys/whisper.rs.h"

#include "ct2rs/include/types.h"

using rust::Slice;
using rust::Vec;

Vec<WhisperGenerationResult> Whisper::generate(
    const StorageView& features,
    const Slice<const VecStr> prompts,
    const WhisperOptions& opts
) const {
    auto futures = impl->generate(
        features,
        from_rust(prompts),
        ctranslate2::models::WhisperOptions {
            opts.beam_size,
            opts.patience,
            opts.length_penalty,
            opts.repetition_penalty,
            opts.no_repeat_ngram_size,
            opts.max_length,
            opts.sampling_topk,
            opts.sampling_temperature,
            opts.num_hypotheses,
            opts.return_scores,
            opts.return_no_speech_prob,
            opts.max_initial_timestamp_index,
            opts.suppress_blank,
            from_rust(opts.suppress_tokens),
        }
    );

    Vec<WhisperGenerationResult> res;
    for (auto& future : futures) {
        const auto& r = future.get();
        res.push_back(WhisperGenerationResult {
            to_rust<VecString>(r.sequences),
            to_rust<VecUSize>(r.sequences_ids),
            to_rust(r.scores),
            r.no_speech_prob,
        });
    }

    return res;
}

Vec<VecDetectionResult> Whisper::detect_language(const StorageView& features) const {
    auto futures = impl->detect_language(features);

    Vec<VecDetectionResult> res;
    for (auto& future : futures) {
        const auto& r = future.get();

        Vec<DetectionResult> pairs;
        for (auto& pair : r) {
            pairs.push_back(DetectionResult {
                std::get<0>(pair),
                std::get<1>(pair),
            });
        }

        res.push_back(VecDetectionResult { pairs });
    }

    return res;
}
