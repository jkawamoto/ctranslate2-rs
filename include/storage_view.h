// storage_view.h
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

#pragma once

#include <memory>
#include <sstream>
#include <string>

#include <ctranslate2/storage_view.h>

#include "rust/cxx.h"

using ctranslate2::Device;
using ctranslate2::StorageView;

inline std::unique_ptr<StorageView> storage_view_from_float(
    const rust::Slice<const size_t> shape,
    const rust::Slice<const float> init,
    const Device device
) {
    return std::make_unique<StorageView>(
        ctranslate2::Shape(shape.begin(), shape.end()),
        std::vector<float>(init.begin(), init.end()),
        device
    );
}

inline std::unique_ptr<StorageView> storage_view_from_int8(
    const rust::Slice<const size_t> shape,
    const rust::Slice<const int8_t> init,
    const Device device
) {
    return std::make_unique<StorageView>(
        ctranslate2::Shape(shape.begin(), shape.end()),
        std::vector<int8_t>(init.begin(), init.end()),
        device
    );
}

inline std::unique_ptr<StorageView> storage_view_from_int16(
    const rust::Slice<const size_t> shape,
    const rust::Slice<const int16_t> init,
    const Device device
) {
    return std::make_unique<StorageView>(
        ctranslate2::Shape(shape.begin(), shape.end()),
        std::vector<int16_t>(init.begin(), init.end()),
        device
    );
}

rust::String to_string(const StorageView& storage) {
    std::ostringstream oss;
    oss << storage;

    return rust::String(oss.str());
}
