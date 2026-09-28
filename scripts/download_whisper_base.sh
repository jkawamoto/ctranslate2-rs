#!/usr/bin/env bash
# Download Whisper model weights for testing and local development at `./.models`
set -euo pipefail

# Determine repository root regardless of where the script is executed from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/.models}"
MODEL_SIZE="${MODEL_SIZE:-faster-whisper-base}"
WHISPER_DIR="$MODELS_DIR/$MODEL_SIZE"

mkdir -p "$WHISPER_DIR"

# Build optional header array if HF_TOKEN is supplied in environment
AUTH_HEADER=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

download_file() {
  local url="$1"
  local dest="$2"

  if [[ -f "$dest" && -s "$dest" && "${FORCE:-0}" != "1" ]]; then
    echo "  [cached] $(basename "$dest")"
    return 0
  fi

  echo "  [downloading] $(basename "$dest")..."
  curl -sSfL ${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"} -o "$dest" "$url"
}

echo "Syncing model files to '$WHISPER_DIR'..."

for f in config.json tokenizer.json vocabulary.txt model.bin; do
  download_file \
    "https://huggingface.co/Systran/$MODEL_SIZE/resolve/main/$f" \
    "$WHISPER_DIR/$f"
done

# Systran/faster-whisper-base doesn't ship preprocessor_config.json; ct2rs::Whisper::new
# requires it, so fetch the standard one from the original OpenAI model.
download_file \
  "https://huggingface.co/openai/whisper-base/resolve/main/preprocessor_config.json" \
  "$WHISPER_DIR/preprocessor_config.json"

echo "Model weights successfully verified in '$WHISPER_DIR'."
