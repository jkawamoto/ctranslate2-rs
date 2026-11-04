#!/usr/bin/env bash
# Download Whisper model weights for testing and local development at `./.models`
#
# Environment variables:
#   MODELS_DIR    Where to store models        (default: <repo>/.models)
#   MODEL_SIZE    Systran repo name            (default: faster-whisper-base)
#   HF_REVISION   Branch/tag/commit to fetch   (default: main; pin a commit SHA for reproducibility)
#   OPENAI_REPO   Repo providing preprocessor_config.json
#                 (default: derived from MODEL_SIZE, e.g. openai/whisper-base)
#   HF_TOKEN      Optional Hugging Face token
#   FORCE=1       Re-download files even if already cached
set -euo pipefail

# Determine repository root regardless of where the script is executed from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/.models}"
MODEL_SIZE="${MODEL_SIZE:-faster-whisper-base}"
HF_REVISION="${HF_REVISION:-main}"
# faster-whisper-base -> whisper-base
OPENAI_REPO="${OPENAI_REPO:-openai/${MODEL_SIZE#faster-}}"
WHISPER_DIR="$MODELS_DIR/$MODEL_SIZE"

mkdir -p "$WHISPER_DIR"

# Build optional header array if HF_TOKEN is supplied in environment
AUTH_HEADER=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

# Show a progress bar on interactive terminals; stay quiet (but show errors) otherwise
if [[ -t 2 ]]; then
  PROGRESS_OPTS=(--progress-bar)
else
  PROGRESS_OPTS=(-sS)
fi

# Track the in-flight temp file so it is removed on failure or interruption
CURRENT_TMP=""
cleanup() {
  if [[ -n "$CURRENT_TMP" ]]; then
    rm -f "$CURRENT_TMP"
  fi
}
trap cleanup EXIT

download_file() {
  local url="$1"
  local dest="$2"

  if [[ -s "$dest" && "${FORCE:-0}" != "1" ]]; then
    echo "  [cached] $(basename "$dest")"
    return 0
  fi

  echo "  [downloading] $(basename "$dest")..."
  CURRENT_TMP="$dest.tmp"
  curl -fL "${PROGRESS_OPTS[@]}" \
    --retry 3 --retry-delay 2 \
    ${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"} \
    -o "$CURRENT_TMP" "$url"

  if [[ ! -s "$CURRENT_TMP" ]]; then
    echo "  [error] downloaded file is empty: $url" >&2
    return 1
  fi

  mv "$CURRENT_TMP" "$dest"
  CURRENT_TMP=""
}

echo "Syncing model files to '$WHISPER_DIR' (revision: $HF_REVISION)..."

for f in config.json tokenizer.json vocabulary.txt model.bin; do
  download_file \
    "https://huggingface.co/Systran/$MODEL_SIZE/resolve/$HF_REVISION/$f" \
    "$WHISPER_DIR/$f"
done

# Systran/faster-whisper-* repos don't ship preprocessor_config.json; ct2rs::Whisper::new
# requires it, so fetch the standard one from the original OpenAI model.
download_file \
  "https://huggingface.co/$OPENAI_REPO/resolve/main/preprocessor_config.json" \
  "$WHISPER_DIR/preprocessor_config.json"

echo "Model files downloaded to '$WHISPER_DIR'."
