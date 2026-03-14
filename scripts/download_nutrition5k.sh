#!/usr/bin/env bash

set -euo pipefail

DATASET_BUCKET="gs://nutrition5k_dataset/nutrition5k_dataset"
TARBALL_URL="https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset.tar.gz"

print_usage() {
  cat <<'EOF'
Usage:
  scripts/download_nutrition5k.sh [OPTIONS]

Download the official Nutrition5K dataset from the public Google Cloud bucket.

Options:
  --output-dir PATH       Directory where the dataset or tarball is written.
                          Default: data
  --method METHOD         Download method: gsutil or tarball.
                          Default: gsutil
  --path SUBPATH          Optional dataset subpath to download with gsutil.
                          Example: metadata or imagery/realsense_overhead
  --extract               Extract the downloaded tarball after download.
                          Only used with --method tarball.
  --keep-tarball          Keep the tarball after extraction.
  --help                  Show this message.

Examples:
  scripts/download_nutrition5k.sh
  scripts/download_nutrition5k.sh --output-dir /datasets --path metadata
  scripts/download_nutrition5k.sh --method tarball --output-dir /datasets --extract

Notes:
  - The full tarball is approximately 181.4 GB.
  - The official dataset source is:
    https://github.com/google-research-datasets/Nutrition5k
EOF
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
}

OUTPUT_DIR="data"
METHOD="gsutil"
SUBPATH=""
EXTRACT_TARBALL="false"
KEEP_TARBALL="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --path)
      SUBPATH="$2"
      shift 2
      ;;
    --extract)
      EXTRACT_TARBALL="true"
      shift
      ;;
    --keep-tarball)
      KEEP_TARBALL="true"
      shift
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

if [[ "${METHOD}" == "gsutil" ]]; then
  require_command "gsutil"

  if [[ -n "${SUBPATH}" ]]; then
    SOURCE_PATH="${DATASET_BUCKET}/${SUBPATH}"
  else
    SOURCE_PATH="${DATASET_BUCKET}"
  fi

  echo "Downloading Nutrition5K from ${SOURCE_PATH} into ${OUTPUT_DIR}"
  gsutil -m cp -r "${SOURCE_PATH}" "${`OUTPUT_DIR`}"
  exit 0
fi

if [[ "${METHOD}" == "tarball" ]]; then
  require_command "curl"

  TARBALL_PATH="${OUTPUT_DIR}/nutrition5k_dataset.tar.gz"
  echo "Downloading Nutrition5K tarball to ${TARBALL_PATH}"
  curl -L -C - --fail --output "${TARBALL_PATH}" "${TARBALL_URL}"

  if [[ "${EXTRACT_TARBALL}" == "true" ]]; then
    require_command "tar"
    echo "Extracting ${TARBALL_PATH} into ${OUTPUT_DIR}"
    tar -xzf "${TARBALL_PATH}" -C "${OUTPUT_DIR}"

    if [[ "${KEEP_TARBALL}" != "true" ]]; then
      rm -f "${TARBALL_PATH}"
    fi
  fi
  exit 0
fi

echo "Unsupported method: ${METHOD}" >&2
print_usage >&2
exit 1
