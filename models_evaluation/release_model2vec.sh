#!/bin/bash
# Prepare the release of the Model2Vec models from a finished training grid, in one command.
#
# Usage: models_evaluation/release_model2vec.sh CHECKPOINTS_DIR CACHE_DIR TEST_PATH OUTPUT_DIR
#
# 1. Select the released configuration and seed of each model (fails if a configuration misses runs).
# 2. Export the selected models to the Hugging Face Hub format, with their model cards.
# 3. Check them end to end (load, same predictions as their checkpoints, retrain and reload), without network.
# 4. Write the Results tables of the documentation in OUTPUT_DIR (to copy in docs/source).
#
# Set ALLOW_INCOMPLETE=1 to preview the release on an unfinished grid.
#
# Nothing is pushed: once the output is reviewed, push with
#   python -m models_evaluation.export_model_to_hub OUTPUT_DIR/selection.json --output_dir OUTPUT_DIR/hub_export --push
set -euo pipefail

if [ "$#" -ne 4 ]; then
  sed -n '2,14p' "$0"
  exit 2
fi
CHECKPOINTS_DIR=$1
CACHE_DIR=$2
TEST_PATH=$3
OUTPUT_DIR=$4
mkdir -p "$OUTPUT_DIR"

echo "== 1. Selection"
python -m models_evaluation.select_best_model "$CHECKPOINTS_DIR" --output "$OUTPUT_DIR/selection.json" \
  --rst_table "$OUTPUT_DIR/model2vec_grid.inc" --rst_selection_table "$OUTPUT_DIR/model2vec_selection.inc" ${ALLOW_INCOMPLETE:+--allow_incomplete}

echo "== 2. Export"
rm -rf "$OUTPUT_DIR/hub_export"
python -m models_evaluation.export_model_to_hub "$OUTPUT_DIR/selection.json" --output_dir "$OUTPUT_DIR/hub_export"

echo "== 3. End-to-end check"
python -m models_evaluation.check_exported_models "$OUTPUT_DIR/selection.json" "$OUTPUT_DIR/hub_export" \
  "$CACHE_DIR" "$TEST_PATH"

echo "== 4. Documentation tables"
ls "$OUTPUT_DIR"/model2vec_*.inc
echo "Ready to review: $OUTPUT_DIR (copy the .inc files to docs/source, then push with --push)."
