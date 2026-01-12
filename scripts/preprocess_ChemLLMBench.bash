#!/usr/bin/env bash
set -euo pipefail

# =========================
# Variables
# =========================
PREPROCESS_DIR=<preprocess>
DATA_DIR=<raw_data>
# suggested to end with .../ChemLLMBench/
OUTPUT_DIR=<output_data>

# =========================
# Name Prediction
# =========================
python "${PREPROCESS_DIR}/name_prediction.py" \
  --csv_path "${DATA_DIR}/name_prediction/llm_test.csv" \
  --output_dir "${OUTPUT_DIR}/name_prediction"

# =========================
# Property Prediction
# =========================
python "${PREPROCESS_DIR}/property_prediction.py" \
  --csv_dir "${DATA_DIR}/property_prediction" \
  --output_dir "${OUTPUT_DIR}/property_prediction"

# =========================
# Molecule Design
# =========================
python "${PREPROCESS_DIR}/molecule_design.py" \
  --csv_path "${DATA_DIR}/molecule_design/molecule_design_test.csv" \
  --output_dir "${OUTPUT_DIR}/molecule_design"

# =========================
# Molecule Captioning
# =========================
python "${PREPROCESS_DIR}/molecule_captioning.py" \
  --csv_path "${DATA_DIR}/molecule_captioning/molecule_captioning_test.csv" \
  --output_dir "${OUTPUT_DIR}/molecule_captioning"

# =========================
# Yield Prediction
# =========================
python "${PREPROCESS_DIR}/yield_prediction.py" \
  --npz_dir "${DATA_DIR}/yield_prediction" \
  --output_dir "${OUTPUT_DIR}/yield_prediction"

# =========================
# Reagent Selection
# =========================
python "${PREPROCESS_DIR}/reagent_selection.py" \
  --csv_dir "${DATA_DIR}/reagent_selection" \
  --output_dir "${OUTPUT_DIR}/reagent_selection"

# =========================
# Reaction Prediction
# =========================
python "${PREPROCESS_DIR}/reaction_prediction.py" \
  --csv_path "${DATA_DIR}/reaction_prediction/uspto_test.csv" \
  --output_dir "${OUTPUT_DIR}/reaction_prediction"

# =========================
# Retrosynthesis
# =========================
python "${PREPROCESS_DIR}/retro.py" \
  --csv_path "${DATA_DIR}/retro/uspto50k_retro_test.csv" \
  --output_dir "${OUTPUT_DIR}/retro"
