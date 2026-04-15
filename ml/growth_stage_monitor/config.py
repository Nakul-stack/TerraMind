"""
Configuration module for the During Growth Monitor pipeline.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
DATASET_FILE = BASE_DIR.parents[1] / "dataset during growth" / "fertilizer_giant_training_dataset.csv"

# Ensure output dir exists
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Artifact paths
PREPROCESSING_PIPELINE_PATH = SAVED_MODELS_DIR / "preprocessing_pipeline.pkl"
PEST_LEVEL_ENCODER_PATH = SAVED_MODELS_DIR / "pest_level_encoder.pkl"
FERTILIZER_ENCODER_PATH = SAVED_MODELS_DIR / "fertilizer_encoder.pkl"

PEST_LEVEL_MODEL_PATH = SAVED_MODELS_DIR / "pest_level_model.pkl"
FERTILIZER_MODEL_PATH = SAVED_MODELS_DIR / "recommended_fertilizer_model.pkl"
DOSAGE_MODEL_PATH = SAVED_MODELS_DIR / "dosage_model.pkl"
APPLY_AFTER_DAYS_MODEL_PATH = SAVED_MODELS_DIR / "apply_after_days_model.pkl"
EXPECTED_YIELD_MODEL_PATH = SAVED_MODELS_DIR / "expected_yield_model.pkl"

METADATA_PATH = SAVED_MODELS_DIR / "metadata.json"
EVALUATION_SUMMARY_PATH = SAVED_MODELS_DIR / "evaluation_summary.json"

# Features and Targets
FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "moisture",
    "soil_type",
    "crop_type",
    "N",
    "P",
    "K",
    "ph",
    "rainfall",
]

CLASSIFICATION_TARGETS = [
    "pest_level",
    "recommended_fertilizer",
]

REGRESSION_TARGETS = [
    "dosage",
    "apply_after_days",
    "expected_yield_after_dosage",
]

ALL_TARGETS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS

# Column Mapping to handle unit and symbol stripping
COLUMN_MAPPING = {
    "Temperature (°C)": "temperature",
    "Humidity (%)": "humidity",
    "Moisture (%)": "moisture",
    "Soil Type": "soil_type",
    "Crop Type": "crop_type",
    "Nitrogen (ppm)": "N",
    "Phosphorus (ppm)": "P",
    "Potassium (ppm)": "K",
    "pH": "ph",
    "Rainfall (mm)": "rainfall",
    "Pest Level": "pest_level",
    "Recommended Fertilizer": "recommended_fertilizer",
    "Dosage (kg/acre)": "dosage",
    "Apply_After_Days": "apply_after_days",
    "Expected Yield (q/ha)": "expected_yield_after_dosage",
}

# Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
