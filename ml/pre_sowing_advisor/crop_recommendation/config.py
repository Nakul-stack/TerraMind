"""
Configuration for Model 1 — Crop Recommender.

Training data : crop_dataset_rebuilt.csv
Features (10) : N, P, K, temperature, humidity, ph, rainfall
                + engineered: NP_ratio, KP_ratio, rainfall_humidity
Target        : label  (crop name)
Models tried  : RandomForestClassifier + GradientBoostingClassifier → keep best

Recall improvements:
    - Merge 'paddy' → 'rice' (identical crops causing confusion)
    - Add 3 interaction features for better class separation
    - Increase estimators, lower learning rate for better generalization
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = PROJECT_ROOT / "dataset before sowing"
DATASET_FILE = DATASET_DIR / "crop_dataset_rebuilt.csv"

SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_PATH = SAVED_MODELS_DIR / "model.pkl"
SCALER_PATH = SAVED_MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = SAVED_MODELS_DIR / "label_encoder.pkl"
METADATA_PATH = SAVED_MODELS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Feature / Target
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
]
# Engineered features added during preprocessing
ENGINEERED_FEATURES: list[str] = [
    "NP_ratio", "KP_ratio", "rainfall_humidity",
]
ALL_FEATURES: list[str] = FEATURE_COLUMNS + ENGINEERED_FEATURES
TARGET_COLUMN: str = "label"

# Label merging — paddy and rice are the same crop in Indian agriculture
LABEL_MERGE_MAP: dict[str, str] = {
    "paddy": "rice",
}

# ---------------------------------------------------------------------------
# Training Hyper-parameters
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.10          # carved from train for model comparison
RANDOM_STATE: int = 42

# RandomForest
RF_N_ESTIMATORS: int = 500
RF_MAX_DEPTH: int | None = None
RF_MIN_SAMPLES_SPLIT: int = 2
RF_MIN_SAMPLES_LEAF: int = 1

# GradientBoosting
GB_N_ESTIMATORS: int = 500
GB_MAX_DEPTH: int = 6
GB_LEARNING_RATE: float = 0.05
GB_MIN_SAMPLES_LEAF: int = 3
GB_SUBSAMPLE: float = 0.85
