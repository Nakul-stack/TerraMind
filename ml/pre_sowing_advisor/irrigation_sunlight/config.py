"""
Configuration for Model 3 — Pre-Sowing Advisor (Irrigation & Sunlight).

Training data: irrigation_prediction.csv
Inputs (7+4):  crop, ph, temperature, humidity, rainfall, soil_type, season
               + engineered: temp_rainfall_ratio, humidity_rainfall,
                             rainfall_log, temp_humidity_interaction
Outputs (3):   sunlight_hours (regression), irrigation_type (classification),
               irrigation_need (classification)

Accuracy improvements:
    - Feature engineering: 4 interaction features to help classifiers separate
    - class_weight='balanced' on both classifiers
    - GradientBoosting classifiers instead of RandomForest for irrigation type
    - Increased estimators with lower learning rate
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = PROJECT_ROOT / "dataset before sowing"
DATASET_FILE = DATASET_DIR / "irrigation_prediction.csv"

# ICRISAT datasets for district irrigation prior (NOT for training Model 3)
ICRISAT_SOURCE_FILE = DATASET_DIR / "ICRISAT-District Level Data Source.csv"
ICRISAT_IRRIGATION_FILE = DATASET_DIR / "ICRISAT-District Level Data Irrigation.csv"

SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
PREPROCESSOR_PATH = SAVED_MODELS_DIR / "preprocessor.pkl"
MODEL_SUNLIGHT_PATH = SAVED_MODELS_DIR / "model_sunlight.pkl"
MODEL_IRR_TYPE_PATH = SAVED_MODELS_DIR / "model_irrigation_type.pkl"
MODEL_IRR_NEED_PATH = SAVED_MODELS_DIR / "model_irrigation_need.pkl"
METADATA_PATH = SAVED_MODELS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Column Configuration
# ---------------------------------------------------------------------------
RAW_CROP_COL: str = "crop"
RAW_PH_COL: str = "ph"
RAW_TEMP_COL: str = "temperature"
RAW_HUMIDITY_COL: str = "humidity"
RAW_RAINFALL_COL: str = "rainfall"
RAW_SOIL_COL: str = "soil_type"
RAW_SEASON_COL: str = "season"

RAW_SUNLIGHT_COL: str = "Sunlight_Hours"
RAW_IRR_TYPE_COL: str = "irrigation_type"
RAW_IRR_NEED_COL: str = "irrigation_need"

NUMERIC_FEATURES: list[str] = [RAW_PH_COL, RAW_TEMP_COL, RAW_HUMIDITY_COL, RAW_RAINFALL_COL]
CATEGORICAL_FEATURES: list[str] = [RAW_CROP_COL, RAW_SOIL_COL, RAW_SEASON_COL]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Engineered numeric features (added during preprocessing)
ENGINEERED_NUMERIC_FEATURES: list[str] = [
    "temp_rainfall_ratio",
    "humidity_rainfall",
    "rainfall_log",
    "temp_humidity_interaction",
]

TARGET_COLUMNS: list[str] = ["sunlight_hours", RAW_IRR_TYPE_COL, RAW_IRR_NEED_COL]

# ---------------------------------------------------------------------------
# Training Hyper-parameters
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42

# Sunlight — GradientBoostingRegressor
GBR_SUN_N_ESTIMATORS: int = 300
GBR_SUN_MAX_DEPTH: int = 6
GBR_SUN_LEARNING_RATE: float = 0.05
GBR_SUN_MIN_SAMPLES_LEAF: int = 8
GBR_SUN_SUBSAMPLE: float = 0.85

# Irrigation Type — GradientBoostingClassifier (replaced RF for better accuracy)
GB_CLF_TYPE_N_ESTIMATORS: int = 500
GB_CLF_TYPE_MAX_DEPTH: int = 6
GB_CLF_TYPE_LEARNING_RATE: float = 0.05
GB_CLF_TYPE_MIN_SAMPLES_LEAF: int = 5
GB_CLF_TYPE_SUBSAMPLE: float = 0.85

# Irrigation Need — GradientBoostingClassifier (replaced RF for better accuracy)
GB_CLF_NEED_N_ESTIMATORS: int = 500
GB_CLF_NEED_MAX_DEPTH: int = 6
GB_CLF_NEED_LEARNING_RATE: float = 0.05
GB_CLF_NEED_MIN_SAMPLES_LEAF: int = 5
GB_CLF_NEED_SUBSAMPLE: float = 0.85

# Backup: Extra Trees (high-recall ensemble)
ET_CLF_N_ESTIMATORS: int = 500
ET_CLF_MAX_DEPTH: int | None = None
