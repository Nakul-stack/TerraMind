"""
Configuration for Model 2 — Yield Predictor.

Training data:
    Primary   : crop_production.csv.xlsx
    Secondary : India Agriculture Crop Production.csv (merged for enrichment)

Features : crop, state, district, season, area (optional)
Target   : yield (tons per hectare where derivable, else production/area)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = PROJECT_ROOT / "dataset before sowing"
PRIMARY_DATASET = DATASET_DIR / "crop_production.csv.xlsx"
SECONDARY_DATASET = DATASET_DIR / "India Agriculture Crop Production.csv"
ICRISAT_DISTRICT = DATASET_DIR / "ICRISAT-District Level Data.csv"

SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_PATH = SAVED_MODELS_DIR / "model.pkl"
PREPROCESSOR_PATH = SAVED_MODELS_DIR / "preprocessor.pkl"
RESIDUAL_STD_PATH = SAVED_MODELS_DIR / "residual_std.pkl"
METADATA_PATH = SAVED_MODELS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Column Configuration — Primary dataset (crop_production.csv.xlsx)
# ---------------------------------------------------------------------------
RAW_STATE_COL: str = "State_Name"
RAW_DISTRICT_COL: str = "District_Name"
RAW_YEAR_COL: str = "Crop_Year"
RAW_SEASON_COL: str = "Season"
RAW_CROP_COL: str = "label"
RAW_AREA_COL: str = "Area"
RAW_PRODUCTION_COL: str = "Production"

# Column Configuration — Secondary dataset (India Agriculture Crop Production.csv)
SEC_STATE_COL: str = "State"
SEC_DISTRICT_COL: str = "District"
SEC_YEAR_COL: str = "Year"
SEC_SEASON_COL: str = "Season"
SEC_CROP_COL: str = "Crop"
SEC_AREA_COL: str = "Area"
SEC_PRODUCTION_COL: str = "Production"
SEC_YIELD_COL: str = "Yield"

# Engineered target
TARGET_COL: str = "yield"

# Features used for modelling
FEATURE_COLUMNS: list[str] = [
    "crop", "state", "district", "season",
]

# ---------------------------------------------------------------------------
# Training Hyper-parameters
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42

# XGBoost (primary)
XGB_N_ESTIMATORS: int = 300
XGB_MAX_DEPTH: int = 10
XGB_LEARNING_RATE: float = 0.1
XGB_MIN_CHILD_WEIGHT: int = 10

# RandomForest (fallback)
RF_N_ESTIMATORS: int = 200
RF_MAX_DEPTH: int = 20
RF_MIN_SAMPLES_SPLIT: int = 10
RF_MIN_SAMPLES_LEAF: int = 5

# Confidence band
CONFIDENCE_MULTIPLIER: float = 1.5  # residual_std * multiplier for ~87% coverage
