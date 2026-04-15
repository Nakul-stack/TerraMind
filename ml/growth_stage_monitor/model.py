import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from . import config
from .utils import get_logger, save_artifact

logger = get_logger(__name__)

def build_all_models() -> dict:
    return {
        "pest_level": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=config.RANDOM_STATE, class_weight="balanced"),
        "recommended_fertilizer": RandomForestClassifier(n_estimators=250, max_depth=12, random_state=config.RANDOM_STATE, class_weight="balanced"),
        "dosage": GradientBoostingRegressor(n_estimators=250, learning_rate=0.08, max_depth=8, random_state=config.RANDOM_STATE),
        "apply_after_days": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=8, random_state=config.RANDOM_STATE),
        "expected_yield_after_dosage": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=config.RANDOM_STATE)
    }

def _augment_features_for_stage2(X: np.ndarray, y_pest: np.ndarray, y_fert: np.ndarray) -> np.ndarray:
    return np.column_stack([X, y_pest.reshape(-1, 1), y_fert.reshape(-1, 1)])

def train_all_models(models: dict, X_train: np.ndarray, y_train: dict) -> dict:
    models["pest_level"].fit(X_train, y_train["pest_level"])
    save_artifact(models["pest_level"], config.PEST_LEVEL_MODEL_PATH)
    
    models["recommended_fertilizer"].fit(X_train, y_train["recommended_fertilizer"])
    save_artifact(models["recommended_fertilizer"], config.FERTILIZER_MODEL_PATH)
    
    X_train_aug = _augment_features_for_stage2(X_train, y_train["pest_level"], y_train["recommended_fertilizer"])
    
    for target in config.REGRESSION_TARGETS:
        models[target].fit(X_train_aug, y_train[target])
        if target == "dosage":
            save_artifact(models[target], config.DOSAGE_MODEL_PATH)
        elif target == "apply_after_days":
            save_artifact(models[target], config.APPLY_AFTER_DAYS_MODEL_PATH)
        elif target == "expected_yield_after_dosage":
            save_artifact(models[target], config.EXPECTED_YIELD_MODEL_PATH)
            
    return models
