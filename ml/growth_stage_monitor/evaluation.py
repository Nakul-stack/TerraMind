import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score, mean_absolute_error, mean_squared_error

from . import config
from .utils import get_logger, save_json

logger = get_logger(__name__)

def evaluate_classifier(y_true: pd.Series, y_pred: np.ndarray, y_train_true=None, y_train_pred=None) -> dict:
    val_acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    result = {
        "val_accuracy": val_acc,
        "val_precision": prec,
        "val_recall": rec,
        "val_f1": f1,
        "type": "classification"
    }
    
    if y_train_true is not None and y_train_pred is not None:
        train_acc = accuracy_score(y_train_true, y_train_pred)
        result["train_accuracy"] = train_acc
        
        # Check for massive overfitting
        if train_acc - val_acc > 0.15:
            result["health"] = "OVERFIT"
        else:
            result["health"] = "HEALTHY"
            
    return result

def evaluate_regressor(y_true: pd.Series, y_pred: np.ndarray, y_train_true=None, y_train_pred=None) -> dict:
    val_r2 = r2_score(y_true, y_pred)
    val_mae = mean_absolute_error(y_true, y_pred)
    val_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    result = {
        "val_r2": val_r2,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "type": "regression"
    }
    
    if y_train_true is not None and y_train_pred is not None:
        train_r2 = r2_score(y_train_true, y_train_pred)
        result["train_r2"] = train_r2
        
        if train_r2 - val_r2 > 0.15:
            result["health"] = "OVERFIT"
        else:
            result["health"] = "HEALTHY"
            
    return result

def evaluate_all_models(models: dict, X_train: np.ndarray, y_train: dict, X_val: np.ndarray, y_val: dict) -> dict:
    logger.info("Starting performance evaluation for all targets...")
    summary = {}
    
    # 1. Classification Targets
    logger.info("\n--- Classification Performance ---")
    for target in config.CLASSIFICATION_TARGETS:
        model = models[target]
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        res = evaluate_classifier(y_val[target], val_pred, y_train[target], train_pred)
        summary[target] = res
        
        health_flag = f" [! {res['health']} !]" if res.get('health') == 'OVERFIT' else ""
        logger.info(f"{target.upper()}{health_flag}: ")
        logger.info(f"   Train Acc : {res.get('train_accuracy', 0):.4f}")
        logger.info(f"   Val Acc   : {res['val_accuracy']:.4f}")

    # 2. Regression Targets
    # Stage 2 regression needs augmented features
    X_train_aug = np.column_stack([X_train, y_train["pest_level"].reshape(-1, 1), y_train["recommended_fertilizer"].reshape(-1, 1)])
    X_val_aug = np.column_stack([X_val, y_val["pest_level"].reshape(-1, 1), y_val["recommended_fertilizer"].reshape(-1, 1)])
    
    logger.info("\n--- Regression Performance ---")
    for target in config.REGRESSION_TARGETS:
        model = models[target]
        train_pred = model.predict(X_train_aug)
        val_pred = model.predict(X_val_aug)
        
        res = evaluate_regressor(y_val[target], val_pred, y_train[target], train_pred)
        summary[target] = res
        
        health_flag = f" [! {res['health']} !]" if res.get('health') == 'OVERFIT' else ""
        logger.info(f"{target.upper()}{health_flag}: ")
        logger.info(f"   Train R2 : {res.get('train_r2', 0):.4f}")
        logger.info(f"   Val R2   : {res['val_r2']:.4f}")
        logger.info(f"   Val MAE  : {res['val_mae']:.4f}")
        
    save_json(summary, config.EVALUATION_SUMMARY_PATH)
    logger.info(f"\nEvaluation summary saved to: {config.EVALUATION_SUMMARY_PATH}")
    return summary
