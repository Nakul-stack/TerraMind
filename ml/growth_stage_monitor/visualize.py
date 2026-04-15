"""
Visualization script for the During Growth Monitor pipeline metrics.

Generates:
1. Confusion Matrix heatmaps for Classification Targets (Pest Level, Recommended Fertilizer).
2. Actual vs Predicted scatter plots for Regression Targets (Dosage, Apply After Days, Expected Yield).
3. Train vs Validation comparison bar charts for Accuracy and R² Score.

Saves all plots to the ``evaluation/`` directory.
"""

import sys
from pathlib import Path

# Allow running as ``python visualize.py`` from the module directory
if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ml.growth_stage_monitor import config
from ml.growth_stage_monitor.utils import get_logger, load_artifact, load_json
from ml.growth_stage_monitor.preprocessing import (
    load_dataset,
    normalise_columns,
    validate_and_clean,
    prepare_data,
)
from ml.growth_stage_monitor.model import _augment_features_for_stage2

logger = get_logger(__name__)

# Ensure evaluation directory exists
EVAL_DIR = Path(__file__).resolve().parent / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, labels, target_name):
    """Generate and save the confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Confusion Matrix - {target_name.replace('_', ' ').title()}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    
    out_path = EVAL_DIR / f"{target_name}_confusion_matrix.png"
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Confusion Matrix: %s", out_path.name)


def plot_actual_vs_predicted(y_true, y_pred, target_name):
    """Generate and save an Actual vs Predicted scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color="teal")
    
    # Ideal y=x trendline
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal")
    
    plt.title(f"Actual vs Predicted - {target_name.replace('_', ' ').title()}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    out_path = EVAL_DIR / f"{target_name}_actual_vs_predicted.png"
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Target Scatter Plot: %s", out_path.name)


def plot_metrics_comparison(summary_data):
    """Generate Train vs Validation comparison bar charts using the summary JSON."""
    # Split into classification (Accuracy) and regression (R2)
    cls_targets = []
    cls_train_acc = []
    cls_val_acc = []
    
    reg_targets = []
    reg_train_r2 = []
    reg_val_r2 = []
    
    for target, metrics in summary_data.items():
        if metrics["type"] == "classification":
            cls_targets.append(target.replace('_', ' ').title())
            cls_train_acc.append(metrics["train_accuracy"])
            cls_val_acc.append(metrics["val_accuracy"])
        elif metrics["type"] == "regression":
            reg_targets.append(target.replace('_', ' ').title())
            reg_train_r2.append(metrics["train_r2"])
            reg_val_r2.append(metrics["val_r2"])

    # Classification Plot
    if cls_targets:
        x = np.arange(len(cls_targets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, cls_train_acc, width, label='Train Accuracy', color='skyblue')
        rects2 = ax.bar(x + width/2, cls_val_acc, width, label='Validation Accuracy', color='salmon')

        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Performance (Accuracy)')
        ax.set_xticks(x)
        ax.set_xticklabels(cls_targets)
        ax.legend()
        ax.set_ylim([0, 1.1])
        
        # Add labels
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        out_path = EVAL_DIR / "classification_metrics_comparison.png"
        plt.savefig(out_path)
        plt.close()
        logger.info("Saved Classification Comparison: %s", out_path.name)

    # Regression Plot
    if reg_targets:
        x = np.arange(len(reg_targets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, reg_train_r2, width, label='Train R²', color='mediumseagreen')
        rects2 = ax.bar(x + width/2, reg_val_r2, width, label='Validation R²', color='coral')

        ax.set_ylabel('R² Score')
        ax.set_title('Regression Performance (R²)')
        ax.set_xticks(x)
        ax.set_xticklabels(reg_targets)
        ax.legend()
        ax.set_ylim([0, 1.1])
        
        # Add labels
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        out_path = EVAL_DIR / "regression_metrics_comparison.png"
        plt.savefig(out_path)
        plt.close()
        logger.info("Saved Regression Comparison: %s", out_path.name)


def main():
    logger.info("Starting visualization process...")
    
    # 1. Plot aggregate JSON metrics first
    summary_data = load_json(config.EVALUATION_SUMMARY_PATH)
    plot_metrics_comparison(summary_data)
    
    # 2. Load Models & Transformation Artifacts
    logger.info("Loading modeling artifacts...")
    preprocessor = load_artifact(config.PREPROCESSING_PIPELINE_PATH)
    pest_level_encoder = load_artifact(config.PEST_LEVEL_ENCODER_PATH)
    fertilizer_encoder = load_artifact(config.FERTILIZER_ENCODER_PATH)
    
    models = {
        "pest_level": load_artifact(config.PEST_LEVEL_MODEL_PATH),
        "recommended_fertilizer": load_artifact(config.FERTILIZER_MODEL_PATH),
        "dosage": load_artifact(config.DOSAGE_MODEL_PATH),
        "apply_after_days": load_artifact(config.APPLY_AFTER_DAYS_MODEL_PATH),
        "expected_yield_after_dosage": load_artifact(config.EXPECTED_YIELD_MODEL_PATH),
    }

    # 3. Load & Process Dataset for Predictions
    df = load_dataset()
    df = normalise_columns(df)
    df = validate_and_clean(df)
    
    data = prepare_data(
        df, 
        fit=False, 
        preprocessor=preprocessor,
        pest_level_encoder=pest_level_encoder,
        fertilizer_encoder=fertilizer_encoder
    )
    
    X_val = data["X_val"]
    y_val = data["y_val"]

    # 4. Generate Target Plots
    logger.info("Generating evaluation plots...")

    # --- Stage 1: Classification Targets ---
    pest_pred = models["pest_level"].predict(X_val)
    plot_confusion_matrix(
        y_val["pest_level"], 
        pest_pred, 
        list(pest_level_encoder.classes_), 
        "pest_level"
    )

    fert_pred = models["recommended_fertilizer"].predict(X_val)
    plot_confusion_matrix(
        y_val["recommended_fertilizer"], 
        fert_pred, 
        list(fertilizer_encoder.classes_), 
        "recommended_fertilizer"
    )

    # --- Stage 2: Regression Targets ---
    # Use *actual* validation class values mapped for Stage 2 augmentation
    X_val_aug = _augment_features_for_stage2(
        X_val, 
        y_val["pest_level"], 
        y_val["recommended_fertilizer"]
    )
    
    for target in config.REGRESSION_TARGETS:
        target_pred = models[target].predict(X_val_aug)
        plot_actual_vs_predicted(y_val[target], target_pred, target)

    # Done
    logger.info("\n==========================")
    logger.info(" All plots generated and saved to: %s", EVAL_DIR)
    logger.info("==========================")


if __name__ == "__main__":
    main()
