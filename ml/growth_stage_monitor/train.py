import sys
from pathlib import Path

# Fix relative imports when executing directly
if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from ml.growth_stage_monitor.utils import get_logger
from ml.growth_stage_monitor.preprocessing import (
    load_dataset,
    normalise_columns,
    validate_and_clean,
    prepare_data
)
from ml.growth_stage_monitor.model import build_all_models, train_all_models
from ml.growth_stage_monitor.evaluation import evaluate_all_models

logger = get_logger(__name__)

def main():
    logger.info("Initializing During Growth Monitor Training Pipeline (2-Stage)...")
    
    # 1. Data Ingestion
    logger.info("Loading dataset...")
    df = load_dataset()
    logger.info(f"Loaded {len(df)} records.")
    
    # 2. Preprocessing & Validation
    logger.info("Cleaning & validating data...")
    df = normalise_columns(df)
    df = validate_and_clean(df)
    
    # 3. Preparation & Splitting
    logger.info("Encoding targets & building preprocessor...")
    data_artifacts = prepare_data(df, fit=True)
    X_train = data_artifacts["X_train"]
    X_val = data_artifacts["X_val"]
    y_train = data_artifacts["y_train"]
    y_val = data_artifacts["y_val"]
    
    # 4. Model Building & Training
    logger.info("Initializing models across 2-Stage architecture...")
    models = build_all_models()
    
    logger.info("Training models & saving artifacts...")
    models = train_all_models(models, X_train, y_train)
    
    # 5. Evaluation
    evaluate_all_models(models, X_train, y_train, X_val, y_val)
    
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
