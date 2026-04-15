import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from . import config
from .utils import get_logger, save_artifact

logger = get_logger(__name__)

def load_dataset() -> pd.DataFrame:
    return pd.read_csv(config.DATASET_FILE)

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=config.COLUMN_MAPPING)
    return df

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_data(df: pd.DataFrame, fit: bool = True, preprocessor=None, pest_level_encoder=None, fertilizer_encoder=None) -> dict:
    X = df[config.FEATURE_COLUMNS]
    
    cat_cols = ["soil_type", "crop_type"]
    num_cols = ["temperature", "humidity", "moisture", "N", "P", "K", "ph", "rainfall"]
    
    if fit:
        num_transformer = StandardScaler()
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        preprocessor = ColumnTransformer(transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ])
        X_trans = preprocessor.fit_transform(X)
        save_artifact(preprocessor, config.PREPROCESSING_PIPELINE_PATH)
        
        pest_level_encoder = LabelEncoder()
        y_pest = pest_level_encoder.fit_transform(df["pest_level"])
        save_artifact(pest_level_encoder, config.PEST_LEVEL_ENCODER_PATH)
        
        fertilizer_encoder = LabelEncoder()
        y_fert = fertilizer_encoder.fit_transform(df["recommended_fertilizer"])
        save_artifact(fertilizer_encoder, config.FERTILIZER_ENCODER_PATH)
        
        y_dos = df["dosage"].values
        y_days = df["apply_after_days"].values
        y_yld = df["expected_yield_after_dosage"].values
        
        y_all = np.column_stack([y_pest, y_fert, y_dos, y_days, y_yld])
        
        X_train, X_val, y_train_full, y_val_full = train_test_split(X_trans, y_all, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y_all[:, :2])
        
        y_train = {
            "pest_level": y_train_full[:, 0],
            "recommended_fertilizer": y_train_full[:, 1],
            "dosage": y_train_full[:, 2],
            "apply_after_days": y_train_full[:, 3],
            "expected_yield_after_dosage": y_train_full[:, 4],
        }
        y_val = {
            "pest_level": y_val_full[:, 0],
            "recommended_fertilizer": y_val_full[:, 1],
            "dosage": y_val_full[:, 2],
            "apply_after_days": y_val_full[:, 3],
            "expected_yield_after_dosage": y_val_full[:, 4],
        }
    else:
        X_trans = preprocessor.transform(X)
        
        # Inference mode handling
        if "pest_level" in df.columns:
            y_pest = pest_level_encoder.transform(df["pest_level"])
            y_fert = fertilizer_encoder.transform(df["recommended_fertilizer"])
            y_dos = df["dosage"].values
            y_days = df["apply_after_days"].values
            y_yld = df["expected_yield_after_dosage"].values
            y_train, X_train = None, None
            y_val = {
                "pest_level": y_pest,
                "recommended_fertilizer": y_fert,
                "dosage": y_dos,
                "apply_after_days": y_days,
                "expected_yield_after_dosage": y_yld,
            }
            X_val = X_trans
        else:
            return X_trans
            
    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "preprocessor": preprocessor,
        "pest_level_encoder": pest_level_encoder,
        "fertilizer_encoder": fertilizer_encoder
    }
    
def transform_input(input_dict: dict, preprocessor) -> np.ndarray:
    df = pd.DataFrame([input_dict])
    return preprocessor.transform(df)
