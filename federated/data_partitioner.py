"""Data partitioning and dataset preparation utilities for TerraMind federated simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

STATE_PROFILES = {
    "Andhra_Pradesh": {
        "dominant_crops": ["rice", "cotton", "sugarcane", "groundnut", "maize"],
        "climate": "tropical",
    },
    "Arunachal_Pradesh": {
        "dominant_crops": ["rice", "maize", "millet", "wheat"],
        "climate": "subtropical",
    },
    "Assam": {
        "dominant_crops": ["rice", "jute", "banana", "maize"],
        "climate": "humid_subtropical",
    },
    "Bihar": {
        "dominant_crops": ["rice", "wheat", "maize", "sugarcane", "lentil"],
        "climate": "humid_subtropical",
    },
    "Chhattisgarh": {
        "dominant_crops": ["rice", "maize", "soybean", "wheat"],
        "climate": "sub_humid",
    },
    "Goa": {
        "dominant_crops": ["rice", "coconut", "banana"],
        "climate": "tropical",
    },
    "Gujarat": {
        "dominant_crops": ["cotton", "groundnut", "wheat", "bajra", "sugarcane"],
        "climate": "semi_arid",
    },
    "Haryana": {
        "dominant_crops": ["wheat", "rice", "cotton", "sugarcane", "maize"],
        "climate": "semi_arid",
    },
    "Himachal_Pradesh": {
        "dominant_crops": ["wheat", "maize", "rice", "potato"],
        "climate": "temperate",
    },
    "Jharkhand": {
        "dominant_crops": ["rice", "maize", "wheat", "potato"],
        "climate": "humid_subtropical",
    },
    "Karnataka": {
        "dominant_crops": ["rice", "cotton", "sugarcane", "coffee", "maize", "jowar"],
        "climate": "tropical",
    },
    "Kerala": {
        "dominant_crops": ["rice", "coconut", "banana", "coffee"],
        "climate": "tropical",
    },
    "Madhya_Pradesh": {
        "dominant_crops": ["soybean", "wheat", "rice", "maize", "cotton"],
        "climate": "sub_humid",
    },
    "Maharashtra": {
        "dominant_crops": ["cotton", "sugarcane", "soybean", "jowar", "rice"],
        "climate": "semi_arid",
    },
    "Manipur": {
        "dominant_crops": ["rice", "maize"],
        "climate": "subtropical",
    },
    "Meghalaya": {
        "dominant_crops": ["rice", "maize", "potato"],
        "climate": "subtropical",
    },
    "Mizoram": {
        "dominant_crops": ["rice", "maize"],
        "climate": "subtropical",
    },
    "Nagaland": {
        "dominant_crops": ["rice", "maize"],
        "climate": "subtropical",
    },
    "Odisha": {
        "dominant_crops": ["rice", "jute", "potato", "mustard", "maize"],
        "climate": "humid_subtropical",
    },
    "Punjab": {
        "dominant_crops": ["wheat", "rice", "maize", "cotton"],
        "climate": "semi_arid",
    },
    "Rajasthan": {
        "dominant_crops": ["bajra", "jowar", "cotton", "wheat", "groundnut"],
        "climate": "arid",
    },
    "Sikkim": {
        "dominant_crops": ["rice", "maize"],
        "climate": "temperate",
    },
    "Tamil_Nadu": {
        "dominant_crops": ["rice", "cotton", "sugarcane", "banana", "groundnut"],
        "climate": "tropical",
    },
    "Telangana": {
        "dominant_crops": ["rice", "cotton", "maize", "sugarcane"],
        "climate": "tropical",
    },
    "Tripura": {
        "dominant_crops": ["rice", "jute"],
        "climate": "humid_subtropical",
    },
    "Uttar_Pradesh": {
        "dominant_crops": ["wheat", "rice", "sugarcane", "potato", "maize", "lentil"],
        "climate": "humid_subtropical",
    },
    "Uttarakhand": {
        "dominant_crops": ["wheat", "rice", "soybean", "maize"],
        "climate": "temperate",
    },
    "West_Bengal": {
        "dominant_crops": ["rice", "jute", "potato", "mustard", "wheat"],
        "climate": "humid_subtropical",
    },
}


FEATURE_COLUMNS = ["n", "p", "k", "ph", "temperature", "humidity", "rainfall"]
REQUIRED_PRIMARY_COLUMNS = FEATURE_COLUMNS + ["label"]


class FarmDataset(Dataset):
    """Torch dataset for TerraMind multi-task training."""

    def __init__(
        self,
        X: np.ndarray,
        y_crop: np.ndarray,
        y_yield: np.ndarray,
        y_sunlight: np.ndarray,
        y_irr_type: np.ndarray,
        y_irr_needed: np.ndarray,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_crop = torch.tensor(y_crop, dtype=torch.long)
        self.y_yield = torch.tensor(y_yield, dtype=torch.float32)
        self.y_sunlight = torch.tensor(y_sunlight, dtype=torch.float32)
        self.y_irr_type = torch.tensor(y_irr_type, dtype=torch.long)
        self.y_irr_needed = torch.tensor(y_irr_needed, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            self.X[idx],
            self.y_crop[idx],
            self.y_yield[idx],
            self.y_sunlight[idx],
            self.y_irr_type[idx],
            self.y_irr_needed[idx],
        )


@dataclass
class _IrrigationColumns:
    irrigation_type_col: Optional[str]
    irrigation_needed_col: Optional[str]
    crop_col: Optional[str]


class DataPartitioner:
    """Load datasets, generate targets, and create FL client partitions."""

    def __init__(self, data_path: str, num_clients: int = 28, random_seed: int = 42) -> None:
        self.data_path = data_path
        self.random_seed = random_seed
        self.num_clients = num_clients
        self.rng = np.random.default_rng(random_seed)

        self.results_dir = Path("federated/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.irrigation_type_encoder = LabelEncoder()

        self.crop_production_df: Optional[pd.DataFrame] = None
        self.irrigation_df: Optional[pd.DataFrame] = None
        self.irrigation_type_classes: List[str] = []
        self.state_metadata_list: List[Dict[str, object]] = []

    def load_datasets(self) -> pd.DataFrame:
        """Load primary dataset and optional auxiliary datasets."""
        primary_path = Path(self.data_path)
        if not primary_path.exists():
            primary_path = Path("dataset before sowing/crop_dataset_rebuilt.csv")

        if not primary_path.exists():
            raise FileNotFoundError(
                "Primary dataset not found. Expected at "
                "'dataset before sowing/crop_dataset_rebuilt.csv'"
            )

        df = pd.read_csv(primary_path)
        df.columns = [str(c).strip().lower() for c in df.columns]

        missing_cols = [col for col in REQUIRED_PRIMARY_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"[DataPartitioner] Missing required columns in primary dataset: {missing_cols}"
            )

        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

        if "yield" not in df.columns:
            simulated_yield = (
                df["n"] * 0.02
                + df["p"] * 0.015
                + df["k"] * 0.01
                + self.rng.normal(loc=2.5, scale=0.8, size=len(df))
            )
            df["yield"] = np.clip(simulated_yield, 0.5, 8.0)
            print("[DataPartitioner] yield column not found — simulating from soil features")
        else:
            df["yield"] = pd.to_numeric(df["yield"], errors="coerce").fillna(df["yield"].median())

        production_path = Path("dataset before sowing/crop_production_.csv.xlsx")
        self.crop_production_df = None
        if production_path.exists():
            try:
                self.crop_production_df = pd.read_excel(production_path)
            except Exception:
                try:
                    self.crop_production_df = pd.read_csv(production_path)
                except Exception:
                    print("[DataPartitioner] crop_production_ could not be loaded — skipping")
        else:
            print("[DataPartitioner] crop_production_ could not be loaded — skipping")

        irrigation_path = Path("dataset before sowing/irrigation_prediction.csv")
        self.irrigation_df = None
        if irrigation_path.exists():
            try:
                self.irrigation_df = pd.read_csv(irrigation_path)
                self.irrigation_df.columns = [
                    str(c).strip().lower() for c in self.irrigation_df.columns
                ]
            except Exception:
                print("[DataPartitioner] irrigation_prediction could not be loaded — skipping")
        else:
            print("[DataPartitioner] irrigation_prediction could not be loaded — skipping")

        df = self._generate_irrigation_targets(df)
        df = self._enrich_yield_from_production(df)

        return df.reset_index(drop=True)

    def _resolve_irrigation_columns(self) -> _IrrigationColumns:
        """Resolve potential irrigation column names from auxiliary dataset."""
        if self.irrigation_df is None:
            return _IrrigationColumns(None, None, None)

        columns = list(self.irrigation_df.columns)

        def first_match(candidates: List[str]) -> Optional[str]:
            for name in candidates:
                if name in columns:
                    return name
            return None

        irr_type_col = first_match(["irrigation_type", "irrigation_method", "type", "method"])
        irr_need_col = first_match(["irrigation_needed", "water_requirement", "irrigation_mm"])
        crop_col = first_match(["crop", "crop_name", "label"])

        found = {
            "irrigation_type": irr_type_col,
            "irrigation_needed": irr_need_col,
            "crop": crop_col,
        }
        print(f"[DataPartitioner] Irrigation columns found: {found}")

        return _IrrigationColumns(irr_type_col, irr_need_col, crop_col)

    def _simulate_sunlight(self, crop_series: pd.Series) -> np.ndarray:
        """Simulate sunlight hours by crop category with noise and clipping."""
        cereals = {"wheat", "rice", "maize"}
        vegetables = {"potato", "tomato"}
        fruits = {"banana", "mango"}
        legumes = {"lentil", "chickpea"}
        cash_crops = {"cotton", "jute"}

        values: List[float] = []
        for crop in crop_series.astype(str).str.lower():
            if crop in cereals:
                base = self.rng.uniform(6.0, 8.0)
            elif crop in vegetables:
                base = self.rng.uniform(5.0, 7.0)
            elif crop in fruits:
                base = self.rng.uniform(7.0, 10.0)
            elif crop in legumes:
                base = self.rng.uniform(6.0, 8.0)
            elif crop in cash_crops:
                base = self.rng.uniform(8.0, 10.0)
            else:
                base = 6.0
            values.append(base)

        noisy = np.array(values) + self.rng.normal(0.0, 0.3, size=len(values))
        return np.clip(noisy, 3.0, 12.0)

    def _simulate_irrigation_type(self, df: pd.DataFrame) -> List[str]:
        """Simulate irrigation type from rainfall and crop rules."""
        result: List[str] = []
        for _, row in df.iterrows():
            rainfall = float(row["rainfall"])
            crop = str(row["label"]).strip().lower()

            if crop == "rice":
                irr = "flood"
            elif rainfall < 500:
                irr = "drip"
            elif rainfall <= 1000:
                irr = "sprinkler"
            else:
                irr = "flood"
            result.append(irr)
        return result

    def _simulate_irrigation_needed(self, df: pd.DataFrame) -> np.ndarray:
        """Simulate irrigation needed (mm/day) with crop-based base values."""
        base_map = {
            "rice": 8.0,
            "wheat": 5.0,
            "cotton": 6.0,
            "sugarcane": 9.0,
            "maize": 5.5,
        }

        vals: List[float] = []
        for _, row in df.iterrows():
            crop = str(row["label"]).strip().lower()
            rainfall = float(row["rainfall"])
            base = base_map.get(crop, 5.0)
            actual = max(0.0, base - (rainfall / 365.0))
            vals.append(actual)

        noisy = np.array(vals) + self.rng.normal(0.0, 0.4, size=len(vals))
        return np.clip(noisy, 0.0, 20.0)

    def _generate_irrigation_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate sunlight, irrigation type, and irrigation needed targets."""
        out = df.copy()
        out["sunlight_hours"] = self._simulate_sunlight(out["label"])
        out["irrigation_type"] = self._simulate_irrigation_type(out)
        out["irrigation_needed"] = self._simulate_irrigation_needed(out)

        cols = self._resolve_irrigation_columns()
        if self.irrigation_df is not None and cols.crop_col is not None:
            irr = self.irrigation_df.copy()
            irr[cols.crop_col] = irr[cols.crop_col].astype(str).str.strip().str.lower()

            merge_cols = [cols.crop_col]
            rename_map = {cols.crop_col: "label"}
            if cols.irrigation_type_col is not None:
                merge_cols.append(cols.irrigation_type_col)
                rename_map[cols.irrigation_type_col] = "irrigation_type_aux"
            if cols.irrigation_needed_col is not None:
                merge_cols.append(cols.irrigation_needed_col)
                rename_map[cols.irrigation_needed_col] = "irrigation_needed_aux"

            merged_aux = irr[merge_cols].rename(columns=rename_map)
            # Avoid cartesian row explosion when auxiliary dataset has repeated crop rows.
            if "irrigation_type_aux" in merged_aux.columns:
                merged_aux["irrigation_type_aux"] = (
                    merged_aux["irrigation_type_aux"].astype(str).str.strip().str.lower()
                )
            agg_map = {}
            if "irrigation_type_aux" in merged_aux.columns:
                agg_map["irrigation_type_aux"] = (
                    lambda s: s.mode().iat[0] if not s.mode().empty else s.dropna().iloc[0] if not s.dropna().empty else np.nan
                )
            if "irrigation_needed_aux" in merged_aux.columns:
                merged_aux["irrigation_needed_aux"] = pd.to_numeric(
                    merged_aux["irrigation_needed_aux"], errors="coerce"
                )
                agg_map["irrigation_needed_aux"] = "median"
            if agg_map:
                merged_aux = merged_aux.groupby("label", as_index=False).agg(agg_map)
            out = out.merge(merged_aux, on="label", how="left")

            if "irrigation_type_aux" in out.columns:
                out["irrigation_type_aux"] = (
                    out["irrigation_type_aux"].astype(str).str.strip().str.lower()
                )
                use_aux_type = out["irrigation_type_aux"].notna() & (out["irrigation_type_aux"] != "nan")
                out.loc[use_aux_type, "irrigation_type"] = out.loc[use_aux_type, "irrigation_type_aux"]

            if "irrigation_needed_aux" in out.columns:
                aux_need = pd.to_numeric(out["irrigation_needed_aux"], errors="coerce")
                out["irrigation_needed"] = np.where(
                    np.isnan(aux_need),
                    out["irrigation_needed"],
                    aux_need,
                )

            out = out.drop(columns=[c for c in ["irrigation_type_aux", "irrigation_needed_aux"] if c in out.columns])

        out["irrigation_type"] = out["irrigation_type"].astype(str).str.strip().str.lower()
        out["irrigation_type_encoded"] = self.irrigation_type_encoder.fit_transform(out["irrigation_type"])
        self.irrigation_type_classes = self.irrigation_type_encoder.classes_.tolist()

        try:
            joblib.dump(
                self.irrigation_type_encoder,
                self.results_dir / "irrigation_type_encoder.pkl",
            )
        except Exception as exc:
            print(f"[DataPartitioner] Failed to save irrigation_type_encoder: {exc}")

        out["irrigation_needed"] = pd.to_numeric(out["irrigation_needed"], errors="coerce").fillna(
            pd.Series(self._simulate_irrigation_needed(out), index=out.index)
        )
        out["irrigation_needed"] = np.clip(out["irrigation_needed"], 0.0, 20.0)

        return out

    def _enrich_yield_from_production(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optionally enrich yield signal using crop production statistics."""
        if self.crop_production_df is None:
            return df

        prod = self.crop_production_df.copy()
        prod.columns = [str(c).strip().lower() for c in prod.columns]

        crop_col_candidates = ["label", "crop", "crop_name"]
        crop_col = next((c for c in crop_col_candidates if c in prod.columns), None)

        if crop_col is None:
            return df

        if "yield" in prod.columns:
            prod_yield = pd.to_numeric(prod["yield"], errors="coerce")
        elif "production" in prod.columns and "area" in prod.columns:
            production = pd.to_numeric(prod["production"], errors="coerce")
            area = pd.to_numeric(prod["area"], errors="coerce")
            prod_yield = production / np.where(area <= 0, np.nan, area)
        else:
            return df

        prod_df = pd.DataFrame(
            {
                "label": prod[crop_col].astype(str).str.strip().str.lower(),
                "yield_enriched": prod_yield,
            }
        ).dropna(subset=["yield_enriched"])

        if prod_df.empty:
            return df

        stats = prod_df.groupby("label", as_index=False)["yield_enriched"].median()
        out = df.merge(stats, on="label", how="left")

        enriched_mask = out["yield_enriched"].notna()
        out.loc[enriched_mask, "yield"] = (
            0.7 * out.loc[enriched_mask, "yield"] + 0.3 * out.loc[enriched_mask, "yield_enriched"]
        )
        out = out.drop(columns=["yield_enriched"])

        print("[DataPartitioner] Yield enriched from crop_production data")
        return out

    def _build_partition(self, partition_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create numpy arrays for model inputs and all task targets."""
        X_raw = partition_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        X_scaled = self.scaler.transform(X_raw)

        y_crop = self.label_encoder.transform(partition_df["label"].astype(str).to_numpy())
        y_yield = partition_df["yield"].to_numpy(dtype=np.float32)
        y_sunlight = partition_df["sunlight_hours"].to_numpy(dtype=np.float32)
        y_irr_type = partition_df["irrigation_type_encoded"].to_numpy(dtype=np.int64)
        y_irr_needed = partition_df["irrigation_needed"].to_numpy(dtype=np.float32)

        return {
            "X": X_scaled,
            "y_crop": y_crop,
            "y_yield": y_yield,
            "y_sunlight": y_sunlight,
            "y_irr_type": y_irr_type,
            "y_irr_needed": y_irr_needed,
        }

    def _to_datasets(self, partition_df: pd.DataFrame, split_seed: int) -> Tuple[FarmDataset, FarmDataset]:
        """Split partition into train/val and convert to FarmDataset objects."""
        if len(partition_df) < 2:
            partition_df = pd.concat([partition_df, partition_df], ignore_index=True)

        train_df, val_df = train_test_split(
            partition_df,
            test_size=0.15,
            random_state=split_seed,
            shuffle=True,
        )

        train_arrays = self._build_partition(train_df)
        val_arrays = self._build_partition(val_df)

        train_ds = FarmDataset(
            train_arrays["X"],
            train_arrays["y_crop"],
            train_arrays["y_yield"],
            train_arrays["y_sunlight"],
            train_arrays["y_irr_type"],
            train_arrays["y_irr_needed"],
        )
        val_ds = FarmDataset(
            val_arrays["X"],
            val_arrays["y_crop"],
            val_arrays["y_yield"],
            val_arrays["y_sunlight"],
            val_arrays["y_irr_type"],
            val_arrays["y_irr_needed"],
        )

        return train_ds, val_ds

    def _save_core_artifacts(self) -> None:
        """Persist scaler and crop encoder artifacts."""
        try:
            joblib.dump(self.scaler, self.results_dir / "federated_scaler.pkl")
            joblib.dump(self.label_encoder, self.results_dir / "federated_label_encoder.pkl")
        except Exception as exc:
            print(f"[DataPartitioner] Failed to save scaler/label encoder: {exc}")

    def _save_state_metadata(self) -> None:
        """Persist state-level partition metadata for reporting."""
        try:
            with open(self.results_dir / "state_partition_metadata.json", "w", encoding="utf-8") as f:
                json.dump(self.state_metadata_list, f, indent=2)
        except Exception as exc:
            print(f"[DataPartitioner] Failed to save state metadata: {exc}")

    def _print_state_summary(self, metadata: List[Dict[str, object]]) -> None:
        """Print required per-state summary table."""
        print("-" * 58)
        print(f"{'State':<20} {'Samples':<9} {'Padded':<8} Top 3 Crops")
        print("-" * 58)

        total_samples = 0
        for row in metadata:
            state_name = str(row["state_name"])
            samples = int(row["sample_count"])
            padded = "Yes" if bool(row["is_padded"]) else "No"
            crops = ", ".join(row.get("top3_actual_crops", []))
            print(f"{state_name:<20} {samples:<9} {padded:<8} {crops}")
            total_samples += samples

        print("-" * 58)
        print(f"Total clients: {len(metadata)} | Total samples: {total_samples}")

    def create_non_iid_partitions(self, df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        """Create non-IID partitions with one client per Indian state profile."""
        state_names = list(STATE_PROFILES.keys())
        self.num_clients = len(state_names)

        # Fit global preprocessing on full dataset
        self.scaler.fit(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        self.label_encoder.fit(df["label"].astype(str).to_numpy())
        self._save_core_artifacts()

        partitions: Dict[str, Dict[str, object]] = {}
        metadata_rows: List[Dict[str, object]] = []

        for idx, state_name in enumerate(state_names):
            profile = STATE_PROFILES[state_name]
            dominant_crops = [c.strip().lower() for c in profile["dominant_crops"]]

            native = df[df["label"].isin(dominant_crops)].copy()
            non_matching = df[~df["label"].isin(dominant_crops)].copy()

            # Add 10% random non-matching rows to simulate experimentation
            extra_count = int(round(0.1 * len(native)))
            if extra_count > 0 and len(non_matching) > 0:
                replace = extra_count > len(non_matching)
                sampled_extra = non_matching.sample(
                    n=extra_count,
                    random_state=self.random_seed + idx,
                    replace=replace,
                )
                state_df = pd.concat([native, sampled_extra], ignore_index=True)
            else:
                state_df = native

            is_padded = False
            if len(state_df) < 10:
                is_padded = True
                needed = max(0, 30 - len(state_df))
                if needed > 0:
                    pad_df = df.sample(
                        n=needed,
                        random_state=self.random_seed + 1000 + idx,
                        replace=(needed > len(df)),
                    )
                    state_df = pd.concat([state_df, pad_df], ignore_index=True)
                print(
                    f"[DataPartitioner] {state_name} has fewer than 10 native samples — "
                    "padded to 30 with random data"
                )

            state_df = state_df.sample(frac=1.0, random_state=self.random_seed + idx).reset_index(drop=True)

            top3 = (
                state_df["label"].value_counts().head(3).index.astype(str).tolist()
                if not state_df.empty
                else []
            )

            train_ds, val_ds = self._to_datasets(state_df, split_seed=self.random_seed + idx)

            partitions[state_name] = {
                "train": train_ds,
                "val": val_ds,
                "metadata": {
                    "state_name": state_name,
                    "sample_count": int(len(state_df)),
                    "dominant_crops": dominant_crops,
                    "is_padded": is_padded,
                    "top3_actual_crops": top3,
                    "climate": profile["climate"],
                },
            }

            metadata_rows.append(partitions[state_name]["metadata"])

        self.state_metadata_list = metadata_rows
        self._save_state_metadata()
        self._print_state_summary(metadata_rows)

        return partitions

    def create_iid_partitions(self, df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        """Create IID ablation partitions across 28 clients."""
        self.num_clients = 28

        self.scaler.fit(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        self.label_encoder.fit(df["label"].astype(str).to_numpy())
        self._save_core_artifacts()

        shuffled = df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        index_chunks = np.array_split(np.arange(len(shuffled)), self.num_clients)

        partitions: Dict[str, Dict[str, object]] = {}
        metadata_rows: List[Dict[str, object]] = []

        for idx, chunk in enumerate(index_chunks):
            client_name = f"client_{idx}"
            client_df = shuffled.iloc[chunk].copy().reset_index(drop=True)

            if client_df.empty:
                client_df = shuffled.sample(n=30, random_state=self.random_seed + idx, replace=True)

            train_ds, val_ds = self._to_datasets(client_df, split_seed=self.random_seed + idx)
            top3 = client_df["label"].value_counts().head(3).index.astype(str).tolist()

            partitions[client_name] = {
                "train": train_ds,
                "val": val_ds,
                "metadata": {
                    "state_name": client_name,
                    "sample_count": int(len(client_df)),
                    "dominant_crops": [],
                    "is_padded": False,
                    "top3_actual_crops": top3,
                    "climate": "mixed",
                },
            }
            metadata_rows.append(partitions[client_name]["metadata"])

        self.state_metadata_list = metadata_rows
        self._save_state_metadata()
        self._print_state_summary(metadata_rows)

        return partitions

    def get_centralized_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Return centralized train and validation loaders using the same targets."""
        self.scaler.fit(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        self.label_encoder.fit(df["label"].astype(str).to_numpy())

        train_df, val_df = train_test_split(
            df,
            test_size=0.15,
            random_state=self.random_seed,
            shuffle=True,
        )

        train_arrays = self._build_partition(train_df)
        val_arrays = self._build_partition(val_df)

        train_ds = FarmDataset(
            train_arrays["X"],
            train_arrays["y_crop"],
            train_arrays["y_yield"],
            train_arrays["y_sunlight"],
            train_arrays["y_irr_type"],
            train_arrays["y_irr_needed"],
        )
        val_ds = FarmDataset(
            val_arrays["X"],
            val_arrays["y_crop"],
            val_arrays["y_yield"],
            val_arrays["y_sunlight"],
            val_arrays["y_irr_type"],
            val_arrays["y_irr_needed"],
        )

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        return train_loader, val_loader
