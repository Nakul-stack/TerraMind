"""Federated inference module for TerraMind (no Flower runtime dependency)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn

from federated.model import AdvisorNet


class FederatedAdvisor:
    """Load and run inference using saved TerraMind federated model artifacts."""

    WEIGHTS_PATH = "federated/results/federated_advisor_final.pth"
    SCALER_PATH = "federated/results/federated_scaler.pkl"
    ENCODER_PATH = "federated/results/federated_label_encoder.pkl"
    METADATA_PATH = "federated/results/federated_model_metadata.json"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> None:
        self.weights_path = weights_path or self.WEIGHTS_PATH
        self.scaler_path = scaler_path or self.SCALER_PATH
        self.encoder_path = encoder_path or self.ENCODER_PATH
        self.metadata_path = metadata_path or self.METADATA_PATH

        self.is_loaded: bool = False
        self.missing_files: List[str] = []
        self.load_error: Optional[str] = None

        self.metadata: Dict[str, Any] = {}
        self.num_classes: int = 0
        self.feature_names: List[str] = []
        self.class_names: List[str] = []

        self.model: Optional[nn.Module] = None
        self.scaler = None
        self.label_encoder = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model + preprocessors from saved federated artifacts."""
        try:
            required_files = [
                self.weights_path,
                self.scaler_path,
                self.encoder_path,
                self.metadata_path,
            ]

            self.missing_files = [p for p in required_files if not os.path.exists(p)]
            if self.missing_files:
                self.is_loaded = False
                print(f"[FederatedAdvisor] Missing files: {self.missing_files}")
                print("[FederatedAdvisor] Run the FL simulation first: python -m federated.run_simulation")
                return

            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            self.num_classes = int(self.metadata.get("num_classes", 0))
            self.feature_names = list(self.metadata.get("feature_names", []))
            self.class_names = list(self.metadata.get("class_names", []))

            checkpoint = torch.load(self.weights_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", {})
            num_irrigation_types = int(checkpoint.get("num_irrigation_types", 4))
            if "irrigation_type_head.2.bias" in state_dict:
                try:
                    num_irrigation_types = int(state_dict["irrigation_type_head.2.bias"].shape[0])
                except Exception:
                    pass

            self.model = AdvisorNet(
                num_classes=self.num_classes,
                num_irrigation_types=num_irrigation_types,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.model.eval()

            self.scaler = joblib.load(self.scaler_path)
            self.label_encoder = joblib.load(self.encoder_path)

            self.is_loaded = True
            print("[FederatedAdvisor] Model loaded successfully")
            print(f"[FederatedAdvisor]    Classes: {self.num_classes} crops")
            try:
                print(f"[FederatedAdvisor]    FL Accuracy: {float(self.metadata.get('federated_accuracy', 0.0)):.4f}")
            except Exception:
                pass
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            print(f"[FederatedAdvisor] ERROR loading model: {e}")

    def is_available(self) -> bool:
        """Return whether federated artifacts were loaded successfully."""
        return self.is_loaded

    @staticmethod
    def _validate_inputs(
        N: float,
        P: float,
        K: float,
        ph: float,
        temperature: float,
        humidity: float,
        rainfall: float,
    ) -> Tuple[bool, Optional[str], Optional[List[float]]]:
        try:
            values = [float(N), float(P), float(K), float(ph), float(temperature), float(humidity), float(rainfall)]
        except Exception:
            return False, "All inputs must be numeric.", None

        n, p, k, ph_val, temp, hum, rain = values

        if not (0 <= n <= 200):
            return False, "N must be in range [0, 200].", None
        if not (0 <= p <= 200):
            return False, "P must be in range [0, 200].", None
        if not (0 <= k <= 200):
            return False, "K must be in range [0, 200].", None
        if not (0 <= ph_val <= 14):
            return False, "ph must be in range [0, 14].", None
        if not (-10 <= temp <= 60):
            return False, "temperature must be in range [-10, 60].", None
        if not (0 <= hum <= 100):
            return False, "humidity must be in range [0, 100].", None
        if not (0 <= rain <= 5000):
            return False, "rainfall must be in range [0, 5000].", None

        return True, None, values

    def predict(
        self,
        N: float,
        P: float,
        K: float,
        ph: float,
        temperature: float,
        humidity: float,
        rainfall: float,
    ) -> Dict[str, Any]:
        """Run federated model inference on a single farm input. Returns crop recommendation with confidence scores and yield estimate. Raw input is never stored."""
        if not self.is_loaded or self.model is None:
            return {
                "success": False,
                "error": "Federated model not loaded",
                "hint": "Run: python -m federated.run_simulation",
            }

        ok, message, features = self._validate_inputs(N, P, K, ph, temperature, humidity, rainfall)
        if not ok or features is None:
            return {"success": False, "error": "Validation failed", "detail": message}

        try:
            X = np.array([features], dtype=np.float32)
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)

            with torch.no_grad():
                outputs = self.model(X_tensor)

                if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                    crop_logits = outputs[0]
                    yield_pred = outputs[1]
                else:
                    raise ValueError("Unexpected model output format from AdvisorNet")

                probabilities = torch.softmax(crop_logits, dim=1)
                top3_probs, top3_indices = torch.topk(probabilities, k=3)

            predicted_class_idx = top3_indices[0][0].item()
            predicted_crop = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            confidence = top3_probs[0][0].item()

            if hasattr(yield_pred, "ndim") and getattr(yield_pred, "ndim", 1) > 1:
                yield_estimate = float(yield_pred[0][0].item())
            else:
                yield_estimate = float(yield_pred[0].item())

            top3_predictions: List[Dict[str, Any]] = []
            for i in range(3):
                idx = top3_indices[0][i].item()
                crop_name = self.label_encoder.inverse_transform([idx])[0]
                prob = top3_probs[0][i].item()
                top3_predictions.append({"crop": crop_name, "confidence": round(prob, 4)})

            return {
                "success": True,
                "model_type": "federated",
                "privacy_guarantee": (
                    "Input data not used in training. Model trained via federated learning - "
                    "raw farm data never left client devices."
                ),
                "predicted_crop": predicted_crop,
                "recommended_crop": predicted_crop,
                "confidence": round(confidence, 4),
                "confidence_pct": f"{confidence * 100:.1f}%",
                "yield_estimate_tons_per_hectare": round(max(0.1, yield_estimate), 2),
                "top_3_predictions": top3_predictions,
                "federated_model_accuracy": self.metadata.get("federated_accuracy"),
                "centralized_model_accuracy": self.metadata.get("centralized_accuracy"),
                "accuracy_gap_pct": self.metadata.get("accuracy_gap_pct"),
                "privacy_note": (
                    "This prediction was made using a privacy-preserving federated model. "
                    "Your farm data was never stored on any central server."
                ),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata if loaded, or missing file diagnostics otherwise."""
        if not self.is_loaded:
            return {"loaded": False, "missing_files": self.missing_files}

        return {
            "loaded": True,
            "num_classes": self.metadata.get("num_classes"),
            "num_clients": self.metadata.get("num_clients"),
            "num_rounds": self.metadata.get("num_rounds"),
            "partition_mode": self.metadata.get("partition_mode"),
            "feature_names": self.metadata.get("feature_names"),
            "federated_accuracy": self.metadata.get("federated_accuracy"),
            "centralized_accuracy": self.metadata.get("centralized_accuracy"),
            "accuracy_gap_pct": self.metadata.get("accuracy_gap_pct"),
        }
