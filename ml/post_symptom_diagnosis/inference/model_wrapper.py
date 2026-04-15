"""
Singleton model wrapper for the Post-Symptom Diagnosis TorchScript model.

Loads the model and metadata once, caches them in-process, and exposes a
clean `predict()` method used by the service layer.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ml.post_symptom_diagnosis.inference.preprocessing import preprocess_image

logger = logging.getLogger(__name__)

# ── Artifact paths (project-relative) ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # …/TerraMind-Smarter-Farming-Every-Stage
_ARTIFACTS_DIR = _PROJECT_ROOT / "ml" / "post_symptom_diagnosis" / "saved_models" / "trained_artifacts_fast"
_TORCHSCRIPT_PATH = _ARTIFACTS_DIR / "plant_disease_model_fast_torchscript.pt"
_METADATA_PATH = _ARTIFACTS_DIR / "class_metadata_fast.json"
# Optional fallback (not used for primary inference):
# _PTH_PATH = _ARTIFACTS_DIR / "best_plant_disease_model_fast.pth"


class DiagnosisModelWrapper:
    """Lazily-loaded, thread-safe(ish) singleton wrapper around the TorchScript plant-disease model."""

    _instance: Optional["DiagnosisModelWrapper"] = None

    def __init__(self) -> None:
        self.model: Optional[torch.jit.ScriptModule] = None
        self.class_names: List[str] = []
        self.class_to_crop: Dict[str, str] = {}
        self.img_size: int = 192  # default; overwritten by metadata
        self.num_classes: int = 0
        self._loaded = False

    # ── Singleton access ──────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "DiagnosisModelWrapper":
        """Return (and lazily create) the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        if not cls._instance._loaded:
            cls._instance.load()
        return cls._instance

    # ── Loading ───────────────────────────────────────────────────────────
    def load(self) -> None:
        """Load the TorchScript model and metadata JSON from disk."""
        logger.info("Loading Post-Symptom Diagnosis model artifacts …")

        # --- Metadata ---
        if not _METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {_METADATA_PATH}. "
                "Ensure trained artifacts are placed in ml/post_symptom_diagnosis/saved_models/trained_artifacts_fast/"
            )
        with open(_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.class_names = metadata["class_names"]
        self.class_to_crop = metadata["class_to_crop"]
        self.img_size = metadata.get("img_size", 192)
        self.num_classes = metadata.get("num_classes", len(self.class_names))
        logger.info(
            "Metadata loaded — %d classes, img_size=%d", self.num_classes, self.img_size
        )

        # Validate consistency
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"Metadata mismatch: class_names has {len(self.class_names)} entries "
                f"but num_classes is {self.num_classes}"
            )

        # --- TorchScript model ---
        if not _TORCHSCRIPT_PATH.exists():
            raise FileNotFoundError(
                f"TorchScript model not found at {_TORCHSCRIPT_PATH}. "
                "Ensure trained artifacts are placed in ml/post_symptom_diagnosis/saved_models/trained_artifacts_fast/"
            )
        self.model = torch.jit.load(str(_TORCHSCRIPT_PATH), map_location="cpu")
        self.model.eval()
        logger.info("TorchScript model loaded successfully from %s", _TORCHSCRIPT_PATH.name)

        self._loaded = True

    # ── Inference ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, image_bytes: bytes, top_k: int = 3) -> Dict[str, Any]:
        """
        Run inference on raw image bytes.

        Parameters
        ----------
        image_bytes : bytes
            Raw file bytes of the uploaded leaf/plant image.
        top_k : int
            Number of top predictions to return.

        Returns
        -------
        dict
            {
                "identified_crop": str,
                "identified_class": str,
                "confidence": float,
                "top_k_predictions": [
                    {"crop": str, "class": str, "confidence": float}, …
                ],
                "assistant_available": True,
            }
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Clamp top_k
        top_k = max(1, min(top_k, self.num_classes))

        # Preprocess
        tensor = preprocess_image(image_bytes, self.img_size)

        # Inference
        logits = self.model(tensor)  # (1, num_classes)
        probabilities = torch.nn.functional.softmax(logits, dim=1)  # (1, num_classes)

        # Top-K
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
        top_probs = top_probs.squeeze(0).tolist()
        top_indices = top_indices.squeeze(0).tolist()

        # Map to human-readable names
        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.class_names[idx]
            crop_name = self.class_to_crop.get(class_name, class_name.split("__")[0])
            top_predictions.append({
                "crop": crop_name,
                "class": class_name,
                "confidence": round(prob, 4),
            })

        # Best prediction
        best = top_predictions[0]

        result = {
            "identified_crop": best["crop"],
            "identified_class": best["class"],
            "confidence": best["confidence"],
            "top_k_predictions": top_predictions,
            "assistant_available": True,
        }

        logger.info(
            "Prediction: %s → %s (%.2f%%)",
            result["identified_crop"],
            result["identified_class"],
            result["confidence"] * 100,
        )
        return result
