"""
Public inference API for the Post-Symptom Diagnosis model.

This module provides a single-function entry point used by the backend
service layer.  It delegates to :class:`DiagnosisModelWrapper` singleton.
"""

import logging
from typing import Any, Dict

from ml.post_symptom_diagnosis.inference.model_wrapper import DiagnosisModelWrapper

logger = logging.getLogger(__name__)


def predict_disease(image_bytes: bytes, top_k: int = 3) -> Dict[str, Any]:
    """
    Run plant-disease inference on raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        The uploaded image file content.
    top_k : int, optional
        Number of top predictions to return (default 3).

    Returns
    -------
    dict
        Structured prediction result — see :meth:`DiagnosisModelWrapper.predict`.
    """
    wrapper = DiagnosisModelWrapper.get_instance()
    return wrapper.predict(image_bytes, top_k=top_k)
