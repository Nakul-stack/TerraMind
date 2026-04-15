"""Ensemble inference engine combining centralized RF and 28-state federated model."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from federated.inference import FederatedAdvisor
from ml.pre_sowing_advisor.crop_recommendation.predict import predict_crop
from ml.pre_sowing_advisor.irrigation_sunlight.predict import predict_irrigation_advisory
from ml.pre_sowing_advisor.yield_prediction.predict import predict_yield


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


class EnsembleAdvisor:
    """Three-strategy hybrid decision engine for TerraMind advisory inference."""

    def __init__(self, rf_models_dir: str = "ml/", fl_weights_dir: str = "federated/results/") -> None:
        self.rf_models_dir = Path(rf_models_dir)
        self.fl_weights_dir = Path(fl_weights_dir)

        self.is_loaded = False
        self.rf_available = False

        self.crop_model = None
        self.crop_scaler = None
        self.crop_label_encoder = None
        self.irr_type_encoder = None

        self.fl_advisor = FederatedAdvisor()
        self._load_rf_components()

        self.is_loaded = bool(self.rf_available or self.fl_advisor.is_available())
        if self.is_loaded:
            print("[EnsembleAdvisor] Ensemble initialized with graceful fallback support")
        else:
            print("[EnsembleAdvisor] No model backend available (RF and FL both unavailable)")

    def _load_rf_components(self) -> None:
        """Load RF artifacts used for confidence and top-3 extraction."""
        try:
            crop_dir = self.rf_models_dir / "pre_sowing_advisor" / "crop_recommendation" / "saved_models"
            model_path = crop_dir / "model.pkl"
            scaler_path = crop_dir / "scaler.pkl"
            encoder_path = crop_dir / "label_encoder.pkl"

            if not (model_path.exists() and scaler_path.exists() and encoder_path.exists()):
                print("[EnsembleAdvisor] RF crop artifacts missing; RF path will degrade if called")
                self.rf_available = False
                return

            self.crop_model = joblib.load(model_path)
            self.crop_scaler = joblib.load(scaler_path)
            self.crop_label_encoder = joblib.load(encoder_path)

            irr_path = self.fl_weights_dir / "irrigation_type_encoder.pkl"
            if irr_path.exists():
                try:
                    self.irr_type_encoder = joblib.load(irr_path)
                except Exception:
                    self.irr_type_encoder = None

            self.rf_available = True
            print("[EnsembleAdvisor] RF components loaded")
        except Exception as exc:
            self.rf_available = False
            print(f"[EnsembleAdvisor] Failed to load RF components: {exc}")

    def is_available(self) -> bool:
        """Return whether at least one backend (RF/FL) is available."""
        return bool(self.is_loaded)

    def _rf_crop_with_confidence(self, N: float, P: float, K: float, ph: float, temperature: float, humidity: float, rainfall: float) -> Dict[str, Any]:
        """Run RF crop classifier and extract confidence + top-3 probabilities."""
        if self.crop_model is None or self.crop_scaler is None:
            # Fallback to legacy API without confidence details
            basic = predict_crop(
                {
                    "N": N,
                    "P": P,
                    "K": K,
                    "temperature": temperature,
                    "humidity": humidity,
                    "rainfall": rainfall,
                    "ph": ph,
                }
            )
            return {
                "recommended_crop": basic.get("recommended_crop"),
                "crop_confidence": 0.0,
                "crop_confidence_pct": "0.0%",
                "top_3_predictions": [],
                "top_3_probabilities": [0.0, 0.0, 0.0],
            }

        x = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=np.float64)
        x_scaled = self.crop_scaler.transform(x)

        pred_encoded = self.crop_model.predict(x_scaled)[0]
        pred_crop = str(self.crop_label_encoder.inverse_transform([pred_encoded])[0])

        confidence = 0.0
        top_3_predictions: List[Dict[str, Any]] = []
        top_3_probabilities = [0.0, 0.0, 0.0]

        if hasattr(self.crop_model, "predict_proba"):
            probs = self.crop_model.predict_proba(x_scaled)[0]
            classes = self.crop_model.classes_

            pred_idx = int(np.where(classes == pred_encoded)[0][0]) if pred_encoded in classes else int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            sorted_idx = np.argsort(probs)[::-1][:3]
            top_3_probabilities = [float(probs[i]) for i in sorted_idx]
            top_3_predictions = [
                {
                    "crop": str(self.crop_label_encoder.inverse_transform([int(classes[i])])[0]),
                    "confidence": round(float(probs[i]), 4),
                }
                for i in sorted_idx
            ]

        return {
            "recommended_crop": pred_crop,
            "crop_confidence": float(confidence),
            "crop_confidence_pct": f"{confidence * 100:.1f}%",
            "top_3_predictions": top_3_predictions,
            "top_3_probabilities": top_3_probabilities,
        }

    def _run_rf_pipeline(self, N: float, P: float, K: float, ph: float, temperature: float, humidity: float, rainfall: float) -> Dict[str, Any]:
        """Run centralized RF stack and normalize to a unified dictionary."""
        crop_part = self._rf_crop_with_confidence(N, P, K, ph, temperature, humidity, rainfall)
        crop = crop_part.get("recommended_crop")

        # RF model family needs additional context; use stable defaults when API does not supply them.
        state_name = "karnataka"
        district_name = "mysore"
        season = "kharif"
        soil_type = "loamy"

        y = predict_yield(
            {
                "crop": crop,
                "state_name": state_name,
                "district_name": district_name,
                "season": season,
            }
        )
        irr = predict_irrigation_advisory(
            {
                "crop": crop,
                "ph": ph,
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "soil_type": soil_type,
                "season": season,
            }
        )

        irrigation_need_raw = irr.get("irrigation_need", 5.0)
        try:
            irrigation_need_num = float(irrigation_need_raw)
        except Exception:
            mapping = {"low": 2.5, "medium": 5.0, "high": 8.0}
            irrigation_need_num = float(mapping.get(str(irrigation_need_raw).strip().lower(), 5.0))

        return {
            "success": True,
            "model_type": "standard",
            "recommended_crop": crop,
            "crop_confidence": float(crop_part.get("crop_confidence", 0.0)),
            "crop_confidence_pct": crop_part.get("crop_confidence_pct", "0.0%"),
            "top_3_predictions": crop_part.get("top_3_predictions", []),
            "top_3_probabilities": crop_part.get("top_3_probabilities", [0.0, 0.0, 0.0]),
            "expected_yield": float(y.get("predicted_yield", 0.0)),
            "sunlight_hours": float(irr.get("sunlight_hours", 6.0)),
            "irrigation_type": str(irr.get("irrigation_type", "sprinkler")),
            "irrigation_needed": float(irrigation_need_num),
            "privacy_note": "Standard centralized Random Forest.",
        }

    def _run_decision_engine(self, rf_result: Dict[str, Any], fl_result: Dict[str, Any]) -> Dict[str, Any]:
        """Three-strategy hybrid decision engine."""
        rf_crop = rf_result.get("recommended_crop")
        rf_confidence = float(rf_result.get("crop_confidence", 0.0))
        rf_top3_probs = rf_result.get("top_3_probabilities", [rf_confidence, 0.0, 0.0])

        fl_crop = fl_result.get("recommended_crop")
        fl_confidence = float(fl_result.get("confidence", 0.0))
        fl_top3 = fl_result.get("top_3_predictions", [])
        fl_top3_probs = [float(p.get("confidence", 0.0)) for p in fl_top3[:3]]

        # Strategy A - Simple voting
        if rf_crop == fl_crop:
            a_crop = rf_crop
            a_conf = max(rf_confidence, fl_confidence)
            a_source = "agreement"
        else:
            if rf_confidence >= fl_confidence:
                a_crop = rf_crop
                a_conf = rf_confidence
                a_source = "rf_higher_confidence"
            else:
                a_crop = fl_crop
                a_conf = fl_confidence
                a_source = "fl_higher_confidence"

        # Strategy B - Weighted ensemble (RF=0.6, FL=0.4)
        if rf_crop == fl_crop:
            b_crop = rf_crop
            b_conf = 0.6 * rf_confidence + 0.4 * fl_confidence
        else:
            score_rf = 0.6 * rf_confidence + 0.4 * (fl_top3_probs[1] if len(fl_top3_probs) > 1 else 0.0)
            score_fl = 0.4 * fl_confidence + 0.6 * (rf_top3_probs[1] if len(rf_top3_probs) > 1 else 0.0)
            if score_rf >= score_fl:
                b_crop = rf_crop
                b_conf = score_rf
            else:
                b_crop = fl_crop
                b_conf = score_fl

        # Strategy C - Confidence gating
        fl_threshold = 0.85
        if fl_confidence >= fl_threshold:
            c_crop = fl_crop
            c_conf = fl_confidence
            c_source = "fl_trusted_high_confidence"
        else:
            c_crop = rf_crop
            c_conf = rf_confidence
            c_source = "rf_fallback"

        votes = [a_crop, b_crop, c_crop]
        vote_counts = Counter(votes)
        winner_crop, winner_votes = vote_counts.most_common(1)[0]

        if winner_votes >= 2:
            final_crop = winner_crop
            decision_source = "majority_vote"
            if final_crop == a_crop:
                final_conf = a_conf
            elif final_crop == b_crop:
                final_conf = b_conf
            else:
                final_conf = c_conf
        else:
            final_crop = b_crop
            final_conf = b_conf
            decision_source = "tiebreak_weighted_ensemble"

        final_conf = min(1.0, float(final_conf))

        return {
            "final_crop": final_crop,
            "final_confidence": final_conf,
            "decision_source": decision_source,
            "all_strategies_agree": bool(a_crop == b_crop == c_crop),
            "strategy_detail": {
                "strategy_a": {
                    "result": a_crop,
                    "confidence": round(float(a_conf), 4),
                    "source": a_source,
                    "description": "Simple voting",
                },
                "strategy_b": {
                    "result": b_crop,
                    "confidence": round(float(b_conf), 4),
                    "description": "Weighted RF=60% FL=40%",
                },
                "strategy_c": {
                    "result": c_crop,
                    "confidence": round(float(c_conf), 4),
                    "source": c_source,
                    "description": "FL leads if conf>=85%",
                },
                "votes": {
                    "vote_counts": dict(vote_counts),
                    "winning_votes": int(winner_votes),
                    "total_strategies": 3,
                },
            },
        }

    def _merge_top3(self, rf_result: Dict[str, Any], fl_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Merge RF and FL top-3 predictions into blended top-3 unique crops."""
        scores: Dict[str, Dict[str, float]] = {}

        for item in rf_result.get("top_3_predictions", []):
            crop = str(item.get("crop", "")).strip()
            if not crop:
                continue
            scores.setdefault(crop, {"rf": 0.0, "fl": 0.0})
            scores[crop]["rf"] = max(scores[crop]["rf"], float(item.get("confidence", 0.0)))

        for item in fl_result.get("top_3_predictions", []):
            crop = str(item.get("crop", "")).strip()
            if not crop:
                continue
            scores.setdefault(crop, {"rf": 0.0, "fl": 0.0})
            scores[crop]["fl"] = max(scores[crop]["fl"], float(item.get("confidence", 0.0)))

        blended = []
        for crop, parts in scores.items():
            score = 0.6 * parts["rf"] + 0.4 * parts["fl"]
            blended.append((crop, score))

        blended.sort(key=lambda x: x[1], reverse=True)
        return [{"crop": crop, "confidence": round(float(score), 4)} for crop, score in blended[:3]]

    def _rf_only_response(self, rf_result: Dict[str, Any], rf_failed: bool, fl_failed: bool) -> Dict[str, Any]:
        conf = float(rf_result.get("crop_confidence", 0.0))
        final_sun = _clip(float(rf_result.get("sunlight_hours", 6.0)), 3.0, 12.0)
        final_need = _clip(float(rf_result.get("irrigation_needed", 5.0)), 0.0, 20.0)
        final_yield = _clip(float(rf_result.get("expected_yield", 0.1)), 0.1, 15.0)

        return {
            "success": True,
            "model_type": "ensemble",
            "recommended_crop": rf_result.get("recommended_crop"),
            "confidence": round(conf, 4),
            "confidence_pct": f"{conf * 100:.1f}%",
            "expected_yield": round(final_yield, 2),
            "sunlight_hours": round(final_sun, 1),
            "sunlight_hours_display": f"{round(final_sun, 1)} hours/day",
            "irrigation_type": rf_result.get("irrigation_type", "sprinkler"),
            "irrigation_needed": round(final_need, 2),
            "irrigation_needed_display": f"{round(final_need, 2)} mm/day",
            "top_3_predictions": rf_result.get("top_3_predictions", []),
            "decision_engine": {
                "method": "rf_only_degraded",
                "description": "Federated backend unavailable; serving centralized Random Forest output.",
                "all_strategies_agree": False,
            },
            "individual_model_outputs": {
                "random_forest": {
                    "recommended_crop": rf_result.get("recommended_crop"),
                    "confidence": round(conf, 4),
                    "confidence_pct": f"{conf * 100:.1f}%",
                    "expected_yield": rf_result.get("expected_yield"),
                    "sunlight_hours": rf_result.get("sunlight_hours"),
                    "irrigation_type": rf_result.get("irrigation_type"),
                    "irrigation_needed": rf_result.get("irrigation_needed"),
                    "model_type": "Random Forest (Centralized)",
                    "failed": rf_failed,
                },
                "federated_model": {
                    "model_type": "AdvisorNet (28-State Federated)",
                    "failed": fl_failed,
                },
            },
            "agreement_analysis": {
                "rf_fl_agree": False,
                "all_strategies_agree": False,
                "confidence_level": (
                    "very_high" if conf >= 0.85 else "high" if conf >= 0.70 else "medium" if conf >= 0.50 else "low"
                ),
                "recommendation": "Federated model unavailable. Using centralized RF output.",
            },
            "privacy_note": (
                "TerraMind attempted hybrid inference, but only the centralized Random Forest "
                "was available for this request."
            ),
        }

    def _fl_only_response(self, fl_result: Dict[str, Any], rf_failed: bool, fl_failed: bool) -> Dict[str, Any]:
        conf = float(fl_result.get("confidence", 0.0))
        final_sun = _clip(float(fl_result.get("sunlight_hours", 6.0)), 3.0, 12.0)
        final_need = _clip(float(fl_result.get("irrigation_needed", 5.0)), 0.0, 20.0)
        final_yield = _clip(float(fl_result.get("expected_yield", 0.1)), 0.1, 15.0)

        return {
            "success": True,
            "model_type": "ensemble",
            "recommended_crop": fl_result.get("recommended_crop"),
            "confidence": round(conf, 4),
            "confidence_pct": f"{conf * 100:.1f}%",
            "expected_yield": round(final_yield, 2),
            "sunlight_hours": round(final_sun, 1),
            "sunlight_hours_display": f"{round(final_sun, 1)} hours/day",
            "irrigation_type": fl_result.get("irrigation_type", "sprinkler"),
            "irrigation_needed": round(final_need, 2),
            "irrigation_needed_display": f"{round(final_need, 2)} mm/day",
            "top_3_predictions": fl_result.get("top_3_predictions", []),
            "decision_engine": {
                "method": "fl_only_degraded",
                "description": "Centralized RF backend unavailable; serving federated output.",
                "all_strategies_agree": False,
            },
            "individual_model_outputs": {
                "random_forest": {
                    "model_type": "Random Forest (Centralized)",
                    "failed": rf_failed,
                },
                "federated_model": {
                    "recommended_crop": fl_result.get("recommended_crop"),
                    "confidence": round(conf, 4),
                    "confidence_pct": f"{conf * 100:.1f}%",
                    "expected_yield": fl_result.get("expected_yield"),
                    "sunlight_hours": fl_result.get("sunlight_hours"),
                    "irrigation_type": fl_result.get("irrigation_type"),
                    "irrigation_needed": fl_result.get("irrigation_needed"),
                    "model_type": "AdvisorNet (28-State Federated)",
                    "states_trained": 28,
                    "privacy_note": "Trained across 28 Indian states. Raw farm data never left any state device during training.",
                    "failed": fl_failed,
                },
            },
            "agreement_analysis": {
                "rf_fl_agree": False,
                "all_strategies_agree": False,
                "confidence_level": (
                    "very_high" if conf >= 0.85 else "high" if conf >= 0.70 else "medium" if conf >= 0.50 else "low"
                ),
                "recommendation": "Centralized RF unavailable. Using federated model output.",
            },
            "privacy_note": (
                "TerraMind attempted hybrid inference, but only the privacy-preserving federated "
                "model was available for this request."
            ),
        }

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
        """Run RF + FL and return three-strategy ensemble output."""
        rf_result: Dict[str, Any] = {}
        fl_result: Dict[str, Any] = {}
        rf_failed = False
        fl_failed = False

        # STEP 1 - RF
        try:
            rf_result = self._run_rf_pipeline(N, P, K, ph, temperature, humidity, rainfall)
        except Exception as exc:
            rf_failed = True
            rf_result = {"success": False, "error": str(exc)}

        # STEP 2 - FL
        try:
            fl_result = self.fl_advisor.predict(N, P, K, ph, temperature, humidity, rainfall)
            fl_failed = not bool(fl_result.get("success", False))
        except Exception as exc:
            fl_failed = True
            fl_result = {"success": False, "error": str(exc)}

        # STEP 3 - graceful degradation
        if rf_failed and fl_failed:
            return {"success": False, "error": "Both models failed"}
        if rf_failed and not fl_failed:
            return self._fl_only_response(fl_result, rf_failed=rf_failed, fl_failed=fl_failed)
        if fl_failed and not rf_failed:
            return self._rf_only_response(rf_result, rf_failed=rf_failed, fl_failed=fl_failed)

        # STEP 4 - decision engine
        decision = self._run_decision_engine(rf_result, fl_result)

        rf_crop = rf_result.get("recommended_crop")
        rf_confidence = float(rf_result.get("crop_confidence", 0.0))
        fl_crop = fl_result.get("recommended_crop")
        fl_confidence = float(fl_result.get("confidence", 0.0))

        rf_yield = float(rf_result.get("expected_yield", 0.0))
        fl_yield = float(fl_result.get("expected_yield", 0.0))
        final_yield = _clip(0.6 * rf_yield + 0.4 * fl_yield, 0.1, 15.0)

        rf_sun = float(rf_result.get("sunlight_hours", 6.0))
        fl_sun = float(fl_result.get("sunlight_hours", 6.0))
        final_sun = _clip(0.6 * rf_sun + 0.4 * fl_sun, 3.0, 12.0)

        rf_irr = rf_result.get("irrigation_type")
        fl_irr = fl_result.get("irrigation_type")
        fl_irr_conf = float(fl_result.get("irrigation_type_confidence", 0.0))
        if rf_irr == fl_irr:
            final_irr = rf_irr
        elif fl_irr_conf >= 0.75:
            final_irr = fl_irr
        else:
            final_irr = rf_irr or "sprinkler"

        rf_need = float(rf_result.get("irrigation_needed", 5.0))
        fl_need = float(fl_result.get("irrigation_needed", 5.0))
        final_need = _clip(0.6 * rf_need + 0.4 * fl_need, 0.0, 20.0)

        merged_top3 = self._merge_top3(rf_result, fl_result)

        final_conf = float(decision["final_confidence"])

        return {
            "success": True,
            "model_type": "ensemble",
            "recommended_crop": decision["final_crop"],
            "confidence": round(final_conf, 4),
            "confidence_pct": f"{final_conf * 100:.1f}%",
            "expected_yield": round(final_yield, 2),
            "sunlight_hours": round(final_sun, 1),
            "sunlight_hours_display": f"{round(final_sun, 1)} hours/day",
            "irrigation_type": final_irr,
            "irrigation_needed": round(final_need, 2),
            "irrigation_needed_display": f"{round(final_need, 2)} mm/day",
            "top_3_predictions": merged_top3,
            "decision_engine": {
                "method": "three_strategy_hybrid",
                "description": (
                    "Combines centralized Random Forest with a 28-state federated model "
                    "using Simple Voting, Weighted Ensemble (RF 60%/FL 40%), and "
                    "Confidence Gating (FL threshold 85%). Final answer by majority vote "
                    "across all 3 strategies."
                ),
                "strategy_detail": decision["strategy_detail"],
                "all_strategies_agree": decision["all_strategies_agree"],
            },
            "individual_model_outputs": {
                "random_forest": {
                    "recommended_crop": rf_crop,
                    "confidence": round(rf_confidence, 4),
                    "confidence_pct": f"{rf_confidence * 100:.1f}%",
                    "expected_yield": rf_result.get("expected_yield"),
                    "sunlight_hours": rf_result.get("sunlight_hours"),
                    "irrigation_type": rf_result.get("irrigation_type"),
                    "irrigation_needed": rf_result.get("irrigation_needed"),
                    "model_type": "Random Forest (Centralized)",
                    "failed": rf_failed,
                },
                "federated_model": {
                    "recommended_crop": fl_crop,
                    "confidence": round(fl_confidence, 4),
                    "confidence_pct": f"{fl_confidence * 100:.1f}%",
                    "expected_yield": fl_result.get("expected_yield"),
                    "sunlight_hours": fl_result.get("sunlight_hours"),
                    "irrigation_type": fl_result.get("irrigation_type"),
                    "irrigation_needed": fl_result.get("irrigation_needed"),
                    "model_type": "AdvisorNet (28-State Federated)",
                    "states_trained": 28,
                    "privacy_note": (
                        "Trained across 28 Indian states. Raw farm data never left any "
                        "state device during training."
                    ),
                    "failed": fl_failed,
                },
            },
            "agreement_analysis": {
                "rf_fl_agree": bool(rf_crop == fl_crop),
                "all_strategies_agree": decision["all_strategies_agree"],
                "confidence_level": (
                    "very_high" if final_conf >= 0.85 else "high" if final_conf >= 0.70 else "medium" if final_conf >= 0.50 else "low"
                ),
                "recommendation": (
                    "All models agree — high confidence."
                    if rf_crop == fl_crop
                    else "Models gave different predictions. The decision engine resolved the conflict using data from 28 Indian state models. Consider additional soil testing if confidence is below 70%."
                ),
            },
            "privacy_note": (
                "TerraMind ran both a centralized Random Forest and a privacy-preserving "
                "federated model trained across all 28 Indian states. Your farm data was "
                "never stored on any central server."
            ),
        }
