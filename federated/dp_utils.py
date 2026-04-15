"""Differential privacy utilities for TerraMind federated simulation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from opacus import PrivacyEngine


def wrap_model_with_dp(
    model: Any,
    optimizer: Any,
    train_loader: Any,
    target_epsilon: float = 10.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    num_epochs: int = 3,
) -> Tuple[Any, Any, Any, PrivacyEngine]:
    """Wrap model, optimizer, and data loader with Opacus DP engine."""
    privacy_engine = PrivacyEngine()

    private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=num_epochs,
        max_grad_norm=max_grad_norm,
    )

    return private_model, private_optimizer, private_loader, privacy_engine


def get_privacy_spent(privacy_engine: PrivacyEngine) -> Dict[str, Any]:
    """Return DP budget summary with farmer-friendly interpretation."""
    epsilon = float(privacy_engine.get_epsilon(delta=1e-5))
    epsilon_rounded = round(epsilon, 4)

    interpretation = (
        "Differential privacy adds carefully controlled noise so that no single farm record "
        "can significantly influence model updates. Lower epsilon generally means stronger "
        "privacy protection, while slightly reducing model accuracy."
    )

    privacy_statement = (
        "This model was trained across 28 Indian states. No raw farm data from any state "
        "was ever shared with the central server. With differential privacy enabled, even "
        "the weight updates are mathematically bounded — "
        f"an (epsilon={epsilon_rounded}, delta=1e-5) guarantee."
    )

    return {
        "epsilon": epsilon_rounded,
        "delta": 1e-5,
        "interpretation": interpretation,
        "states_protected": 28,
        "privacy_statement": privacy_statement,
    }
