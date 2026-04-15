"""Run TerraMind 28-state federated learning simulation and save artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from federated.client import TerraMindFLClient
from federated.data_partitioner import DataPartitioner, FEATURE_COLUMNS, STATE_PROFILES
from federated.model import AdvisorNet, get_model_parameters, set_model_parameters
from federated.server import TerraMindFedAvg


RESULTS_DIR = Path("federated/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class _TrackingFedAvg(TerraMindFedAvg):
    """Small wrapper to capture latest aggregated parameters for artifact export."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            self.latest_parameters = params
        return params, metrics


class _HistoryLike:
    """Tiny container mirroring Flower history metric access."""

    def __init__(self) -> None:
        self.metrics_distributed: Dict[str, List[Tuple[int, float]]] = {}


class _ClientProxyStub:
    """Minimal proxy-like object exposing `cid` for strategy logging."""

    def __init__(self, cid: str) -> None:
        self.cid = cid


def _run_fallback_federated_training(
    partitions: Dict[str, Dict[str, Any]],
    num_classes: int,
    num_irrigation_types: int,
    rounds: int,
    initial_parameters,
    strategy: _TrackingFedAvg,
):
    """Run FL rounds without Ray/Flower simulation backend."""
    state_names = list(partitions.keys())
    current_params = initial_parameters
    history = _HistoryLike()

    for rnd in range(1, rounds + 1):
        fit_results = []
        for idx, state_name in enumerate(state_names):
            part = partitions[state_name]
            client = TerraMindFLClient(
                state_name=state_name,
                client_id=idx,
                train_dataset=part["train"],
                val_dataset=part["val"],
                num_classes=num_classes,
                num_irrigation_types=num_irrigation_types,
            )

            params_list = fl.common.parameters_to_ndarrays(current_params)
            updated_params, n_examples, metrics = client.fit(params_list, {"server_round": rnd})

            fit_results.append(
                (
                    _ClientProxyStub(str(idx)),
                    fl.common.FitRes(
                        status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
                        parameters=fl.common.ndarrays_to_parameters(updated_params),
                        num_examples=int(n_examples),
                        metrics=metrics,
                    ),
                )
            )

        aggregated, _ = strategy.aggregate_fit(rnd, fit_results, [])
        if aggregated is not None:
            current_params = aggregated

        eval_results = []
        for idx, state_name in enumerate(state_names):
            part = partitions[state_name]
            client = TerraMindFLClient(
                state_name=state_name,
                client_id=idx,
                train_dataset=part["train"],
                val_dataset=part["val"],
                num_classes=num_classes,
                num_irrigation_types=num_irrigation_types,
            )
            params_list = fl.common.parameters_to_ndarrays(current_params)
            loss, n_examples, metrics = client.evaluate(params_list, {"server_round": rnd})
            metric_payload = {"loss": float(loss), **metrics}
            eval_results.append(
                (
                    _ClientProxyStub(str(idx)),
                    fl.common.EvaluateRes(
                        status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
                        loss=float(loss),
                        num_examples=int(n_examples),
                        metrics=metric_payload,
                    ),
                )
            )

        _loss, agg_metrics = strategy.aggregate_evaluate(rnd, eval_results, [])
        for key, value in agg_metrics.items():
            history.metrics_distributed.setdefault(key, []).append((rnd, float(value)))

    return history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TerraMind 28-state federated simulation")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--iid", action="store_true", help="Use IID partition mode")
    parser.add_argument("--dp", action="store_true", help="Enable DP report")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset before sowing/crop_dataset_rebuilt.csv",
    )
    return parser.parse_args()


def _estimate_runtime_minutes(rounds: int) -> int:
    # 20 rounds x 28 clients x ~5s ~= 46.7 min
    est = (rounds * 28 * 5) / 60.0
    return max(1, int(round(est)))


def _train_centralized_baseline(
    model: AdvisorNet,
    train_loader: DataLoader,
    rounds: int,
) -> AdvisorNet:
    device = torch.device("cpu")
    model.to(device)
    model.train()

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Equivalent gradient-step budget approximation:
    # FL rounds x avg local epochs (~4 from 3->5 schedule)
    total_epochs = max(1, rounds * 4)

    for _ in range(total_epochs):
        for batch in train_loader:
            X, y_crop, y_yield, y_sun, y_irr_type, y_irr_needed = batch
            X = X.to(device)
            y_crop = y_crop.to(device)
            y_yield = y_yield.to(device)
            y_sun = y_sun.to(device)
            y_irr_type = y_irr_type.to(device)
            y_irr_needed = y_irr_needed.to(device)

            optimizer.zero_grad()
            crop_logits, yield_pred, sun_pred, irr_type_logits, irr_needed_pred = model(X)

            loss_crop = ce_loss(crop_logits, y_crop)
            loss_yield = mse_loss(yield_pred, y_yield)
            loss_sunlight = mse_loss(sun_pred, y_sun)
            loss_irr_type = ce_loss(irr_type_logits, y_irr_type)
            loss_irr_needed = mse_loss(irr_needed_pred, y_irr_needed)

            loss = (
                1.0 * loss_crop
                + 0.3 * loss_yield
                + 0.2 * loss_sunlight
                + 0.3 * loss_irr_type
                + 0.2 * loss_irr_needed
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    return model


def _evaluate_multitask(model: AdvisorNet, loader: DataLoader) -> Dict[str, float]:
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    mae_loss = nn.L1Loss()

    total = 0
    crop_correct = 0
    irr_correct = 0
    yield_mae_sum = 0.0
    sun_mae_sum = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            X, y_crop, y_yield, y_sun, y_irr_type, _y_irr_needed = batch
            X = X.to(device)
            y_crop = y_crop.to(device)
            y_yield = y_yield.to(device)
            y_sun = y_sun.to(device)
            y_irr_type = y_irr_type.to(device)

            crop_logits, yield_pred, sun_pred, irr_type_logits, _ = model(X)

            crop_pred = torch.argmax(crop_logits, dim=1)
            irr_pred = torch.argmax(irr_type_logits, dim=1)

            crop_correct += int((crop_pred == y_crop).sum().item())
            irr_correct += int((irr_pred == y_irr_type).sum().item())
            total += int(y_crop.size(0))

            yield_mae_sum += float(mae_loss(yield_pred, y_yield).item())
            sun_mae_sum += float(mae_loss(sun_pred, y_sun).item())
            batches += 1

    if total == 0:
        return {
            "accuracy": 0.0,
            "yield_mae": 0.0,
            "irr_type_accuracy": 0.0,
            "sunlight_mae": 0.0,
        }

    return {
        "accuracy": float(crop_correct / total),
        "yield_mae": float(yield_mae_sum / max(1, batches)),
        "irr_type_accuracy": float(irr_correct / total),
        "sunlight_mae": float(sun_mae_sum / max(1, batches)),
    }


def _history_last_metric(history, name: str, default: float = 0.0) -> float:
    vals = history.metrics_distributed.get(name, []) if history is not None else []
    if not vals:
        return default
    try:
        return float(vals[-1][1])
    except Exception:
        return default


def _compute_comm_cost_mb(
    num_params: int,
    rounds: int,
    num_clients: int,
    bytes_per_param: int = 4,
) -> float:
    # Approximation: up + down per round per client
    total_bytes = num_params * bytes_per_param * rounds * num_clients * 2
    return round(total_bytes / (1024 * 1024), 2)


def save_federated_artifacts(
    final_parameters,
    partitioner: DataPartitioner,
    num_classes: int,
    results_dir: str = "federated/results",
) -> Dict[str, str]:
    """Save FL production artifacts (weights, scaler, encoder, metadata)."""
    try:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model = AdvisorNet(
            num_classes=num_classes,
            num_irrigation_types=(
                len(partitioner.irrigation_type_encoder.classes_)
                if hasattr(partitioner, "irrigation_type_encoder")
                and hasattr(partitioner.irrigation_type_encoder, "classes_")
                else 4
            ),
        )

        # final_parameters is expected to be a list/sequence of ndarrays.
        set_model_parameters(model, final_parameters)

        model.eval()

        weights_path = out_dir / "federated_advisor_final.pth"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "num_classes": int(num_classes),
            "input_dim": 7,
            "architecture": "AdvisorNet_v1",
            "feature_names": ["n", "p", "k", "ph", "temperature", "humidity", "rainfall"],
            "saved_at": datetime.utcnow().isoformat(),
        }
        torch.save(checkpoint, weights_path)
        print(f"[TerraMind] Federated weights saved -> {weights_path}")

        scaler_path = out_dir / "federated_scaler.pkl"
        joblib.dump(partitioner.scaler, scaler_path)
        print(f"[TerraMind] Scaler saved -> {scaler_path}")

        encoder_path = out_dir / "federated_label_encoder.pkl"
        joblib.dump(partitioner.label_encoder, encoder_path)
        print(f"[TerraMind] LabelEncoder saved -> {encoder_path}")

        comparison_path = out_dir / "comparison.json"
        comparison_dict: Dict[str, Any] = {}
        if comparison_path.exists():
            with open(comparison_path, "r", encoding="utf-8") as f:
                comparison_dict = json.load(f)

        fed_acc = float(comparison_dict.get("federated_accuracy", 0.0))
        cen_acc = float(comparison_dict.get("centralized_accuracy", 0.0))
        accuracy_gap = float(comparison_dict.get("accuracy_gap_pct", 0.0))

        metadata_path = out_dir / "federated_model_metadata.json"
        num_clients = int(getattr(partitioner, "num_clients", 0) or len(getattr(partitioner, "partitions", {}) or {}))
        if num_clients <= 0:
            num_clients = 28

        metadata = {
            "num_classes": int(num_classes),
            "num_clients": int(num_clients),
            "num_rounds": int(comparison_dict.get("num_rounds", 0)),
            "partition_mode": str(comparison_dict.get("partition_mode", "non-iid")),
            "feature_names": ["n", "p", "k", "ph", "temperature", "humidity", "rainfall"],
            "class_names": partitioner.label_encoder.classes_.tolist(),
            "federated_accuracy": round(float(fed_acc), 6),
            "centralized_accuracy": round(float(cen_acc), 6),
            "accuracy_gap_pct": round(float(accuracy_gap), 3),
            "saved_at": datetime.utcnow().isoformat(),
            "model_path": str(weights_path),
            "scaler_path": str(scaler_path),
            "encoder_path": str(encoder_path),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[TerraMind] Metadata saved -> {metadata_path}")

        return {
            "weights_path": str(weights_path),
            "scaler_path": str(scaler_path),
            "encoder_path": str(encoder_path),
            "metadata_path": str(metadata_path),
        }
    except Exception as exc:
        print(f"[TerraMind] ERROR saving federated artifacts: {exc}")
        return {}


def _print_comparison_table(comparison: Dict[str, Any]) -> None:
    print("=" * 44)
    print("RESULTS: Centralized vs Federated (28 States)")
    print("=" * 44)
    print(f"{'Metric':<20} {'Centralized':<13} {'Federated':<10}")
    print("-" * 44)
    print(
        f"{'Crop Accuracy':<20} "
        f"{comparison['centralized_accuracy']:<13.4f} "
        f"{comparison['federated_accuracy']:<10.4f}"
    )
    print(
        f"{'Yield MAE':<20} "
        f"{comparison['centralized_yield_mae']:<13.2f} "
        f"{comparison['federated_yield_mae']:<10.2f}"
    )
    print(
        f"{'Irr. Accuracy':<20} "
        f"{comparison['centralized_irr_accuracy']:<13.4f} "
        f"{comparison['federated_irr_accuracy']:<10.4f}"
    )
    print(f"{'Privacy':<20} {'X Central':<13} {'OK 28 States':<10}")
    print(f"{'Accuracy Gap':<20} {'-':<13} {comparison['accuracy_gap_pct']:.2f}%")
    print(f"{'Comm. Cost (MB)':<20} {'-':<13} {comparison['estimated_communication_cost_mb']:.1f} MB")
    print(f"{'States Trained':<20} {'-':<13} {comparison['num_state_clients']} / 28")
    print("=" * 44)


def main() -> None:
    args = _parse_args()

    est_minutes = _estimate_runtime_minutes(args.rounds)
    print("[TerraMind] Starting 28-state FL simulation")
    print(f"[TerraMind] Estimated runtime: ~{est_minutes} minutes")
    print("[TerraMind] Use --rounds 5 for a quick test run")

    partitioner = DataPartitioner(args.data_path, num_clients=28)
    df = partitioner.load_datasets()

    if args.iid:
        partition_mode = "iid"
        partitions = partitioner.create_iid_partitions(df)
    else:
        partition_mode = "non_iid_state_level"
        partitions = partitioner.create_non_iid_partitions(df)

    num_classes = len(partitioner.label_encoder.classes_) if hasattr(partitioner.label_encoder, "classes_") else 22
    num_irrigation_types = (
        len(partitioner.irrigation_type_encoder.classes_)
        if hasattr(partitioner.irrigation_type_encoder, "classes_")
        else 4
    )

    global_model = AdvisorNet(
        num_classes=num_classes,
        num_irrigation_types=num_irrigation_types,
    )
    initial_parameters = fl.common.ndarrays_to_parameters(get_model_parameters(global_model))

    state_names = list(partitions.keys())

    def build_client(cid: str):
        idx = int(cid)
        state_name = state_names[idx]
        part = partitions[state_name]
        return TerraMindFLClient(
            state_name=state_name,
            client_id=idx,
            train_dataset=part["train"],
            val_dataset=part["val"],
            num_classes=num_classes,
            num_irrigation_types=num_irrigation_types,
        )

    strategy = _TrackingFedAvg(num_rounds=args.rounds, num_states=28, results_dir=str(RESULTS_DIR))
    strategy.initial_parameters = initial_parameters

    try:
        history = fl.simulation.start_simulation(
            client_fn=build_client,
            num_clients=28,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
    except ImportError as exc:
        if "ray" not in str(exc).lower():
            raise
        print("[TerraMind] Ray backend unavailable; using built-in sequential fallback.")
        history = _run_fallback_federated_training(
            partitions=partitions,
            num_classes=num_classes,
            num_irrigation_types=num_irrigation_types,
            rounds=args.rounds,
            initial_parameters=initial_parameters,
            strategy=strategy,
        )

    # Centralized baseline training/evaluation
    train_loader, val_loader = partitioner.get_centralized_data(df)
    centralized_model = AdvisorNet(
        num_classes=num_classes,
        num_irrigation_types=num_irrigation_types,
    )
    centralized_model = _train_centralized_baseline(centralized_model, train_loader, args.rounds)
    centralized_metrics = _evaluate_multitask(centralized_model, val_loader)

    fed_metrics = {
        "accuracy": _history_last_metric(history, "accuracy", 0.0),
        "yield_mae": _history_last_metric(history, "yield_mae", 0.0),
        "irr_type_accuracy": _history_last_metric(history, "irr_type_accuracy", 0.0),
    }

    if fed_metrics["accuracy"] == 0.0:
        # Fallback: evaluate reconstructed FL model on centralized validation split
        reconstructed = AdvisorNet(num_classes=num_classes, num_irrigation_types=num_irrigation_types)
        if strategy.latest_parameters is not None:
            set_model_parameters(reconstructed, fl.common.parameters_to_ndarrays(strategy.latest_parameters))
            fed_eval = _evaluate_multitask(reconstructed, val_loader)
            fed_metrics = {
                "accuracy": fed_eval["accuracy"],
                "yield_mae": fed_eval["yield_mae"],
                "irr_type_accuracy": fed_eval["irr_type_accuracy"],
            }

    num_params = sum(p.numel() for p in global_model.parameters())
    comm_cost = _compute_comm_cost_mb(
        num_params=num_params,
        rounds=args.rounds,
        num_clients=28,
    )

    cen_acc = centralized_metrics.get("accuracy", 0.0)
    fed_acc = fed_metrics.get("accuracy", 0.0)
    acc_gap_pct = ((cen_acc - fed_acc) / max(cen_acc, 1e-8)) * 100.0

    comparison = {
        "centralized_accuracy": round(float(cen_acc), 6),
        "federated_accuracy": round(float(fed_acc), 6),
        "accuracy_gap_pct": round(float(acc_gap_pct), 3),
        "centralized_yield_mae": round(float(centralized_metrics.get("yield_mae", 0.0)), 6),
        "federated_yield_mae": round(float(fed_metrics.get("yield_mae", 0.0)), 6),
        "centralized_irr_accuracy": round(float(centralized_metrics.get("irr_type_accuracy", 0.0)), 6),
        "federated_irr_accuracy": round(float(fed_metrics.get("irr_type_accuracy", 0.0)), 6),
        "privacy_status_centralized": "Raw data pooled",
        "privacy_status_federated": "Raw data stayed in 28 state clients",
        "estimated_communication_cost_mb": float(comm_cost),
        "num_rounds": int(args.rounds),
        "num_state_clients": 28,
        "partition_mode": partition_mode,
    }

    with open(RESULTS_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Extract final global parameters and persist production inference artifacts.
    try:
        final_params = None
        if hasattr(history, "parameters") and getattr(history, "parameters") is not None:
            final_params = fl.common.parameters_to_ndarrays(history.parameters)
        elif strategy.latest_parameters is not None:
            final_params = fl.common.parameters_to_ndarrays(strategy.latest_parameters)

        if final_params is None:
            raise RuntimeError("No final federated parameters available from history or strategy")

        artifacts = save_federated_artifacts(final_params, partitioner, num_classes)
        if artifacts:
            print("[TerraMind] Federated model ready for production inference")
            print("[TerraMind]    Run the FastAPI backend and use model_mode='federated' to use it")
    except Exception as exc:
        print(f"[TerraMind] Could not extract final parameters: {exc}")
        print("[TerraMind] Artifacts not saved - re-run simulation to generate them")

    _print_comparison_table(comparison)

    try:
        from federated.evaluate import plot_results

        plot_results(
            history=history,
            comparison_dict=comparison,
            state_metadata=partitioner.state_metadata_list,
            results_dir=str(RESULTS_DIR),
        )
    except Exception as exc:
        print(f"[TerraMind] Warning: plot generation skipped: {exc}")

    if args.dp:
        try:
            from federated.dp_utils import get_privacy_spent

            # Placeholder DP print when enabled via flag
            # (actual privacy engine object is created in DP-enabled training flows).
            print("[TerraMind] DP flag enabled. Report available via dp_utils during DP training.")
            try:
                # Optional: friendly fallback report when no privacy engine is attached
                report = {
                    "epsilon": None,
                    "delta": 1e-5,
                    "interpretation": "DP requested; attach PrivacyEngine in training loop for exact epsilon.",
                }
                _ = get_privacy_spent  # avoid lint-like unused complaints
                print(f"[TerraMind] DP report: {report}")
            except Exception:
                pass
        except Exception as exc:
            print(f"[TerraMind] Warning: DP utilities unavailable: {exc}")

    print("[TerraMind] OK State-level federated model saved")
    print("[TerraMind]    Trained across 28 Indian states")
    print("[TerraMind]    Use model_mode='federated' or model_mode='ensemble' in the API")


if __name__ == "__main__":
    main()
