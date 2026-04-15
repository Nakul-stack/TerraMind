"""Custom Flower server strategy for TerraMind 28-state federated simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import FitIns, Metrics, NDArrays, Parameters, Scalar


class TerraMindFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy with TerraMind-specific round configs and analytics."""

    def __init__(
        self,
        num_rounds: int,
        num_states: int = 28,
        results_dir: str = "federated/results",
    ) -> None:
        self.num_rounds = num_rounds
        self.num_states = num_states
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.round_history: List[Dict[str, object]] = []

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=5,
            min_evaluate_clients=5,
            min_available_clients=5,
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:
        """Configure per-round training hyperparameters and sample all available clients."""
        learning_rate = 0.001 * (0.5 ** (server_round / max(1, self.num_rounds)))
        local_epochs = 3 if server_round <= (self.num_rounds // 2) else 5

        fit_config: Dict[str, Scalar] = {
            "learning_rate": float(learning_rate),
            "local_epochs": int(local_epochs),
            "server_round": int(server_round),
        }

        fit_ins = FitIns(parameters, fit_config)

        available_clients = client_manager.num_available()
        sample_size, min_num_clients = self.num_fit_clients(available_clients)
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates and persist per-round per-state metrics history."""
        if failures:
            print(f"[Server] Round {server_round}: {len(failures)}/{self.num_states} states failed")

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        round_entry: Dict[str, object] = {
            "round": server_round,
            "num_participants": len(results),
            "num_failures": len(failures),
            "participants": [],
        }

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics if fit_res.metrics is not None else {}
            state_name = str(metrics.get("state", client_proxy.cid))

            round_entry["participants"].append(
                {
                    "client_id": client_proxy.cid,
                    "state": state_name,
                    "num_examples": fit_res.num_examples,
                    "train_accuracy": float(metrics.get("train_accuracy", 0.0)),
                    "classification_loss": float(metrics.get("classification_loss", 0.0)),
                    "yield_mse": float(metrics.get("yield_mse", 0.0)),
                    "sunlight_mse": float(metrics.get("sunlight_mse", 0.0)),
                    "irr_type_accuracy": float(metrics.get("irr_type_accuracy", 0.0)),
                    "irr_needed_mse": float(metrics.get("irr_needed_mse", 0.0)),
                }
            )

        self.round_history.append(round_entry)

        try:
            with open(self.results_dir / "round_history.json", "w", encoding="utf-8") as f:
                json.dump(self.round_history, f, indent=2)
        except Exception as exc:
            print(f"[Server] Failed to save round_history.json: {exc}")

        print(
            f"[Server] Round {server_round} aggregated | "
            f"{len(results)}/{self.num_states} states participated"
        )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Compute weighted averages for evaluation metrics across client states."""
        if not results:
            if failures:
                print(f"[Server] Round {server_round}: no evaluation results due to failures")
            return None, {}

        total_examples = sum(res.num_examples for _, res in results)
        if total_examples <= 0:
            return None, {}

        weighted_loss = 0.0
        weighted_acc = 0.0
        weighted_yield_mae = 0.0
        weighted_irr_acc = 0.0

        for _, eval_res in results:
            num_ex = eval_res.num_examples
            metrics = eval_res.metrics if eval_res.metrics is not None else {}

            weighted_loss += float(eval_res.loss) * num_ex
            weighted_acc += float(metrics.get("accuracy", 0.0)) * num_ex
            weighted_yield_mae += float(metrics.get("yield_mae", 0.0)) * num_ex
            weighted_irr_acc += float(metrics.get("irr_type_accuracy", 0.0)) * num_ex

        avg_loss = weighted_loss / total_examples
        avg_acc = weighted_acc / total_examples
        avg_yield_mae = weighted_yield_mae / total_examples
        avg_irr_acc = weighted_irr_acc / total_examples

        if failures:
            print(f"[Server] Round {server_round}: {len(failures)}/{self.num_states} states failed")

        print(
            f"[Server] Round {server_round} | "
            f"Accuracy: {avg_acc:.4f} | "
            f"Yield MAE: {avg_yield_mae:.4f} | "
            f"Irr Acc: {avg_irr_acc:.4f} | "
            f"States: {len(results)}/{self.num_states}"
        )

        out_metrics: Dict[str, Scalar] = {
            "accuracy": float(avg_acc),
            "yield_mae": float(avg_yield_mae),
            "irr_type_accuracy": float(avg_irr_acc),
            "num_participants": int(len(results)),
        }
        return float(avg_loss), out_metrics

    def get_state_participation_summary(self) -> Dict[str, int]:
        """Return number of rounds each state participated in."""
        counts: Dict[str, int] = {}
        for round_entry in self.round_history:
            participants = round_entry.get("participants", [])
            if not isinstance(participants, list):
                continue
            for item in participants:
                if not isinstance(item, dict):
                    continue
                state = str(item.get("state", "unknown"))
                counts[state] = counts.get(state, 0) + 1
        return counts
