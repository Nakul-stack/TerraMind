"""Flower client implementation for TerraMind 28-state federated training."""

from __future__ import annotations

from typing import Dict, Tuple

import flwr as fl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from federated.model import AdvisorNet, get_model_parameters, set_model_parameters


class TerraMindFLClient(fl.client.NumPyClient):
    """Federated client representing one Indian state partition."""

    def __init__(
        self,
        state_name: str,
        client_id: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_classes: int = 22,
        num_irrigation_types: int = 4,
    ) -> None:
        self.state_name = state_name
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model = AdvisorNet(
            num_classes=num_classes,
            num_irrigation_types=num_irrigation_types,
        )

        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        print(
            f"[{self.state_name}] Client initialized | "
            f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)}"
        )

    def get_parameters(self, config):
        """Return model parameters in Flower NumPy format."""
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        """Run local multi-task training and return updated weights + metrics."""
        set_model_parameters(self.model, parameters)
        self.model.train()

        local_epochs = int(config.get("local_epochs", 3))
        learning_rate = float(config.get("learning_rate", 0.001))

        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, local_epochs))

        total_correct_crop = 0
        total_crop_samples = 0
        total_correct_irr = 0
        total_irr_samples = 0

        total_classification_loss = 0.0
        total_yield_mse = 0.0
        total_sunlight_mse = 0.0
        total_irr_needed_mse = 0.0
        total_batches = 0

        for _epoch in range(local_epochs):
            for batch in self.train_loader:
                (
                    X,
                    y_crop,
                    y_yield,
                    y_sun,
                    y_irr_type,
                    y_irr_needed,
                ) = batch

                X = X.to(self.device)
                y_crop = y_crop.to(self.device)
                y_yield = y_yield.to(self.device)
                y_sun = y_sun.to(self.device)
                y_irr_type = y_irr_type.to(self.device)
                y_irr_needed = y_irr_needed.to(self.device)

                optimizer.zero_grad()

                (
                    crop_logits,
                    yield_pred,
                    sun_pred,
                    irr_type_logits,
                    irr_needed_pred,
                ) = self.model(X)

                loss_crop = self.ce_loss(crop_logits, y_crop)
                loss_yield = self.mse_loss(yield_pred, y_yield)
                loss_sunlight = self.mse_loss(sun_pred, y_sun)
                loss_irr_type = self.ce_loss(irr_type_logits, y_irr_type)
                loss_irr_needed = self.mse_loss(irr_needed_pred, y_irr_needed)

                total_loss = (
                    1.0 * loss_crop
                    + 0.3 * loss_yield
                    + 0.2 * loss_sunlight
                    + 0.3 * loss_irr_type
                    + 0.2 * loss_irr_needed
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_classification_loss += float(loss_crop.item())
                total_yield_mse += float(loss_yield.item())
                total_sunlight_mse += float(loss_sunlight.item())
                total_irr_needed_mse += float(loss_irr_needed.item())
                total_batches += 1

                crop_preds = torch.argmax(crop_logits, dim=1)
                irr_preds = torch.argmax(irr_type_logits, dim=1)

                total_correct_crop += int((crop_preds == y_crop).sum().item())
                total_crop_samples += int(y_crop.size(0))

                total_correct_irr += int((irr_preds == y_irr_type).sum().item())
                total_irr_samples += int(y_irr_type.size(0))

            scheduler.step()

        train_accuracy = (total_correct_crop / total_crop_samples) if total_crop_samples else 0.0
        irr_type_accuracy = (total_correct_irr / total_irr_samples) if total_irr_samples else 0.0

        classification_loss = (total_classification_loss / total_batches) if total_batches else 0.0
        yield_mse = (total_yield_mse / total_batches) if total_batches else 0.0
        sunlight_mse = (total_sunlight_mse / total_batches) if total_batches else 0.0
        irr_needed_mse = (total_irr_needed_mse / total_batches) if total_batches else 0.0

        print(
            f"[{self.state_name}] Round done | "
            f"Acc: {train_accuracy:.3f} | Loss: {classification_loss:.4f}"
        )

        metrics = {
            "train_accuracy": float(train_accuracy),
            "classification_loss": float(classification_loss),
            "yield_mse": float(yield_mse),
            "sunlight_mse": float(sunlight_mse),
            "irr_type_accuracy": float(irr_type_accuracy),
            "irr_needed_mse": float(irr_needed_mse),
            "state": self.state_name,
        }

        return get_model_parameters(self.model), len(self.train_dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate local validation data and return weighted metrics for server aggregation."""
        set_model_parameters(self.model, parameters)
        self.model.eval()

        total_val_loss = 0.0
        total_batches = 0

        total_correct_crop = 0
        total_crop_samples = 0
        total_correct_irr = 0
        total_irr_samples = 0

        total_yield_mae = 0.0
        total_sunlight_mae = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                (
                    X,
                    y_crop,
                    y_yield,
                    y_sun,
                    y_irr_type,
                    y_irr_needed,
                ) = batch

                X = X.to(self.device)
                y_crop = y_crop.to(self.device)
                y_yield = y_yield.to(self.device)
                y_sun = y_sun.to(self.device)
                y_irr_type = y_irr_type.to(self.device)
                y_irr_needed = y_irr_needed.to(self.device)

                (
                    crop_logits,
                    yield_pred,
                    sun_pred,
                    irr_type_logits,
                    irr_needed_pred,
                ) = self.model(X)

                loss_crop = self.ce_loss(crop_logits, y_crop)
                loss_yield = self.mse_loss(yield_pred, y_yield)
                loss_sunlight = self.mse_loss(sun_pred, y_sun)
                loss_irr_type = self.ce_loss(irr_type_logits, y_irr_type)
                loss_irr_needed = self.mse_loss(irr_needed_pred, y_irr_needed)

                val_loss = (
                    1.0 * loss_crop
                    + 0.3 * loss_yield
                    + 0.2 * loss_sunlight
                    + 0.3 * loss_irr_type
                    + 0.2 * loss_irr_needed
                )

                total_val_loss += float(val_loss.item())
                total_batches += 1

                crop_preds = torch.argmax(crop_logits, dim=1)
                irr_preds = torch.argmax(irr_type_logits, dim=1)

                total_correct_crop += int((crop_preds == y_crop).sum().item())
                total_crop_samples += int(y_crop.size(0))

                total_correct_irr += int((irr_preds == y_irr_type).sum().item())
                total_irr_samples += int(y_irr_type.size(0))

                total_yield_mae += float(self.mae_loss(yield_pred, y_yield).item())
                total_sunlight_mae += float(self.mae_loss(sun_pred, y_sun).item())

        avg_val_loss = (total_val_loss / total_batches) if total_batches else 0.0
        accuracy = (total_correct_crop / total_crop_samples) if total_crop_samples else 0.0
        irr_type_accuracy = (total_correct_irr / total_irr_samples) if total_irr_samples else 0.0
        yield_mae = (total_yield_mae / total_batches) if total_batches else 0.0
        sunlight_mae = (total_sunlight_mae / total_batches) if total_batches else 0.0

        metrics = {
            "accuracy": float(accuracy),
            "yield_mae": float(yield_mae),
            "irr_type_accuracy": float(irr_type_accuracy),
            "sunlight_mae": float(sunlight_mae),
            "state": self.state_name,
        }

        return float(avg_val_loss), len(self.val_dataset), metrics
