"""AdvisorNet model definition and Flower parameter helpers for TerraMind FL."""

from collections import OrderedDict
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ADVISOR_NET_VERSION = "2.0"
NUM_STATES = 28


class AdvisorNet(nn.Module):
    """Multi-task tabular model for crop advisory predictions."""

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: Sequence[int] = (128, 64, 32),
        num_classes: int = 22,
        num_irrigation_types: int = 4,
    ) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims

        self.shared = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
        )

        # Head 1: crop classification
        self.crop_head = nn.Linear(h3, num_classes)

        # Head 2: yield regression
        self.yield_head = nn.Sequential(
            nn.Linear(h3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Head 3: sunlight regression (clipped to [3.0, 12.0])
        self.sunlight_head = nn.Sequential(
            nn.Linear(h3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Head 4: irrigation type classification
        self.irrigation_type_head = nn.Sequential(
            nn.Linear(h3, 16),
            nn.ReLU(),
            nn.Linear(16, num_irrigation_types),
        )

        # Head 5: irrigation needed regression (clipped to [0.0, 20.0])
        self.irrigation_needed_head = nn.Sequential(
            nn.Linear(h3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all Linear layers with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all five task heads from a shared representation."""
        features = self.shared(x)

        crop_logits = self.crop_head(features)
        yield_pred = self.yield_head(features).squeeze(-1)
        sunlight_pred = self.sunlight_head(features).squeeze(-1)
        irrigation_type_logits = self.irrigation_type_head(features)
        irrigation_needed_pred = self.irrigation_needed_head(features).squeeze(-1)

        sunlight_pred = torch.clamp(sunlight_pred, min=3.0, max=12.0)
        irrigation_needed_pred = torch.clamp(irrigation_needed_pred, min=0.0, max=20.0)

        return (
            crop_logits,
            yield_pred,
            sunlight_pred,
            irrigation_type_logits,
            irrigation_needed_pred,
        )


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model state dict parameters in Flower NumPy format."""
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: Sequence[np.ndarray]) -> nn.Module:
    """Load Flower NumPy-format parameters back into model state dict."""
    state_dict = model.state_dict()
    keys = list(state_dict.keys())

    if len(keys) != len(parameters):
        raise ValueError(
            f"Parameter length mismatch. Expected {len(keys)}, got {len(parameters)}"
        )

    new_state_dict = OrderedDict()
    for key, param in zip(keys, parameters):
        tensor = torch.tensor(param)
        if state_dict[key].dtype != tensor.dtype:
            tensor = tensor.to(dtype=state_dict[key].dtype)
        new_state_dict[key] = tensor

    model.load_state_dict(new_state_dict, strict=True)
    return model
