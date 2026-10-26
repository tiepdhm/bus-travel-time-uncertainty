"""Quantile (Pinball) Loss for probabilistic forecasting.

Shared loss function used by TFT-based model trainers.
"""

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Combined quantile (pinball) loss.

    TF ref: QuantileLossCalculator L1059-1091 + utils.tensorflow_quantile_loss.
    Pinball loss per quantile q:
        L_q(y, ŷ) = max(q*(y - ŷ), (q-1)*(y - ŷ))

    Accepts:
        predictions: (B, T, H * num_quantiles)  — flat quantile output
        targets:     (B, T, H)                  — single ground truth

    The target is internally expanded to match each quantile slice.
    """

    def __init__(self, quantiles: list[float] = None):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        Q = len(self.quantiles)
        B, T, HQ = predictions.shape
        H = HQ // Q

        # Reshape predictions: (B, T, H, Q)
        preds = predictions.view(B, T, H, Q)
        # Expand targets: (B, T, H) → (B, T, H, 1)
        y = targets.unsqueeze(-1)

        # Pinball loss per quantile
        errors = y - preds  # (B, T, H, Q)
        loss = torch.tensor(0.0, device=predictions.device)

        for i, q in enumerate(self.quantiles):
            e = errors[..., i]
            loss = loss + torch.max(q * e, (q - 1.0) * e).mean()

        return loss
