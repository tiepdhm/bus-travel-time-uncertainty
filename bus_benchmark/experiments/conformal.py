"""Post-Hoc Split Conformal Quantile Calibration (CQR).

Provides ConformalQuantileCalibrator for multi-horizon quantile time-series models
(such as Temporal Fusion Transformer) to ensure valid empirical coverage (1 - alpha)
on unseen data without altering point/median forecasts.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


def compute_picp(
    y_true: Union[np.ndarray, torch.Tensor],
    q_low: Union[np.ndarray, torch.Tensor],
    q_high: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Prediction Interval Coverage Probability (PICP).

    PICP = proportion of valid observations falling in [q_low, q_high].
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(q_low, torch.Tensor):
        q_low = q_low.detach().cpu().numpy()
    if isinstance(q_high, torch.Tensor):
        q_high = q_high.detach().cpu().numpy()

    valid = ~np.isnan(y_true) & ~np.isnan(q_low) & ~np.isnan(q_high)
    if not np.any(valid):
        return float("nan")

    y_val = y_true[valid]
    low_val = q_low[valid]
    high_val = q_high[valid]

    covered = (y_val >= low_val) & (y_val <= high_val)
    return float(np.mean(covered))


def compute_pinaw(
    q_low: Union[np.ndarray, torch.Tensor],
    q_high: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Prediction Interval Normalized / Mean Average Width (PINAW / Sharpness).

    PINAW = average interval width (q_high - q_low).
    """
    if isinstance(q_low, torch.Tensor):
        q_low = q_low.detach().cpu().numpy()
    if isinstance(q_high, torch.Tensor):
        q_high = q_high.detach().cpu().numpy()

    valid = ~np.isnan(q_low) & ~np.isnan(q_high)
    if not np.any(valid):
        return float("nan")

    width = q_high[valid] - q_low[valid]
    return float(np.mean(width))


def compute_winkler_score(
    y_true: Union[np.ndarray, torch.Tensor],
    q_low: Union[np.ndarray, torch.Tensor],
    q_high: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.10,
) -> float:
    """Compute Winkler Score (Interval Loss) for significance level alpha.

    W_i = (q_high - q_low)
          + (2 / alpha) * (q_low - y) * I(y < q_low)
          + (2 / alpha) * (y - q_high) * I(y > q_high)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(q_low, torch.Tensor):
        q_low = q_low.detach().cpu().numpy()
    if isinstance(q_high, torch.Tensor):
        q_high = q_high.detach().cpu().numpy()

    valid = ~np.isnan(y_true) & ~np.isnan(q_low) & ~np.isnan(q_high)
    if not np.any(valid):
        return float("nan")

    y_val = y_true[valid]
    low_val = q_low[valid]
    high_val = q_high[valid]

    width = high_val - low_val
    penalty_low = np.maximum(0.0, low_val - y_val) * (2.0 / alpha)
    penalty_high = np.maximum(0.0, y_val - high_val) * (2.0 / alpha)

    score = width + penalty_low + penalty_high
    return float(np.mean(score))


class ConformalQuantileCalibrator:
    """Conformal Quantile Regression (CQR) Calibrator.

    Calibrates multi-horizon predicted quantiles to achieve empirical coverage
    of at least (1 - alpha), while guaranteeing point forecasts (median) remain
    strictly untouched.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        nominal_coverage: Optional[float] = None,
        low_idx: int = 0,
        med_idx: int = 1,
        high_idx: int = 2,
        per_horizon: bool = True,
    ) -> None:
        """Initialize ConformalQuantileCalibrator.

        Args:
            alpha: Significance level (default 0.10 for 90% coverage).
            nominal_coverage: Optional 1 - alpha (e.g. 0.90). Overrides alpha if provided.
            low_idx: Quantile index for lower bound (default 0).
            med_idx: Quantile index for median / point forecast (default 1).
            high_idx: Quantile index for upper bound (default 2).
            per_horizon: If True, calculates separate threshold q_alpha per forecast horizon.
        """
        if nominal_coverage is not None:
            if not 0.0 < nominal_coverage < 1.0:
                raise ValueError(f"nominal_coverage must be between 0 and 1, got {nominal_coverage}")
            self.alpha = float(1.0 - nominal_coverage)
        else:
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
            self.alpha = float(alpha)

        self.low_idx = low_idx
        self.med_idx = med_idx
        self.high_idx = high_idx
        self.per_horizon = per_horizon

        # Computed during fit
        self.q_hat: Optional[Union[float, np.ndarray]] = None
        self.is_fitted: bool = False

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    def _to_numpy(self, arr: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, bool]:
        """Convert input to numpy array and track if it was a PyTorch tensor."""
        is_torch = isinstance(arr, torch.Tensor)
        if is_torch:
            arr_np = arr.detach().cpu().numpy()
        else:
            arr_np = np.asarray(arr)
        return arr_np, is_torch

    def check_and_fix_monotonicity(self, q: np.ndarray) -> np.ndarray:
        """Ensure monotonic quantiles (q_low <= q_med <= q_high) without modifying q_med.

        If crossing occurs, adjusts q_low and q_high around q_med so that
        q_low <= q_med <= q_high, strictly preserving q_med as the point forecast.
        """
        q_copy = q.copy()
        q_low = q_copy[..., self.low_idx]
        q_med = q_copy[..., self.med_idx]
        q_high = q_copy[..., self.high_idx]

        # Ensure q_low <= q_high
        real_low = np.minimum(q_low, q_high)
        real_high = np.maximum(q_low, q_high)

        # Enforce bounds around median without altering median
        q_copy[..., self.low_idx] = np.minimum(real_low, q_med)
        q_copy[..., self.high_idx] = np.maximum(real_high, q_med)
        return q_copy

    def _compute_conformal_quantile(self, scores: np.ndarray, alpha: float) -> float:
        """Compute the empirical adjusted quantile threshold q_alpha from nonconformity scores."""
        scores_valid = scores[~np.isnan(scores)]
        n_val = len(scores_valid)
        if n_val == 0:
            raise ValueError("Cannot compute conformal threshold: no valid non-NaN scores in validation set.")

        # p = ceil((n_val + 1) * (1 - alpha)) / n_val
        p = np.ceil((n_val + 1) * (1.0 - alpha)) / n_val
        p = float(np.clip(p, 0.0, 1.0))

        # Use higher quantile to guarantee conservative finite-sample coverage
        q_alpha = float(np.quantile(scores_valid, p, method="higher"))
        return max(0.0, q_alpha)

    def fit(
        self,
        y_val: Union[np.ndarray, torch.Tensor],
        q_val_raw: Union[np.ndarray, torch.Tensor],
        alpha: Optional[float] = None,
        nominal_coverage: Optional[float] = None,
    ) -> "ConformalQuantileCalibrator":
        """Compute conformal threshold(s) q_alpha on validation / calibration set.

        Args:
            y_val: Ground truth values on validation set.
                   Shape (N, Horizon, Num_Links) or (N, Horizon) or (N,).
            q_val_raw: Raw predicted quantiles on validation set.
                       Shape (N, Horizon, Num_Links, Q) or (N, Horizon, Q) or (N, Q).
            alpha: Optional override for significance level.
            nominal_coverage: Optional override for nominal coverage (1 - alpha).

        Returns:
            self
        """
        if nominal_coverage is not None:
            self.alpha = float(1.0 - nominal_coverage)
        elif alpha is not None:
            self.alpha = float(alpha)

        y_np, _ = self._to_numpy(y_val)
        q_np, _ = self._to_numpy(q_val_raw)

        if q_np.shape[-1] <= max(self.low_idx, self.med_idx, self.high_idx):
            raise ValueError(
                f"q_val_raw last dimension ({q_np.shape[-1]}) is smaller than specified indices "
                f"low_idx={self.low_idx}, med_idx={self.med_idx}, high_idx={self.high_idx}"
            )

        # Step 1: Monotonicity check & crossing prevention
        q_np = self.check_and_fix_monotonicity(q_np)

        q_low = q_np[..., self.low_idx]
        q_high = q_np[..., self.high_idx]

        # Step 2: Nonconformity scores
        # s_i = max(q_low_i - y_i, y_i - q_high_i, 0)
        s_low = q_low - y_np
        s_high = y_np - q_high
        scores = np.maximum(np.maximum(s_low, s_high), 0.0)

        # Step 3: Threshold computation
        # Multi-horizon handling
        if self.per_horizon and q_np.ndim >= 3:
            # Assume dimension 1 is the horizon dimension: shape (N, Horizon, ...)
            n_horizons = q_np.shape[1]
            q_hat_per_h = []
            for h_idx in range(n_horizons):
                scores_h = scores[:, h_idx, ...]
                q_hat_h = self._compute_conformal_quantile(scores_h, self.alpha)
                q_hat_per_h.append(q_hat_h)
            self.q_hat = np.array(q_hat_per_h, dtype=float)
        else:
            # Global pooled threshold
            self.q_hat = self._compute_conformal_quantile(scores, self.alpha)

        self.is_fitted = True
        return self

    def calibrate(
        self,
        q_test_raw: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply conformal threshold(s) to test quantile predictions.

        Guarantees that median / point forecast (q_med) is strictly unchanged.

        Args:
            q_test_raw: Raw quantile predictions on test set.

        Returns:
            Calibrated quantiles with same type (NumPy or PyTorch) and shape as input.
        """
        if not self.is_fitted or self.q_hat is None:
            raise RuntimeError("Calibrator has not been fitted. Call .fit() first.")

        q_np, is_torch = self._to_numpy(q_test_raw)
        q_calibrated = self.check_and_fix_monotonicity(q_np)

        # Calibrate bounds
        if isinstance(self.q_hat, np.ndarray) and self.per_horizon and q_calibrated.ndim >= 3:
            # per-horizon adjustment
            # q_calibrated shape is e.g. (N, T, H, Q) or (N, T, Q)
            n_horizons = q_calibrated.shape[1]
            if len(self.q_hat) != n_horizons:
                raise ValueError(
                    f"Horizon mismatch: calibrator has {len(self.q_hat)} thresholds, "
                    f"but test set has {n_horizons} horizons."
                )

            for h_idx in range(n_horizons):
                adj = self.q_hat[h_idx]
                q_calibrated[:, h_idx, ..., self.low_idx] = (
                    q_calibrated[:, h_idx, ..., self.low_idx] - adj
                )
                q_calibrated[:, h_idx, ..., self.high_idx] = (
                    q_calibrated[:, h_idx, ..., self.high_idx] + adj
                )
                # IMPORTANT: med_idx remains strictly untouched!
        else:
            adj = float(self.q_hat if isinstance(self.q_hat, (int, float)) else self.q_hat[0])
            q_calibrated[..., self.low_idx] = q_calibrated[..., self.low_idx] - adj
            q_calibrated[..., self.high_idx] = q_calibrated[..., self.high_idx] + adj
            # IMPORTANT: med_idx remains strictly untouched!

        if is_torch:
            return torch.from_numpy(q_calibrated).to(dtype=q_test_raw.dtype, device=q_test_raw.device)
        return q_calibrated

    def evaluate(
        self,
        y_test: Union[np.ndarray, torch.Tensor],
        q_test_calibrated: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Union[float, Dict[int, float]]]:
        """Evaluate coverage, width, and Winkler score on test predictions.

        Returns:
            Dictionary containing overall and per-horizon metrics:
            - 'picp': Empirical coverage rate
            - 'pinaw': Mean interval width
            - 'winkler_score': Winkler interval loss score
            - 'per_horizon': Dict of metrics per horizon index if multi-horizon
        """
        y_np, _ = self._to_numpy(y_test)
        q_np, _ = self._to_numpy(q_test_calibrated)

        q_low = q_np[..., self.low_idx]
        q_high = q_np[..., self.high_idx]

        picp_overall = compute_picp(y_np, q_low, q_high)
        pinaw_overall = compute_pinaw(q_low, q_high)
        winkler_overall = compute_winkler_score(y_np, q_low, q_high, alpha=self.alpha)

        results: Dict[str, Union[float, Dict[int, float]]] = {
            "picp": picp_overall,
            "pinaw": pinaw_overall,
            "winkler_score": winkler_overall,
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
        }

        # Per-horizon evaluation if multi-horizon
        if q_np.ndim >= 3:
            n_horizons = q_np.shape[1]
            per_h_dict = {}
            for h in range(n_horizons):
                y_h = y_np[:, h, ...]
                q_low_h = q_low[:, h, ...]
                q_high_h = q_high[:, h, ...]
                per_h_dict[h] = {
                    "picp": compute_picp(y_h, q_low_h, q_high_h),
                    "pinaw": compute_pinaw(q_low_h, q_high_h),
                    "winkler_score": compute_winkler_score(y_h, q_low_h, q_high_h, alpha=self.alpha),
                }
            results["per_horizon"] = per_h_dict

        return results
