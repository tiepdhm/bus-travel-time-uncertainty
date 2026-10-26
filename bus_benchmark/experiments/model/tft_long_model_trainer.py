"""TFT Long Model Trainer.

Extends ModelTrainerBase with TFT-specific logic for Long Format:
  - Uses TFTLongDataset
  - Uses TFTLongModel
  - Reconstruction of matrix for Metrics
"""

import logging
from typing import Optional, Literal, Union
import torch
import torch.nn as nn
import numpy as np
import copy
import os

from bus_benchmark.experiments.fixed_interval_dataset import FixedIntervalDataset
from bus_benchmark.experiments.model.model_trainer_base import ModelTrainerBase
from bus_benchmark.experiments.model.quantile_loss import QuantileLoss
from bus_benchmark.experiments.tft_long_dataset import TFTLongDataset
from bus_benchmark.experiments.model.tft_long_model import TFTLongModel
from bus_benchmark.experiments.conformal import ConformalQuantileCalibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFTLongModelTrainer(ModelTrainerBase):
    def __init__(
        self,
        base_dataset: FixedIntervalDataset,
        n_epochs: int = 30,
        seq_len: int = 16,
        horizons: tuple[int, ...] = (1, 2, 4, 8),
        batch_size: int = 512,
        lr: float = 1e-3,
        patience: int = 10,
        factor: float = 0.5,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 5.0,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        num_quantiles: int = 3,
        num_stacks: int = 1,
        use_is_imputed: bool = True,
        use_link_id: bool = True,
        use_link_length: bool = True,
        use_tod: bool = True,
        use_dow: bool = True,
        use_stop_sequence: bool = False,
        use_pattern_profile: bool = False,
        data_set_name: Optional[str] = None,
        verbose: bool = True,
        print_metrics: Literal["overall", "per_horizon", "both"] = "both",
        eval_target_start_idx: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_quantiles = num_quantiles
        self.num_stacks = num_stacks
        self.use_is_imputed = use_is_imputed
        self.use_link_id = use_link_id
        self.use_link_length = use_link_length
        self.use_tod = use_tod
        self.use_dow = use_dow
        self.use_stop_sequence = use_stop_sequence
        self.use_pattern_profile = use_pattern_profile
        self.eval_target_start_idx = eval_target_start_idx

        # Conformal calibration configuration (enabled post-training via trainer.use_conformal(...))
        self.is_conformal_enabled: bool = False
        self.nominal_coverage: float = 0.80
        self.conformal_per_horizon: bool = True
        self.conformal_calibrator: Optional[ConformalQuantileCalibrator] = None
        self.calibrators: list[ConformalQuantileCalibrator] = []

        if "use_conformal" in kwargs or "nominal_coverage" in kwargs or "conformal_per_horizon" in kwargs:
            logger.warning(
                "use_conformal / nominal_coverage / conformal_per_horizon in TFTLongModelTrainer.__init__ are deprecated. "
                "Conformal calibration is a post-training step. Call trainer.use_conformal(nominal_coverage=..., conformal_per_horizon=...) "
                "after training or loading checkpoints, then run evaluate_only()."
            )

        super().__init__(
            model_name="TFTLong",
            base_dataset=base_dataset,
            n_epochs=n_epochs,
            seq_len=seq_len,
            horizons=horizons,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            factor=factor,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            data_set_name=data_set_name,
            verbose=verbose,
            print_metrics=print_metrics,
        )

    def _get_config_dict(self) -> dict:
        return {
            **super()._get_config_dict(),
            "model_class": "TFTLongModel",
            "optimizer": "Adam",
            "loss_fn": "QuantileLoss",
            "scheduler": "ReduceLROnPlateau",
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "num_quantiles": self.num_quantiles,
            "num_stacks": self.num_stacks,
            "use_is_imputed": self.use_is_imputed,
            "use_link_id": self.use_link_id,
            "use_link_length": self.use_link_length,
            "use_tod": self.use_tod,
            "use_dow": self.use_dow,
            "use_stop_sequence": self.use_stop_sequence,
            "use_pattern_profile": self.use_pattern_profile,
        }

    def train_model(self, df_train, df_eval, df_test, df_ha, scaler, absolut_matrix_train, i):
        forecast_horizon = max(self.horizons)
        
        if hasattr(self, "eval_target_start_idx") and self.eval_target_start_idx is not None:
            if self.eval_target_start_idx > self.seq_len:
                skip_steps = self.eval_target_start_idx - self.seq_len
                if skip_steps < len(df_eval):
                    df_eval = df_eval.iloc[skip_steps:]
                if skip_steps < len(df_test):
                    df_test = df_test.iloc[skip_steps:]

        train_ds = TFTLongDataset(
            df_train, encoder_len=self.seq_len, decoder_len=forecast_horizon,
            use_is_imputed=self.use_is_imputed, use_tod=self.use_tod, use_dow=self.use_dow, use_stop_sequence=self.use_stop_sequence, use_pattern_profile=self.use_pattern_profile, df_ha=df_ha, align_windows=True)
        val_ds = TFTLongDataset(
            df_eval, encoder_len=self.seq_len, decoder_len=forecast_horizon,
            use_is_imputed=self.use_is_imputed, use_tod=self.use_tod, use_dow=self.use_dow, use_stop_sequence=self.use_stop_sequence, use_pattern_profile=self.use_pattern_profile, df_ha=df_ha, align_windows=True)
        test_ds = TFTLongDataset(
            df_test, encoder_len=self.seq_len, decoder_len=forecast_horizon,
            use_is_imputed=self.use_is_imputed, use_tod=self.use_tod, use_dow=self.use_dow, use_stop_sequence=self.use_stop_sequence, use_pattern_profile=self.use_pattern_profile, df_ha=df_ha, align_windows=True)

        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            logger.warning("Skipping split %d due to empty datasets.", i)
            return

        self._current_n_links = train_ds.n_links

        train_loader = self._get_dataloader(train_ds, shuffle=True)
        val_loader = self._get_dataloader(val_ds, shuffle=False)
        test_loader = self._get_dataloader(test_ds, shuffle=False)

        model = self._get_model(n_input_features=None, n_output_features=train_ds.n_links, n_output_timesteps=forecast_horizon)
        self.models.append(model)

        import wandb
        run = None
        try:
            run = self._init_wandb(name=f"{i}_{self.group}")
            if run is not None:
                run.watch(model, log="all")

            criterion = self._get_loss_fn()
            optimizer = self._get_optimizer(model)
            scheduler = self._get_scheduler(optimizer, len(train_loader) * self.n_epochs)

            best_val_loss = float("inf")
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            scaler_amp = torch.GradScaler()

            for epoch in range(1, self.n_epochs + 1):
                model.train()
                train_loss = 0.0

                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    x = torch.nan_to_num(x, nan=0.0)

                    with torch.autocast("cuda"):
                        preds = model(x)
                        loss = criterion(preds, y)

                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()

                    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step()

                    train_loss += loss.item() * x.size(0)

                train_loss /= len(train_loader.dataset)
                if run is not None:
                    run.log({"epoch": epoch, "train_loss": train_loss})

                if epoch == 1:
                    self.run_metrics = self.metrics_manager.init_run_metrics(
                        prefix="eval", absolut_matrix_train=absolut_matrix_train,
                        x_list=[], y_list=[], scaler=scaler,
                    )
                else:
                    self.run_metrics.epoch = epoch
                self.run_metrics["epoch"] = epoch
                self.run_metrics["loss"] = train_loss

                if self.verbose:
                    logger.info("Epoch %02d | train %.4f", epoch, train_loss)

                val_loss = self._evaluate_model(model, val_loader, criterion, run, eval_type="eval")

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        logger.info("Early stopping at epoch %d. Best val_loss: %.4f", epoch, best_val_loss)
                        break

            model.load_state_dict(best_model_wts)
            self._evaluate_model(model, test_loader, criterion, run, eval_type="test")
        finally:
            if run is not None:
                run.finish()

    def _evaluate_model(self, model, dataloader, criterion, run, eval_type="eval"):
        model.eval()
        model.training = True # Trick underlying Model to return all Quantiles 
        
        self.run_metrics.update_prefix(eval_type)
        eval_loss = 0.0
        mse = nn.MSELoss()
        
        preds_list, targets_list = [], []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                x = torch.nan_to_num(x, nan=0.0)
                
                preds = model(x) # (B, T_dec, Q)
                B, T, Q = preds.shape
                
                preds_reshaped = preds.view(B, T, 1, Q)
                y_pred_p50 = preds_reshaped[:, :, :, 1].squeeze(-1) # (B, T)
                y_tgt = y.squeeze(-1) # (B, T)
                
                loss = mse(y_pred_p50, y_tgt)
                eval_loss += loss.item() * B
                
                preds_list.append(preds_reshaped.cpu())
                targets_list.append(y.unsqueeze(-1).cpu()) # To form (B, T, 1, 1)

        model.training = False
        eval_loss /= len(dataloader.dataset)
        
        # ── Reconstruct Wide Matrix for metrics ──
        ds = dataloader.dataset
        n_windows = ds.n_windows
        H_links = ds.n_links
        T_dec = max(self.horizons)
        Q_num = self.num_quantiles
        
        all_preds = torch.cat(preds_list)     # (N_valid, T, 1, Q)
        all_targets = torch.cat(targets_list) # (N_valid, T, 1, 1)
        
        y_pred_wide = torch.full((n_windows, T_dec, H_links, Q_num), float('nan'))
        y_true_wide = torch.full((n_windows, T_dec, H_links), float('nan'))
        
        # Map values back using valid_pairs
        w_indices = ds.valid_pairs[:, 0]
        h_indices = ds.valid_pairs[:, 1]
        
        y_pred_wide[w_indices, :, h_indices, :] = all_preds.squeeze(2)
        y_true_wide[w_indices, :, h_indices] = all_targets.squeeze(3).squeeze(2)
        
        # Drop rows that are completely NaN to match QuantConvLSTM's continuous output
        valid_rows_mask = ~torch.isnan(y_true_wide).view(n_windows, -1).all(dim=1)
        y_pred_wide = y_pred_wide[valid_rows_mask]
        y_true_wide = y_true_wide[valid_rows_mask]
        
        # Filter horizons
        horizons_idx = np.array(self.horizons) - 1
        y_pred_res = y_pred_wide[:, horizons_idx, :, :]
        y_true_res = y_true_wide[:, horizons_idx, :]
        
        y_pred_p50 = y_pred_res[:, :, :, 1]
        y_pred_p10 = y_pred_res[:, :, :, 0]
        y_pred_p90 = y_pred_res[:, :, :, 2]

        if eval_type == "test":
            self.test_y_true = y_true_res.numpy()
            self.test_y_pred = y_pred_p50.numpy()
            self.test_y_pred_quantiles = y_pred_res.numpy()

        self.run_metrics[f"{eval_type}/loss"] = eval_loss
        self.run_metrics.model_mae_rmse(y_true_res, y_pred_p50, reduction="overall", scale=self.dataset_object.normalize)
        if self.print_metrics in ["per_horizon", "both"]:
            self.run_metrics.model_mae_rmse(y_true_res, y_pred_p50, reduction="per_horizon", scale=self.dataset_object.normalize)
            
        self.run_metrics.model_icp_mil(y_true_res, y_pred_p10, y_pred_p90, reduction="overall", scale=self.dataset_object.normalize)
        if self.print_metrics in ["per_horizon", "both"]:
            self.run_metrics.model_icp_mil(y_true_res, y_pred_p10, y_pred_p90, reduction="per_horizon", scale=self.dataset_object.normalize)

        if run is not None:
            run.log({f"{eval_type}/loss": eval_loss, **{f"{eval_type}/{k}": v for k, v in self.run_metrics.items()}})

        if self.verbose:
            if self.print_metrics in ["overall", "both"]:
                logger.info(
                    "eval loss %.4f | MAE %.4f | RMSE %.4f | ICP %.4f | MIL %.4f",
                    eval_loss,
                    self.run_metrics[f"{eval_type}/Model_MAE_overall"],
                    self.run_metrics[f"{eval_type}/Model_RMSE_overall"],
                    self.run_metrics.get(f"{eval_type}/Model_ICP_overall", 0.0),
                    self.run_metrics.get(f"{eval_type}/Model_MIL_overall", 0.0)
                )
            if self.print_metrics in ["per_horizon", "both"]:
                for i, h in enumerate(self.horizons):
                    logger.info(
                        "  Horizon %d | MAE %.4f | RMSE %.4f | ICP %.4f | MIL %.4f",
                        h,
                        self.run_metrics.get(f"{eval_type}/Model_MAE_per_horizon_{i}", -1),
                        self.run_metrics.get(f"{eval_type}/Model_RMSE_per_horizon_{i}", -1),
                        self.run_metrics.get(f"{eval_type}/Model_ICP_per_horizon_{i}", -1),
                        self.run_metrics.get(f"{eval_type}/Model_MIL_per_horizon_{i}", -1)
                    )

        return eval_loss

    def _predict_raw_quantiles_wide(self, model: nn.Module, dataloader) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper to generate raw quantile predictions and ground truth wide matrix."""
        model.eval()
        model.training = True # Trick underlying Model to return all Quantiles 
        
        preds_list, targets_list = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                x = torch.nan_to_num(x, nan=0.0)
                
                preds = model(x) # (B, T_dec, Q)
                B, T, Q = preds.shape
                preds_reshaped = preds.view(B, T, 1, Q)
                
                preds_list.append(preds_reshaped.cpu())
                targets_list.append(y.unsqueeze(-1).cpu())

        model.training = False

        ds = dataloader.dataset
        n_windows = ds.n_windows
        H_links = ds.n_links
        T_dec = max(self.horizons)
        Q_num = self.num_quantiles

        all_preds = torch.cat(preds_list)     # (N_valid, T, 1, Q)
        all_targets = torch.cat(targets_list) # (N_valid, T, 1, 1)

        y_pred_wide = torch.full((n_windows, T_dec, H_links, Q_num), float('nan'))
        y_true_wide = torch.full((n_windows, T_dec, H_links), float('nan'))

        w_indices = ds.valid_pairs[:, 0]
        h_indices = ds.valid_pairs[:, 1]

        y_pred_wide[w_indices, :, h_indices, :] = all_preds.squeeze(2)
        y_true_wide[w_indices, :, h_indices] = all_targets.squeeze(3).squeeze(2)

        valid_rows_mask = ~torch.isnan(y_true_wide).view(n_windows, -1).all(dim=1)
        y_pred_wide = y_pred_wide[valid_rows_mask]
        y_true_wide = y_true_wide[valid_rows_mask]

        horizons_idx = np.array(self.horizons) - 1
        y_pred_res = y_pred_wide[:, horizons_idx, :, :]
        y_true_res = y_true_wide[:, horizons_idx, :]

        return y_true_res, y_pred_res

    def _apply_conformal_calibration(self, model: nn.Module, val_loader, run=None) -> None:
        """Apply Post-Hoc Split Conformal Quantile Calibration without altering point forecasts."""
        if not hasattr(self, "test_y_true") or not hasattr(self, "test_y_pred_quantiles"):
            logger.warning("Test predictions not found. Skipping conformal calibration.")
            return

        # 1. Run inference on validation / calibration set
        y_val_res, q_val_res = self._predict_raw_quantiles_wide(model, val_loader)

        # 2. Fit ConformalQuantileCalibrator on validation set
        calibrator = ConformalQuantileCalibrator(
            nominal_coverage=self.nominal_coverage,
            low_idx=0,
            med_idx=1,
            high_idx=2,
            per_horizon=self.conformal_per_horizon,
        )
        calibrator.fit(y_val=y_val_res.numpy(), q_val_raw=q_val_res.numpy())
        self.conformal_calibrator = calibrator
        self.calibrators.append(calibrator)

        # 3. Calibrate test set quantiles
        raw_test_quantiles = self.test_y_pred_quantiles
        calibrated_test_quantiles = calibrator.calibrate(raw_test_quantiles)
        self.test_y_pred_quantiles_calibrated = calibrated_test_quantiles

        # Explicit assertion: median / point forecast (index 1) must remain untouched
        assert np.allclose(
            self.test_y_pred_quantiles_calibrated[..., 1],
            self.test_y_pred,
            equal_nan=True
        ), "Conformal calibration altered median predictions! Point forecasts must be strictly invariant."

        # 4. Evaluate calibrated metrics
        calib_eval = calibrator.evaluate(self.test_y_true, calibrated_test_quantiles)

        # Unscale for metrics if normalize is True to match raw model_icp_mil units
        calib_p10_torch = torch.from_numpy(calibrated_test_quantiles[..., 0])
        calib_p90_torch = torch.from_numpy(calibrated_test_quantiles[..., 2])
        y_true_torch = torch.from_numpy(self.test_y_true)

        scale_flag = getattr(self.dataset_object, "normalize", False)
        if scale_flag and self.run_metrics.scaler is not None:
            calib_p10_eval = self.run_metrics._unscale(calib_p10_torch, self.run_metrics.scaler)
            calib_p90_eval = self.run_metrics._unscale(calib_p90_torch, self.run_metrics.scaler)
            y_true_eval = self.run_metrics._unscale(y_true_torch, self.run_metrics.scaler)
        else:
            calib_p10_eval = calib_p10_torch
            calib_p90_eval = calib_p90_torch
            y_true_eval = y_true_torch

        valid_mask = ~torch.isnan(y_true_eval)
        interval_len = torch.clamp(calib_p90_eval - calib_p10_eval, min=0)
        interval_len[~valid_mask] = float('nan')
        mil_overall = float(interval_len.nanmean().item())

        self.run_metrics["test/Model_Calibrated_ICP_overall"] = calib_eval["picp"]
        self.run_metrics["test/Model_Calibrated_MIL_overall"] = mil_overall

        if self.print_metrics in ["per_horizon", "both"]:
            mil_per_h = interval_len.nanmean(dim=(0, 2))
            for h_idx in range(len(self.horizons)):
                res_h = calib_eval["per_horizon"].get(h_idx, {})
                self.run_metrics[f"test/Model_Calibrated_ICP_per_horizon_{h_idx}"] = res_h.get("picp", 0.0)
                self.run_metrics[f"test/Model_Calibrated_MIL_per_horizon_{h_idx}"] = float(mil_per_h[h_idx].item())

        if run is not None:
            run.log({
                "test/Model_Calibrated_ICP_overall": calib_eval["picp"],
                "test/Model_Calibrated_MIL_overall": mil_overall,
            })

        if self.verbose:
            q_hat_str = np.array2string(calibrator.q_hat, precision=4) if isinstance(calibrator.q_hat, np.ndarray) else f"{calibrator.q_hat:.4f}"
            logger.info(
                "[Conformal Calibration (Target %.1f%%)] Calib ICP %.4f (raw %.4f) | Calib MIL %.4f (raw %.4f) | q_hat %s",
                self.nominal_coverage * 100,
                calib_eval["picp"],
                self.run_metrics.get("test/Model_ICP_overall", 0.0),
                mil_overall,
                self.run_metrics.get("test/Model_MIL_overall", 0.0),
                q_hat_str,
            )
            if self.print_metrics in ["per_horizon", "both"]:
                for h_idx, h in enumerate(self.horizons):
                    res_h = calib_eval["per_horizon"].get(h_idx, {})
                    logger.info(
                        "  [Conformal] Horizon %d | Calib ICP %.4f | Calib MIL %.4f",
                        h,
                        res_h.get("picp", 0.0),
                        float(mil_per_h[h_idx].item()),
                    )

    def _get_model(self, n_input_features, n_output_features, n_output_timesteps) -> nn.Module:
        link_meta = getattr(self.dataset_object, "link_metadata", None)
        model = TFTLongModel(
            encoder_len=self.seq_len,
            decoder_len=n_output_timesteps,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            num_quantiles=self.num_quantiles,
            num_stacks=self.num_stacks,
            link_metadata=link_meta,
            use_is_imputed=self.use_is_imputed,
            use_link_id=self.use_link_id,
            use_link_length=self.use_link_length,
            use_tod=self.use_tod,
            use_dow=self.use_dow,
            use_stop_sequence=self.use_stop_sequence,
            use_pattern_profile=self.use_pattern_profile,
        )
        return model.to(self.device)

    def _get_loss_fn(self) -> nn.Module:
        return QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    def _get_optimizer(self, model) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _get_scheduler(self, optimizer, total_steps):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.factor, patience=self.patience)

    def save_checkpoints(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for i, model in enumerate(self.models):
            path = os.path.join(directory, f"tft_long_model_split_{i}.pt")
            if hasattr(model, 'save_checkpoint'):
                model.save_checkpoint(path)
            else:
                torch.save(model.state_dict(), path)
        logger.info("Saved %d checkpoints to %s", len(self.models), directory)

    def use_conformal(
        self,
        nominal_coverage: Union[float, bool] = 0.80,
        conformal_per_horizon: bool = True,
    ) -> "TFTLongModelTrainer":
        """Configure and enable Conformal Quantile Calibration for post-hoc evaluation.

        Call this method after model training or after loading checkpoints before
        running evaluate_only().

        Args:
            nominal_coverage: Target empirical coverage (e.g. 0.80 for 80% coverage),
                              or False to disable conformal calibration. Defaults to 0.80.
            conformal_per_horizon: Whether to compute separate conformal thresholds per forecast horizon.
                                   Defaults to True.

        Returns:
            self for method chaining.
        """
        if isinstance(nominal_coverage, bool):
            self.is_conformal_enabled = nominal_coverage
            if not nominal_coverage:
                logger.info("Conformal calibration disabled.")
                return self
            nominal_coverage = 0.80

        if not 0.0 < float(nominal_coverage) < 1.0:
            raise ValueError(f"nominal_coverage must be between 0 and 1, got {nominal_coverage}")

        self.is_conformal_enabled = True
        self.nominal_coverage = float(nominal_coverage)
        self.conformal_per_horizon = bool(conformal_per_horizon)
        logger.info(
            f"Conformal calibration enabled: nominal_coverage={self.nominal_coverage * 100:.1f}%, "
            f"conformal_per_horizon={self.conformal_per_horizon}"
        )
        return self

    def load_checkpoints(self, directory: str):
        self.models = []
        for i in range(len(self.splits)):
            path = os.path.join(directory, f"tft_long_model_split_{i}.pt")
            if os.path.exists(path):
                model = TFTLongModel.load_checkpoint(path, device=self.device).to(self.device)
                self.models.append(model)
        logger.info("Loaded %d checkpoints from %s", len(self.models), directory)

    def evaluate_only(self):
        """Run evaluation on the test set for all loaded checkpoints without re-training."""
        assert len(self.models) == len(self.splits), "Number of loaded models must match the number of splits."
        
        from bus_benchmark.experiments.metrics_manager import MetricsManager
        self.metrics_manager = MetricsManager()
        self.calibrators = []
        
        forecast_horizon = max(self.horizons)
        for idx, (df_train, df_val, df_test, df_ha, scaler, absolut_matrix_train) in enumerate(self.splits):
            logger.info("Evaluating split %d/%d from checkpoint", idx, len(self.splits))
            test_ds = TFTLongDataset(
                df_test, encoder_len=self.seq_len, decoder_len=forecast_horizon,
                use_is_imputed=self.use_is_imputed, use_tod=self.use_tod, use_dow=self.use_dow, use_stop_sequence=self.use_stop_sequence, use_pattern_profile=self.use_pattern_profile, df_ha=df_ha, align_windows=True)
            
            test_loader = self._get_dataloader(test_ds, shuffle=False)
            model = self.models[idx]
            
            criterion = self._get_loss_fn()
            
            # Setup run metrics context
            self.run_metrics = self.metrics_manager.init_run_metrics(
                prefix="eval",
                absolut_matrix_train=absolut_matrix_train,
                x_list=[], y_list=[], # Dummy for evaluation
                scaler=scaler,
            )
            self._evaluate_model(
                model=model, dataloader=test_loader,
                criterion=criterion, run=None, eval_type="test")

            if self.is_conformal_enabled:
                val_ds = TFTLongDataset(
                    df_val, encoder_len=self.seq_len, decoder_len=forecast_horizon,
                    use_is_imputed=self.use_is_imputed,
                    use_tod=self.use_tod, use_dow=self.use_dow, use_stop_sequence=self.use_stop_sequence,
                    use_pattern_profile=self.use_pattern_profile, df_ha=df_ha, align_windows=True)
                val_loader = self._get_dataloader(val_ds, shuffle=False)
                self._apply_conformal_calibration(model, val_loader, run=None)
                    
        logger.info("=========================================================================")
        logger.info("Overall Geometric Means across all splits:")
        
        gm = self.metrics_manager.test_geometric_means

        target_metrics = [
            "test_GM/Model_MAE_overall",
            "test_GM/Model_RMSE_overall",
            "test_GM/Model_ICP_overall",
            "test_GM/Model_MIL_overall",
            "test_GM/loss"
        ]
        if self.is_conformal_enabled:
            target_metrics.extend([
                "test_GM/Model_Calibrated_ICP_overall",
                "test_GM/Model_Calibrated_MIL_overall",
            ])
        for k in target_metrics:
            if k in gm:
                logger.info("  %s: %.4f", k, float(gm[k]))

        if self.print_metrics in ["per_horizon", "both"]:
            logger.info("Per-Horizon Geometric Means across all splits:")
            for h_idx, h in enumerate(self.horizons):
                per_h_keys = [
                    f"test_GM/Model_MAE_per_horizon_{h_idx}",
                    f"test_GM/Model_RMSE_per_horizon_{h_idx}",
                    f"test_GM/Model_ICP_per_horizon_{h_idx}",
                    f"test_GM/Model_MIL_per_horizon_{h_idx}",
                ]
                if self.is_conformal_enabled:
                    per_h_keys.extend([
                        f"test_GM/Model_Calibrated_ICP_per_horizon_{h_idx}",
                        f"test_GM/Model_Calibrated_MIL_per_horizon_{h_idx}",
                    ])
                vals = {k: gm[k] for k in per_h_keys if k in gm}
                if vals:
                    parts = " | ".join(
                        f"{k.split('/')[-1].replace(f'_per_horizon_{h_idx}', '')}: {float(v):.4f}"
                        for k, v in vals.items()
                    )
                    logger.info("  Horizon %d | %s", h, parts)

        logger.info("=========================================================================")

    def get_interpretation(self, split_idx: int = -1):
        """Extract interpretation weights on test set for a given split."""
        from bus_benchmark.experiments.model.tft_interpret import interpret_tft_attention_components
        
        if not self.models:
            logger.warning("No models loaded. Cannot get interpretation.")
            return {}
            
        if split_idx == -1:
            split_idx = len(self.splits) - 1
            
        df_train, df_val, df_test, df_ha, scaler, absolut_matrix_train = self.splits[split_idx]
        model = self.models[split_idx]
        
        forecast_horizon = max(self.horizons)
        test_ds = TFTLongDataset(
            df_test, encoder_len=self.seq_len, decoder_len=forecast_horizon,
            use_is_imputed=self.use_is_imputed, use_tod=self.use_tod, use_dow=self.use_dow, use_stop_sequence=self.use_stop_sequence, use_pattern_profile=self.use_pattern_profile, df_ha=df_ha, align_windows=True)
            
        test_loader = self._get_dataloader(test_ds, shuffle=False)
        
        model.eval()
        attention_components_list = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device)
                x = torch.nan_to_num(x, nan=0.0)
                
                B = x.size(0)
                enc = self.seq_len
                link_idx_batch = x[:, 0, -1].long()
                
                obs_inputs = x[:, :enc, :model.num_obs]
                known_inputs = x[:, :, model.num_obs:-1]
                static = model._build_static(link_idx_batch)
                
                _, attention_components = model.tft(obs_inputs, known_inputs, static)
                attention_components_list.append(attention_components)
                
        interpretation = interpret_tft_attention_components(
            attention_components_list, enc_steps=self.seq_len
        )
        return interpretation

    def plot_interpretation(self, split_idx: int = -1):
        """Plot the model's variable importance and temporal attention."""
        from bus_benchmark.experiments.model.tft_interpret import plot_tft_interpretation
        
        interpretation = self.get_interpretation(split_idx)
        if not interpretation:
            return {}
            
        real_idx = split_idx if split_idx != -1 else len(self.splits) - 1
        df_test = self.splits[real_idx][2]
        link_cols = [c for c in df_test.columns if c not in ("sin_tod", "cos_tod")]
        test_ds_n_links = len(link_cols)

        temporal_labels = []
        # Long format has: observed, optional is_imputed, (and lag_1w if enabled)
        obs_labels = ["travel_time"]
        if self.use_is_imputed:
            obs_labels.append("is_imputed")
            
        if self.use_tod:
            temporal_labels.extend(["sin_tod", "cos_tod"])
        if self.use_dow:
            temporal_labels.extend(["sin_dow", "cos_dow"])
            
        if self.use_pattern_profile:
            temporal_labels.append("pattern_profile")

        static_labels = []
        if self.use_link_id:
            static_labels.append("link_id")
        if self.use_link_length:
            static_labels.append("link_length")
        if self.use_stop_sequence:
            static_labels.append("stop_sequence")
        
        historical_labels = temporal_labels + obs_labels
        future_labels = temporal_labels
        
        # Using self.seq_len for interpret plot
        figs = plot_tft_interpretation(
            interpretation, self.seq_len, static_labels, historical_labels, future_labels
        )
        return figs
