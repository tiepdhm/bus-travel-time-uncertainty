import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import numpy as np

from bus_benchmark.experiments.fixed_interval_dataset import FixedIntervalDataset
from bus_benchmark.experiments.model.model_trainer_base import ModelTrainerBase
from bus_benchmark.experiments.model.quant_convlstm_model import QuantConvLSTMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JointQuantileLoss(nn.Module):
    """
    Combined loss for mean and quantiles.
    Translated from dqr-joint-200S.ipynb: joint_tilted_loss
    """
    def __init__(self, quantiles: list[float] = None):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions: (B, S_out, H, Q + 1)
        # targets: (B, S_out, H)
        preds_mean = predictions[..., 0]
        preds_quantiles = predictions[..., 1:]

        # MSE loss for mean
        loss = torch.mean(torch.square(targets - preds_mean))

        # Quantile block
        errors = targets.unsqueeze(-1) - preds_quantiles
        for i, q in enumerate(self.quantiles):
            e = errors[..., i]
            loss = loss + torch.mean(torch.max(q * e, (q - 1.0) * e))

        return loss

class QuantConvLSTMModelTrainer(ModelTrainerBase):
    def __init__(
        self,
        base_dataset: FixedIntervalDataset,
        n_epochs: int = 30,
        seq_len: int = 8,
        horizons: tuple[int, ...] = (1, 2, 4, 8),
        batch_size: int = 512,
        lr: float = 1e-3,
        patience: int = 10,
        factor: float = 0.5,
        weight_decay: float = 1e-5,
        num_quantiles: int = 3,
        data_set_name: Optional[str] = None,
        verbose: bool = True,
        print_metrics: Literal["overall", "per_horizon", "both"] = "both",
    ) -> None:
        self.num_quantiles = num_quantiles
        super().__init__(
            model_name="QuantConvLSTM",
            base_dataset=base_dataset,
            n_epochs=n_epochs,
            seq_len=seq_len,
            horizons=horizons,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            factor=factor,
            weight_decay=weight_decay,
            data_set_name=data_set_name,
            verbose=verbose,
            print_metrics=print_metrics,
        )

    def _get_config_dict(self):
        return {
            **super()._get_config_dict(),
            "model_class": "QuantConvLSTM",
            "optimizer": "RMSprop",
            "loss_fn": "JointQuantileLoss",
            "scheduler": "ReduceLROnPlateau",
            "num_quantiles": self.num_quantiles,
        }

    def _get_model(
        self, n_input_features: int, n_output_features: int, n_output_timesteps: int
    ) -> nn.Module:
        model = QuantConvLSTMModel(
            output_timesteps=n_output_timesteps,
            num_quantiles=self.num_quantiles,
        )
        return model.to(self.device)

    def _run_model(
        self, model: nn.Module, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return model(x)

    def _get_loss_fn(self):
        return JointQuantileLoss(quantiles=[0.1, 0.5, 0.9])

    def _get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _get_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        
    def _evaluate_model(self, model, dataloader, criterion, run,
                        eval_type="eval"):
        import numpy as np
        assert hasattr(self, "metrics_manager")
        self.run_metrics.update_prefix(eval_type)

        model.eval()
        eval_loss = 0.0
        eval_preds, eval_targets = [], []

        mse = nn.MSELoss()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                mask = ~torch.isnan(x)
                x = torch.nan_to_num(x, nan=0.0)
                y_pred_full = self._run_model(model, x, mask)
                
                # index 0 is mean, 1 is p10, 2 is p50, 3 is p90
                y_pred_p50 = y_pred_full[:, :, :, 2]
                eval_loss += mse(y_pred_p50, y).item() * x.size(0)
                eval_preds.append(y_pred_full)
                eval_targets.append(y)
                
        eval_loss /= len(dataloader.dataset)

        y_pred_res = torch.cat(eval_preds)
        y_true_res = torch.cat(eval_targets)

        horizons = np.array(self.horizons) - 1
        y_pred_res = y_pred_res[:, horizons, :, :]
        y_true_res = y_true_res[:, horizons, :]
        
        y_pred_mean = y_pred_res[:, :, :, 0]
        y_pred_p10 = y_pred_res[:, :, :, 1]
        y_pred_p50 = y_pred_res[:, :, :, 2]
        y_pred_p90 = y_pred_res[:, :, :, 3]

        if eval_type == "test":
            self.test_y_true = y_true_res.cpu().numpy()
            self.test_y_pred = y_pred_p50.cpu().numpy()
            self.test_y_pred_mean = y_pred_mean.cpu().numpy()
            self.test_y_pred_quantiles = y_pred_res[:, :, :, 1:].contiguous().cpu().numpy()

        self.run_metrics[f"{eval_type}/loss"] = eval_loss
        self.run_metrics.model_mae_rmse(
            y_true_res, y_pred_p50, reduction="overall",
            scale=self.dataset_object.normalize)
        if hasattr(self, "print_metrics") and self.print_metrics in ["per_horizon", "both"]:
            self.run_metrics.model_mae_rmse(
                y_true_res, y_pred_p50, reduction="per_horizon",
                scale=self.dataset_object.normalize)
        # ICP and MIL metrics calculation
        self.run_metrics.model_icp_mil(
            y_true_res, y_pred_p10, y_pred_p90, reduction="overall",
            scale=self.dataset_object.normalize)
        if hasattr(self, "print_metrics") and self.print_metrics in ["per_horizon", "both"]:
            self.run_metrics.model_icp_mil(
                y_true_res, y_pred_p10, y_pred_p90, reduction="per_horizon",
                scale=self.dataset_object.normalize)
        self.run_metrics.ha_mae_rmse(
            y_true_res, reduction="overall",
            scale=self.dataset_object.normalize)

        if run is not None:
            run.log({
                f"{eval_type}/loss": eval_loss,
                **{f"{eval_type}/{k}": v
                   for k, v in self.run_metrics.items()},
            })

        if self.verbose:
            print_metrics = getattr(self, "print_metrics", "both")
            if print_metrics in ["overall", "both"]:
                logger.info(
                    "eval loss %.4f | MAE %.4f | RMSE %.4f | "
                    "ICP %.4f | MIL %.4f",
                    eval_loss,
                    self.run_metrics.get(f"{eval_type}/Model_MAE_overall", 0),
                    self.run_metrics.get(f"{eval_type}/Model_RMSE_overall", 0),
                    self.run_metrics.get(f"{eval_type}/Model_ICP_overall", 0),
                    self.run_metrics.get(f"{eval_type}/Model_MIL_overall", 0))
            if print_metrics in ["per_horizon", "both"]:
                for i, h in enumerate(self.horizons):
                    logger.info(
                        "  Horizon %d | MAE %.4f | RMSE %.4f | ICP %.4f | MIL %.4f",
                        h,
                        self.run_metrics.get(f"{eval_type}/Model_MAE_per_horizon_{i}", 0.0),
                        self.run_metrics.get(f"{eval_type}/Model_RMSE_per_horizon_{i}", 0.0),
                        self.run_metrics.get(f"{eval_type}/Model_ICP_per_horizon_{i}", 0.0),
                        self.run_metrics.get(f"{eval_type}/Model_MIL_per_horizon_{i}", 0.0)
                    )

        return eval_loss

    def evaluate_only(self):
        """Run evaluation on the test set for all loaded checkpoints without re-training."""
        assert len(self.models) == len(self.splits), "Number of loaded models must match the number of splits."
        
        for idx, (df_train, df_val, df_test, df_ha, scaler, absolut_matrix_train) in enumerate(self.splits):
            logger.info("Evaluating split %d/%d from checkpoint", idx, len(self.splits))
            
            from bus_benchmark.experiments.multi_step_dataset import MultiStepDataset
            test_ds = MultiStepDataset(df_test, seq_len=self.seq_len, horizons=self.horizons)
            
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
                    
        logger.info("=========================================================================")
        logger.info("Overall Geometric Means across all splits:")
        
        target_metrics = [
            "test_GM/Model_MAE_overall",
            "test_GM/Model_RMSE_overall",
            "test_GM/Model_ICP_overall",
            "test_GM/Model_MIL_overall",
            "test_GM/loss"
        ]
        for k in target_metrics:
            if k in self.metrics_manager.test_geometric_means:
                v = self.metrics_manager.test_geometric_means[k]
                logger.info("  %s: %.4f", k, float(v))

        _print_metrics = getattr(self, "print_metrics", "both")
        if _print_metrics in ["per_horizon", "both"]:
            logger.info("Per-Horizon Geometric Means across all splits:")
            for i, h in enumerate(self.horizons):
                ph_metrics = [
                    f"test_GM/Model_MAE_per_horizon_{i}",
                    f"test_GM/Model_RMSE_per_horizon_{i}",
                    f"test_GM/Model_ICP_per_horizon_{i}",
                    f"test_GM/Model_MIL_per_horizon_{i}",
                ]
                gm = self.metrics_manager.test_geometric_means
                vals = {k: float(gm[k]) for k in ph_metrics if k in gm}
                if vals:
                    logger.info(
                        "  Horizon %d | MAE %.4f | RMSE %.4f | ICP %.4f | MIL %.4f",
                        h,
                        vals.get(f"test_GM/Model_MAE_per_horizon_{i}", 0.0),
                        vals.get(f"test_GM/Model_RMSE_per_horizon_{i}", 0.0),
                        vals.get(f"test_GM/Model_ICP_per_horizon_{i}", 0.0),
                        vals.get(f"test_GM/Model_MIL_per_horizon_{i}", 0.0),
                    )

        logger.info("=========================================================================")

    def evaluate_only_with_mean(self):
        """Run evaluation on the test set for all loaded checkpoints using mean predictions without re-training."""
        assert len(self.models) == len(self.splits), "Number of loaded models must match the number of splits."

        for idx, (df_train, df_val, df_test, df_ha, scaler, absolut_matrix_train) in enumerate(self.splits):
            logger.info("Evaluating split %d/%d with mean from checkpoint", idx, len(self.splits))
            
            # Khởi tạo dataloader cho tập test
            from bus_benchmark.experiments.multi_step_dataset import MultiStepDataset
            test_ds = MultiStepDataset(df_test, seq_len=self.seq_len, horizons=self.horizons)
            test_loader = self._get_dataloader(test_ds, shuffle=False)
            model = self.models[idx]
            
            # Setup run metrics context for evaluation
            self.run_metrics = self.metrics_manager.init_run_metrics(
                prefix="eval",
                absolut_matrix_train=absolut_matrix_train,
                x_list=[], y_list=[], # Dummy for evaluation
                scaler=scaler,
            )
            
            self.run_metrics.update_prefix("test")
            model.eval()
            eval_preds_mean, eval_preds_p10, eval_preds_p50, eval_preds_p90, eval_targets = [], [], [], [], []
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    mask = ~torch.isnan(x)
                    x = torch.nan_to_num(x, nan=0.0)
                    y_pred_full = self._run_model(model, x, mask)
                    
                    eval_preds_mean.append(y_pred_full[:, :, :, 0])
                    eval_preds_p10.append(y_pred_full[:, :, :, 1])
                    eval_preds_p50.append(y_pred_full[:, :, :, 2])
                    eval_preds_p90.append(y_pred_full[:, :, :, 3])
                    eval_targets.append(y)
                    
            y_pred_mean = torch.cat(eval_preds_mean)
            y_pred_p10 = torch.cat(eval_preds_p10)
            y_pred_p50 = torch.cat(eval_preds_p50)
            y_pred_p90 = torch.cat(eval_preds_p90)
            y_true_res = torch.cat(eval_targets)
            
            horizons = np.array(self.horizons) - 1
            y_pred_mean = y_pred_mean[:, horizons, :]
            y_pred_p10 = y_pred_p10[:, horizons, :]
            y_pred_p50 = y_pred_p50[:, horizons, :]
            y_pred_p90 = y_pred_p90[:, horizons, :]
            y_true_res = y_true_res[:, horizons, :]
            
            self.test_y_true = y_true_res.cpu().numpy()
            self.test_y_pred_mean = y_pred_mean.cpu().numpy()
            self.test_y_pred = y_pred_p50.cpu().numpy()
            self.test_y_pred_quantiles = torch.stack([y_pred_p10, y_pred_p50, y_pred_p90], dim=-1).cpu().numpy()
            
            # Dummy test/loss to prevent AssertionError in test_metrics property
            # Set to 1.0 instead of 0.0 to avoid ValueError: math domain error in math.log() for geometric mean calculation
            self.run_metrics["test/loss"] = 1.0
            
            # Calculate metrics
            self.run_metrics.model_mae_rmse(
                y_true_res, y_pred_mean, reduction="overall",
                scale=self.dataset_object.normalize)
            self.run_metrics.model_icp_mil(
                y_true_res, y_pred_p10, y_pred_p90, reduction="overall",
                scale=self.dataset_object.normalize)
                
        logger.info("=========================================================================")
        logger.info("Overall Geometric Means using MEAN across all splits:")
        
        target_metrics = [
            "test_GM/Model_MAE_overall",
            "test_GM/Model_RMSE_overall",
            "test_GM/Model_ICP_overall",
            "test_GM/Model_MIL_overall"
        ]
        for k in target_metrics:
            if k in self.metrics_manager.test_geometric_means:
                v = self.metrics_manager.test_geometric_means[k]
                logger.info("  %s: %.4f", k, float(v))
                
        logger.info("=========================================================================")
