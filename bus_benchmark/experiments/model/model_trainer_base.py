import copy
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Union, Literal

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb

from bus_benchmark.experiments.fixed_interval_dataset import FixedIntervalDataset
from bus_benchmark.experiments.metrics_manager import MetricsManager
from bus_benchmark.experiments.multi_step_dataset import MultiStepDataset
from bus_benchmark import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainerBase:
    def __init__(
        self,
        model_name: str,
        base_dataset: FixedIntervalDataset,
        n_epochs: int = 30,
        seq_len: int = 96,
        horizons: tuple[int, ...] = (1, 2, 4, 8),
        batch_size: int = 512,
        lr: float = 1e-3,
        patience: int = 10,
        factor: float = 0.5,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 5.0,
        data_set_name: Optional[str] = None,
        verbose: bool = True,
        print_metrics: Literal["overall", "per_horizon", "both"] = "both",
    ) -> None:
        self.model_name = model_name

        self.dataset_object = base_dataset
        self.splits = base_dataset.data_set

        self.n_epochs = n_epochs
        self.seq_len = seq_len
        self.horizons = horizons

        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        self.print_metrics = print_metrics

        self.metrics_manager = MetricsManager()

        if data_set_name is not None:
            self.group = f"{self.model_name}_{data_set_name}_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}"
        else:
            self.group = f"{self.model_name}_MultiStepDataset_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}"
        self.config_dict = self._get_config_dict()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models: list[nn.Module] = []
        self.total_split_metrics: list[dict[str, Union[int, float, Any]]] = []
        self.geometric_means: dict[str, torch.Tensor] = {}

        logger.info("Initialized trainer on device: %s", self.device)

    def run_experiment(self):
        """
        Run the experiment on all splits.
        """
        for idx, (
            df_train,
            df_val,
            df_test,
            df_ha,
            scaler,
            absolut_matrix_train,
        ) in enumerate(self.splits):
            logger.info("Running split %d/%d", idx, len(self.splits))
            self.train_model(
                df_train, df_val, df_test, df_ha, scaler, absolut_matrix_train, idx
            )

        self._log_total_experiment_metrics()

        return None

    def train_model(
        self,
        df_train: pd.DataFrame,
        df_eval: pd.DataFrame,
        df_test: pd.DataFrame,
        df_ha: pd.DataFrame,
        scaler: Union[StandardScaler, None],
        absolut_matrix_train: pd.DataFrame,
        i: int,
    ) -> None:
        train_ds = MultiStepDataset(df_train, seq_len=self.seq_len, horizons=self.horizons)
        val_ds = MultiStepDataset(df_eval, seq_len=self.seq_len, horizons=self.horizons)
        test_ds = MultiStepDataset(df_test, seq_len=self.seq_len, horizons=self.horizons)

        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            logger.warning(
                f"Skipping split {i} due to empty datasets: "
                f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
            )
            return

        train_loader = self._get_dataloader(train_ds, shuffle=False)
        val_loader = self._get_dataloader(val_ds, shuffle=False)
        test_loader = self._get_dataloader(test_ds, shuffle=False)

        model = self._get_model(
            n_input_features=train_ds.n_features,
            n_output_features=train_ds.n_links,
            n_output_timesteps=max(self.horizons),
        )
        self.models.append(model)

        run = None
        try:
            run = self._init_wandb(name=f"{i}_{self.group}")
            run.watch(model, log="all")

            criterion = self._get_loss_fn()
            optimizer = self._get_optimizer(model)
            scheduler = self._get_scheduler(
                optimizer, len(train_loader) * self.n_epochs
            )

            # early stopping setup
            best_val_loss = float("inf")
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            scaler_amp = torch.GradScaler()

            # training loop
            for epoch in range(1, self.n_epochs + 1):
                # naive forecast setup
                x_list: list[torch.Tensor] = []
                y_list: list[torch.Tensor] = []
                # train
                model.train()

                train_loss = 0.0
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()

                    mask = ~torch.isnan(x)
                    x = torch.nan_to_num(x, nan=0.0)

                    with torch.autocast("cuda"):
                        preds = self._run_model(model, x, mask)
                        loss = criterion(preds, y)

                    scaler_amp.scale(loss).backward()

                    # required by MOMENT?
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                    scaler_amp.step(optimizer)
                    scaler_amp.update()

                    if not isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step()

                    train_loss += loss.item() * x.size(0)

                    x_list.append(x)
                    y_list.append(y)

                train_loss /= len(train_loader.dataset)  # type: ignore

                # log training & validation
                run.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                    }
                )

                # init the run metric object
                # it is used to record metrics about the current run
                if epoch == 1:
                    self.run_metrics = self.metrics_manager.init_run_metrics(
                        prefix="eval",
                        absolut_matrix_train=absolut_matrix_train,
                        x_list=x_list,
                        y_list=y_list,
                        scaler=scaler,
                    )
                else:
                    self.run_metrics.epoch = epoch
                self.run_metrics["epoch"] = epoch
                self.run_metrics["loss"] = train_loss

                if self.verbose:
                    logger.info(f"Epoch {epoch:02d} | train {train_loss:.4f}")

                # validation
                val_loss = self._evaluate_model(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    run=run,
                    eval_type="eval",
                )

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)

                # early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        logger.info(
                            f"Early stopping at epoch {epoch}. Best val_loss: {best_val_loss:.4f}"
                        )
                        break

            # load best model weights
            model.load_state_dict(best_model_wts)

            # final evaluation on test set
            self._evaluate_model(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                run=run,
                eval_type="test",
            )

        # finish W&B run for this split
        finally:
            if run is not None:
                run.finish()

        return None

    def _log_total_experiment_metrics(self) -> None:
        try:
            run = self._init_wandb(
                name=self.group,
                additional_config={
                    "Summary_run": True,
                },
            )

            run.log(self.metrics_manager.test_geometric_means)

            final_metrics_list: Dict[str, List[Union[torch.Tensor, float]]] = (
                self.metrics_manager.final_metrics_list
            )
            for k, v in final_metrics_list.items():
                for value in v:
                    run.log({k: value})
        finally:
            if run is not None:
                run.finish()
        return None

    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        run: wandb.sdk.wandb_run,  # type: ignore
        eval_type: Literal["eval", "test"] = "eval",
    ) -> float:
        assert hasattr(self, "metrics_manager"), (
            "metrics_manager is not set. Call train_model first or set it your self."
        )
        self.run_metrics.update_prefix(eval_type)

        model.eval()
        eval_loss = 0.0
        eval_preds = []
        eval_targets = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                mask = ~torch.isnan(x)
                x = torch.nan_to_num(x, nan=0.0)
                y_pred = self._run_model(model, x, mask)
                eval_loss += criterion(y_pred, y).item() * x.size(0)
                eval_preds.append(y_pred)
                eval_targets.append(y)

        # epoch‐level metrics
        eval_loss /= len(dataloader.dataset)  # type: ignore

        y_pred_res = torch.cat(eval_preds)
        y_true_res = torch.cat(eval_targets)

        # only calculate the metrics for the horizons we are interested in
        horizons = np.array(self.horizons) - 1
        y_pred_res = y_pred_res[:, horizons, :]
        y_true_res = y_true_res[:, horizons, :]

        if eval_type == "test":
            self.test_y_true = y_true_res.cpu().numpy()
            self.test_y_pred = y_pred_res.cpu().numpy()

        # calulate metrics
        self.run_metrics[f"{eval_type}/loss"] = eval_loss
        self.run_metrics.model_mae_rmse(
            y_true_res,
            y_pred_res,
            reduction="overall",
            scale=self.dataset_object.normalize,
        )
        if self.print_metrics in ["per_horizon", "both"]:
            # Added code: Compute metrics per horizon
            self.run_metrics.model_mae_rmse(
                y_true_res,
                y_pred_res,
                reduction="per_horizon",
                scale=self.dataset_object.normalize,
            )
        self.run_metrics.ha_mae_rmse(
            y_true_res, reduction="overall", scale=self.dataset_object.normalize
        )

        run.log(
            {
                f"{eval_type}/loss": eval_loss,
                **{f"{eval_type}/{k}": v for k, v in self.run_metrics.items()},
            }
        )

        if self.verbose:
            if self.print_metrics in ["overall", "both"]:
                logger.info(
                    f"eval loss {eval_loss:.4f} | "
                    f"MAE {self.run_metrics[f'{eval_type}/Model_MAE_overall']:.4f} | RMSE {self.run_metrics[f'{eval_type}/Model_RMSE_overall']:.4f}"
                )
            if self.print_metrics in ["per_horizon", "both"]:
                # Added code: Print metrics for each specific forecast horizon separately
                for i, h in enumerate(self.horizons):
                    logger.info(
                        f"  Horizon {h} | "
                        f"MAE {self.run_metrics[f'{eval_type}/Model_MAE_per_horizon_{i}']:.4f} | "
                        f"RMSE {self.run_metrics[f'{eval_type}/Model_RMSE_per_horizon_{i}']:.4f}"
                    )

        return eval_loss

    def _get_config_dict(self):
        return {
            "n_epochs": self.n_epochs,
            "seq_len": self.seq_len,
            "horizons": self.horizons,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "patience": self.patience,
            "factor": self.factor,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "DS_n_splits": self.dataset_object.n_splits,
            "DS_freq": self.dataset_object.freq,
            "DS_ha_agg_func": self.dataset_object.ha_agg_func,
            "DS_residuals_ffill_limit": self.dataset_object.ffill_limit,
            "DS_mad_thresh": self.dataset_object.mad_thresh,
            "DS_interpolate_ha": self.dataset_object.interpolate_ha,
            "DS_calculate_residuals": self.dataset_object.calculate_residuals,
            "DS_normalize": self.dataset_object.normalize,
        }

    def _init_wandb(
        self, name: str, additional_config: dict[str, Any] = {}
    ) -> wandb.sdk.wandb_run:  # type: ignore
        config_dict = self.config_dict
        config_dict.update(additional_config)

        run = wandb.init(
            entity=config.WANDB_ENTITY,
            project=config.WANDB_PROJECT,
            group=self.group,
            name=name,
            config=config_dict,
            mode="disabled"
        )
        return run

    def _get_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

    def _get_model(
        self, n_input_features: int, n_output_features: int, n_output_timesteps: int
    ) -> nn.Module:
        raise NotImplementedError()

    def _run_model(
        self, model: nn.Module, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _get_loss_fn(self) -> nn.Module:
        raise NotImplementedError()

    def _get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        raise NotImplementedError()

    def _get_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        raise NotImplementedError()

    @staticmethod
    def _setup_seed(seed: int = 42) -> None:
        """
        Set random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
