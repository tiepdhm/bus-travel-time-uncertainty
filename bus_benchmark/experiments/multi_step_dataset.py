import logging

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.lib.stride_tricks import sliding_window_view


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiStepDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 8,
        horizons: tuple[int, ...] = (1, 2, 4, 8),
    ):
        self.X, self.y = self._to_np_arrays(df, seq_len, horizons)
        self.n_features = df.shape[1]
        # drop sin/cos to count pure route columns once
        self.n_links = df.drop(columns=["sin_tod", "cos_tod"], errors="ignore").shape[1]

    @staticmethod
    def _to_np_arrays(
        df: pd.DataFrame, seq_len: int, horizons: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        vals_x = df.values
        vals_y = df.drop(columns=["sin_tod", "cos_tod"], errors="ignore").values

        T, _ = vals_x.shape
        max_h = max(horizons)

        X_all = sliding_window_view(vals_x, window_shape=seq_len, axis=0)
        Y_all = sliding_window_view(vals_y, window_shape=max_h, axis=0)

        n_samples = T - seq_len - max_h + 1
        X_win = X_all[:n_samples]
        Y_win = Y_all[seq_len : seq_len + n_samples]

        valid = ~np.isnan(X_win).any(axis=(1, 2)) & ~np.isnan(Y_win).any(axis=(1, 2))

        X_clean = X_win[valid]
        Y_clean = Y_win[valid]

        X_clean = np.swapaxes(X_clean, 1, 2)
        Y_clean = np.swapaxes(Y_clean, 1, 2)

        if X_clean.size == 0:
            print("No valid windows")

        return X_clean, Y_clean

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.from_numpy(self.y[idx]).float(),
        )


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
