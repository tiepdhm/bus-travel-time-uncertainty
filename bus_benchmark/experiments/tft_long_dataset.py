"""TFT-specific multi-step dataset for Long Format (Entity-based).

Produces (x_packed, y) tuples compatible with ModelTrainerBase's training loop.
Each sample corresponds to a distinct time window + link_idx combination.

x_packed contains:
  - observed link value (imputed + forward filled)
  - is_imputed mask (0 for real data, 1 for forward filled)
  - historical lag values
  - temporal known features
  - link_idx (appended as last dimension)
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFTLongDataset(Dataset):
    """PyTorch Dataset for TFT Long Format (Entity-based).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        encoder_len: int,
        decoder_len: int,
        use_is_imputed: bool = True,
        use_tod: bool = True,
        use_dow: bool = True,
        use_stop_sequence: bool = False,
        use_pattern_profile: bool = False,
        df_ha: pd.DataFrame = None,
        align_windows: bool = False,
        use_lag_1_week: bool = False,
        **kwargs,
    ) -> None:
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.total_len = encoder_len + decoder_len
        self.use_is_imputed = use_is_imputed
        self.use_lag_1_week = use_lag_1_week
        self.use_tod = use_tod
        self.use_dow = use_dow
        self.use_stop_sequence = use_stop_sequence
        self.use_pattern_profile = use_pattern_profile
        self.align_windows = align_windows

        link_cols = [c for c in df.columns if c not in ("sin_tod", "cos_tod")]
        self.n_links = len(link_cols)

        # Raw observations (no longer locally ffilled)
        raw_obs = df[link_cols].values.astype(np.float32)  # (T, H)
        
        self.obs_data_filled = raw_obs
        
        if "was_imputed" in df.attrs:
            # Mask identifying positions that were imputed by FixedIntervalDataset
            self.is_imputed = df.attrs["was_imputed"][link_cols].values.astype(np.float32)
        else:
            self.is_imputed = np.zeros_like(raw_obs, dtype=np.float32)
        
        # known data
        self.known_data = self._compute_temporal_features(df.index)
        
        if self.use_pattern_profile:
            assert df_ha is not None, "df_ha must be provided when use_pattern_profile is True"
            # Extract and scale pattern historical medians
            ha_pivoted = df_ha.reset_index().pivot(index='timeslot', columns='link', values='median')
            ha_pivoted = ha_pivoted.reindex(columns=link_cols)
            # Global mean for completely missing links/slots
            global_mean = ha_pivoted.mean().mean()
            ha_pivoted = ha_pivoted.fillna(global_mean)
            
            # Standard scaling per-link
            mean = ha_pivoted.mean()
            std = ha_pivoted.std().replace(0, 1.0)
            ha_scaled = (ha_pivoted - mean) / std
            
            # Map DatetimeIndex to timeslot strings
            # We skip explicit floor to avoid AmbiguousTimeError on DST boundaries,
            # as the dataset index is already securely aligned to 15-minute bounds.
            timeslots = pd.DatetimeIndex(df.index).strftime("%w_%H:%M")
            self.pattern_data = ha_scaled.loc[timeslots].values.astype(np.float32)

        T = len(df)
        n_windows = T - self.total_len + 1
        self.n_windows = n_windows

        if n_windows <= 0:
            logger.warning("Dataset too short")
            self.valid_pairs = np.empty((0, 2), dtype=int)
            self.n_samples = 0
            return

        # A window w_idx is valid for link h if the decoder portion (w_idx+enc: w_idx+enc+dec) has no NaNs
        is_nan = np.isnan(raw_obs)
        if self.align_windows:
            from numpy.lib.stride_tricks import sliding_window_view
            X_all = sliding_window_view(raw_obs, window_shape=self.encoder_len, axis=0)[:n_windows]
            Y_all = sliding_window_view(raw_obs, window_shape=self.decoder_len, axis=0)[self.encoder_len : self.encoder_len + n_windows]
            valid_1d = ~np.isnan(X_all).any(axis=(1, 2)) & ~np.isnan(Y_all).any(axis=(1, 2))
            valid_mask = np.tile(valid_1d[:, None], (1, self.n_links))
        else:
            valid_mask = np.ones((n_windows, self.n_links), dtype=bool)
            for i in range(self.decoder_len):
                valid_mask &= ~is_nan[self.encoder_len + i : self.encoder_len + i + n_windows, :]
            
        if self.use_lag_1_week:
            # Need minimum 672 timesteps of past data
            valid_mask[:672] = False
            
        w_indices, h_indices = np.where(valid_mask)
        self.valid_pairs = np.column_stack([w_indices, h_indices])
        self.n_samples = len(self.valid_pairs)
        
        if self.n_samples == 0:
            logger.warning("No valid windows found.")
        else:
            logger.info("Created %d entity-based samples from %d time windows across %d links", self.n_samples, n_windows, self.n_links)

    @staticmethod
    def _compute_temporal_features(index: pd.DatetimeIndex) -> np.ndarray:
        idx = pd.DatetimeIndex(index)
        minutes = idx.hour * 60 + idx.minute
        dow = idx.dayofweek
        two_pi = 2.0 * np.pi
        features = np.column_stack([
            np.sin(two_pi * minutes / 1440.0),
            np.cos(two_pi * minutes / 1440.0),
            np.sin(two_pi * dow / 7.0),
            np.cos(two_pi * dow / 7.0),
        ]).astype(np.float32)
        return features

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        w_idx, l_idx = self.valid_pairs[idx]
        
        enc = self.encoder_len
        total = self.total_len
        
        # 1. Observed (filled) and is_imputed mask: only encoder portion has real values
        obs_enc = self.obs_data_filled[w_idx : w_idx + enc, l_idx:l_idx+1] # (T_enc, 1)
        obs_dec = np.zeros((self.decoder_len, 1), dtype=np.float32)
        obs_full = np.concatenate([obs_enc, obs_dec], axis=0) # (T_total, 1)
        
        pack_list = [obs_full]
        
        if self.use_is_imputed:
            imp_enc = self.is_imputed[w_idx : w_idx + enc, l_idx:l_idx+1]
            imp_dec = np.zeros((self.decoder_len, 1), dtype=np.float32)
            imp_full = np.concatenate([imp_enc, imp_dec], axis=0) # (T_total, 1)
            pack_list.append(imp_full)
        
        # 2. Known temporal features
        if self.use_lag_1_week:
            lag1 = self.obs_data_filled[w_idx - 672 : w_idx + total - 672, l_idx:l_idx+1]
            pack_list.append(lag1)
            
        known_full = self.known_data[w_idx : w_idx + total] # (T_total, 4)
        if self.use_tod:
            pack_list.append(known_full[:, 0:2])
        if self.use_dow:
            pack_list.append(known_full[:, 2:4])
            
        if self.use_pattern_profile:
            pattern_full = self.pattern_data[w_idx : w_idx + total, l_idx:l_idx+1]
            pack_list.append(pattern_full)
            
        # 3. Link Idx (appended as last feature)
        link_idx_arr = np.full((total, 1), l_idx, dtype=np.float32)
        pack_list.append(link_idx_arr)
        
        x_packed = np.concatenate(pack_list, axis=-1)
        
        # 4. Target Y (ground truth without imputation)
        y = self.obs_data_filled[w_idx + enc : w_idx + total, l_idx:l_idx+1] # (T_dec, 1)
        
        return (
            torch.from_numpy(x_packed),
            torch.from_numpy(y.copy())
        )
