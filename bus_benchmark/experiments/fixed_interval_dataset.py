from bus_benchmark import config
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Tuple
import logging
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedIntervalDataset:
    def __init__(
        self,
        df: Union[pd.DataFrame, str],
        n_splits: int = 5,
        freq: str = "15min",
        ha_agg_func: str = "median",
        ffill_limit: int = 16,
        mad_thresh: float = 3 * 1.4826,
        interpolate_ha: bool = True,
        calculate_residuals: bool = True,
        normalize: bool = True,
        add_time_feat: bool = True,
        min_split_size: int = (60 // 15) * 24 * 7,
    ) -> None:
        assert isinstance(df, (pd.DataFrame, str)), (
            "df must be a DataFrame or a path to a parquet file"
        )

        self.n_splits = n_splits
        self.freq = freq
        self.ha_agg_func = ha_agg_func
        self.ffill_limit = ffill_limit
        self.mad_thresh = mad_thresh
        self.interpolate_ha = interpolate_ha
        self.calculate_residuals = calculate_residuals
        self.normalize = normalize
        self.add_time_feat = add_time_feat
        self.min_split_size = min_split_size

        if isinstance(df, str):
            df = self._load_df(df)

        # ── TFT static metadata: per-link route/stop IDs ──
        self.link_metadata = self._build_link_metadata(df)

        self.splits = self._split_time_series(df)

        self.data_set = self._preprocess_splits(self.splits)

    @staticmethod
    def _load_df(df) -> pd.DataFrame:
        if isinstance(df, str) and df.endswith(".parquet"):
            df = pd.read_parquet(df)
        else:
            raise ValueError("df must be a path to a parquet file")
        return df

    def _split_time_series(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        start_time = df["from_time"].min()
        end_time = df["from_time"].max()
        split_size = (end_time - start_time) / (self.n_splits + 1)

        splits = []
        for i in range(self.n_splits):
            val_size = 1 / 6

            train_end = start_time + (i + 1) * split_size * (1 - val_size)
            val_end = start_time + (i + 1) * split_size
            test_end = start_time + (i + 2) * split_size

            train = df[(df["from_time"] < train_end)]
            val = df[(df["from_time"] >= train_end) & (df["from_time"] < val_end)]
            test = df[(df["from_time"] >= val_end) & (df["from_time"] < test_end)]

            logger.info(
                f"Split {i + 1}: train={len(train)} (from {start_time} to {train_end}), val={len(val)} (from {train_end} to {val_end}), test={len(test)} (from {val_end} to {test_end})"
            )

            splits.append((train, val, test))

        return splits

    def _calculate_ha(
        self,
        df: pd.DataFrame,
        link_col: str = "link",
        time_col: str = "from_time",
        travel_time_col: str = "travel_time",
    ) -> pd.DataFrame:
        links = df[link_col].unique().tolist()
        timeslots = pd.date_range(
            "1970-01-04", periods=4 * 24 * 7, freq=self.freq
        ).strftime("%w_%H:%M")
        complete_index = pd.MultiIndex.from_product([links, timeslots.to_list()])

        df_ha = df.groupby(
            [df[link_col], df[time_col].dt.floor(self.freq).dt.strftime("%w_%H:%M")]
        ).agg({travel_time_col: "median"})
        df_ha = df_ha.reindex(complete_index)
        df_ha.index = df_ha.index.rename(["link", "timeslot"])
        df_ha = df_ha.rename(columns={travel_time_col: "median"})

        def interpolate_group(grp):
            grp = pd.concat([grp, grp, grp])
            grp["median"] = grp["median"].interpolate(method="linear")
            return grp.iloc[len(grp) // 3 : len(grp) // 3 * 2]

        if self.interpolate_ha:
            df_ha = df_ha.groupby(level="link", group_keys=False).apply(
                interpolate_group
            )

        return df_ha

    def _calculate_ha_residuals(
        self,
        df: pd.DataFrame,
        df_ha: pd.DataFrame,
        agg: str = "median",
        time_col: str = "from_time",
        travel_time_col: str = "travel_time",
        link_col: str = "link",
    ) -> pd.DataFrame:
        if agg not in {"mean", "median"}:
            raise ValueError("`agg` must be either 'mean' or 'median'")

        # copy & ensure datetime (preserve timezone)
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

        # build your 15-min bin key
        df["timeslot"] = df[time_col].dt.floor(self.freq).dt.strftime("%w_%H:%M")

        # pull out just the desired stat from df_ha,
        # renaming its index levels to match df's columns
        stat_df = (
            df_ha[[agg]]
            .rename_axis(index=[link_col, "timeslot"])
            .reset_index()
            .rename(columns={agg: "ref_value"})
        )

        # merge & compute the deviation
        merged = df.merge(
            stat_df,
            on=[link_col, "timeslot"],
            how="left",
            validate="many_to_one",
        )
        diff_col = f"{travel_time_col}_minus_{agg}"
        merged[diff_col] = merged[travel_time_col] - merged["ref_value"]

        return merged

    def _pivot(
        self,
        df: pd.DataFrame,
        freq: str = "15min",
        ffill_limit=4,
        values_col="travel_time_minus_median",
    ) -> pd.DataFrame:
        def add_time_features(df):
            minutes = df.index.hour * 60 + df.index.minute
            df["sin_tod"] = np.sin(2 * np.pi * minutes / 1440)
            df["cos_tod"] = np.cos(2 * np.pi * minutes / 1440)
            return df

        # make sure from_time is datetime & floor it (preserve timezone)
        df_temp = df.copy()
        df_temp["time_slot"] = pd.to_datetime(df_temp["from_time"]).dt.floor(
            self.freq
        )

        # pivot to get a matrix: rows=time_slot, cols=link
        residual_matrix = (
            df_temp.pivot_table(
                index="time_slot",  # floored datetime index
                columns="link",  # one column per link
                values=values_col,  # your residuals
                aggfunc="mean",  # average if multiple days per slot
                fill_value=np.nan,  # or np.nan if you’d rather impute later
            ).sort_index()  # ensure chronological order
        )

        # build the full 15-min index matching the dataframe's timezone
        tz = residual_matrix.index.tz
        full_idx = pd.date_range(
            start=residual_matrix.index.min(),
            end=residual_matrix.index.max(),
            freq=freq,
            tz=tz,
        )

        # reindex to that full range
        residual_matrix = residual_matrix.reindex(full_idx)
        
        # # Chỉ ffill trong cùng một ngày (không copy sang ngày hôm sau)
        # filled = residual_matrix.groupby(residual_matrix.index.date).ffill(limit=ffill_limit)

        # Cho phép ffill liền mạch qua các ngày
        filled = residual_matrix.ffill(limit=ffill_limit)
        
        # Identify nighttime periods: 01:00 to 05:30 inclusive
        hour = residual_matrix.index.hour
        minute = residual_matrix.index.minute
        time_in_minutes = hour * 60 + minute
        is_nighttime = (time_in_minutes >= 1 * 60) & (time_in_minutes <= 5 * 60 + 30)
        
        # Không cho phép ffill ở khung giờ ban đêm (giữ nguyên giá trị gốc)
        filled.loc[is_nighttime] = residual_matrix.loc[is_nighttime]
        
        # Calculate exactly which records were imputed by forward fill
        was_imputed = residual_matrix.isna() & ~filled.isna()
        
        residual_matrix = filled
        residual_matrix.attrs["was_imputed"] = was_imputed
        
        if self.add_time_feat:
            residual_matrix = add_time_features(residual_matrix)

        return residual_matrix

    def _preprocess_splits(
        self, splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
    ) -> List[
        Tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            Union[StandardScaler, None],
            pd.DataFrame,
        ]
    ]:
        new_splits = []
        for df_train, df_val, df_test in splits:
            logger.info(
                f"Processing split with {len(df_train)} train, {len(df_val)} val, {len(df_test)} test rows."
            )

            original_train_size = len(df_train)
            original_val_size = len(df_val)
            original_test_size = len(df_test)

            # filter outliers
            df_train = self._filter_outliers_mad(df_train, target_df=df_train)
            df_val = self._filter_outliers_mad(df_train, target_df=df_val)
            df_test = self._filter_outliers_mad(df_train, target_df=df_test)

            if df_train.empty or df_val.empty or df_test.empty:
                logger.warning(
                    f"One of the splits is empty after filtering outliers. Sizes: train={len(df_train)} (was {original_train_size}), val={len(df_val)} (was {original_val_size}), test={len(df_test)} (was {original_test_size}). Skipping this split."
                )
                continue

            # compute historical average travel times
            df_ha = self._calculate_ha(
                df_train,
                link_col="link",
                time_col="from_time",
                travel_time_col="travel_time",
            )

            # transform travel times to residuals
            if self.calculate_residuals:
                df_residuals_train = self._calculate_ha_residuals(
                    df_train, df_ha, agg=self.ha_agg_func
                )
                df_residuals_val = self._calculate_ha_residuals(
                    df_val, df_ha, agg=self.ha_agg_func
                )
                df_residuals_test = self._calculate_ha_residuals(
                    df_test, df_ha, agg=self.ha_agg_func
                )
            else:
                df_residuals_train = df_train
                df_residuals_val = df_val
                df_residuals_test = df_test

            # pivot the travel times
            values_col = (
                "travel_time_minus_median"
                if self.calculate_residuals
                else "travel_time"
            )
            absolut_matrix_train = self._pivot(
                df_train, values_col="travel_time", ffill_limit=self.ffill_limit
            )
            residual_matrix_train = self._pivot(
                df_residuals_train, values_col=values_col, ffill_limit=self.ffill_limit
            )
            residual_matrix_val = self._pivot(
                df_residuals_val, values_col=values_col, ffill_limit=self.ffill_limit
            )
            residual_matrix_test = self._pivot(
                df_residuals_test, values_col=values_col, ffill_limit=self.ffill_limit
            )

            # Reorder link columns to match route sequence order from link_metadata
            # Giữ nguyên toàn bộ link_names từ metadata để đảm bảo khớp 1-1 với static covariates của TFT
            route_ordered_links = self.link_metadata.get("link_names", [])
            if not route_ordered_links:
                route_ordered_links = [
                    c for c in residual_matrix_train.columns if c not in ("sin_tod", "cos_tod")
                ]

            time_feat_cols = [
                c for c in ["sin_tod", "cos_tod"] if c in residual_matrix_train.columns
            ]
            ordered_cols = route_ordered_links + time_feat_cols

            def _reindex_with_attrs(mat: pd.DataFrame, cols: list) -> pd.DataFrame:
                was_imp = mat.attrs.get("was_imputed", None)
                mat_reindexed = mat.reindex(columns=cols)
                if was_imp is not None:
                    link_only_cols = [c for c in cols if c not in ("sin_tod", "cos_tod")]
                    mat_reindexed.attrs["was_imputed"] = was_imp.reindex(
                        columns=link_only_cols, fill_value=False
                    )
                return mat_reindexed

            if ordered_cols:
                abs_cols = route_ordered_links + [
                    c for c in ["sin_tod", "cos_tod"] if c in absolut_matrix_train.columns
                ]
                absolut_matrix_train = _reindex_with_attrs(absolut_matrix_train, abs_cols)
                residual_matrix_train = _reindex_with_attrs(residual_matrix_train, ordered_cols)
                residual_matrix_val = _reindex_with_attrs(residual_matrix_val, ordered_cols)
                residual_matrix_test = _reindex_with_attrs(residual_matrix_test, ordered_cols)

            if (
                len(residual_matrix_train) < self.min_split_size
                or len(residual_matrix_val) < self.min_split_size
                or len(residual_matrix_test) < self.min_split_size
            ):
                logger.warning(
                    f"One of the splits has less than {self.min_split_size} rows. Skipping this split."
                )
                continue

            # normalize the residuals
            if self.normalize:
                scaler = self._fit_scaler(residual_matrix_train)
                residual_matrix_train = self._apply_scaler(
                    residual_matrix_train, scaler
                )
                residual_matrix_val = self._apply_scaler(residual_matrix_val, scaler)
                residual_matrix_test = self._apply_scaler(residual_matrix_test, scaler)
            else:
                scaler = None

            new_splits.append(
                (
                    residual_matrix_train,
                    residual_matrix_val,
                    residual_matrix_test,
                    df_ha,
                    scaler,
                    absolut_matrix_train,
                )
            )
        return new_splits

    @staticmethod
    def _fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
        scaler = StandardScaler(with_mean=False, with_std=True)
        scaler.fit(
            train_df.drop(columns=["sin_tod", "cos_tod"], errors="ignore").values
        )
        return scaler

    def _apply_scaler(self, df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        scaled = scaler.transform(
            df.drop(columns=["sin_tod", "cos_tod"], errors="ignore").values
        )
        df_scaled = pd.DataFrame(
            scaled,
            index=df.index,
            columns=df.drop(columns=["sin_tod", "cos_tod"], errors="ignore").columns,
        )
        if "sin_tod" in df.columns and "cos_tod" in df.columns:
            # add sin/cos columns back to the scaled DataFrame
            df_scaled["sin_tod"] = df["sin_tod"]
            df_scaled["cos_tod"] = df["cos_tod"]
            
        df_scaled.attrs.update(df.attrs)
        return df_scaled

    def _filter_outliers_mad(
        self,
        reference_df: pd.DataFrame,
        target_df: pd.DataFrame,
        group_col: str = "link",
        target_col: str = "travel_time",
        log_base: float = np.e,
    ) -> pd.DataFrame:
        # compute per-group (med, mad) thresholds on base_df
        thresholds: dict = {}
        for group_val, sub in reference_df.groupby(group_col):
            sub_pos = sub[sub[target_col] > 0]
            if sub_pos.empty:
                continue
            logs = np.log(sub_pos[target_col]) / np.log(log_base)
            med = logs.median()
            mad = (logs - med).abs().median()
            thresholds[group_val] = (med, mad)

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            keep_idx = []
            for group_val, sub in df.groupby(group_col):
                if group_val not in thresholds:
                    continue
                med, mad = thresholds[group_val]
                pos_mask = sub[target_col] > 0
                logs = np.log(sub.loc[pos_mask, target_col]) / np.log(log_base)

                if mad == 0:
                    valid_idx = sub.loc[pos_mask].index
                else:
                    dev = (logs - med).abs() / mad
                    valid_idx = logs[dev <= self.mad_thresh].index

                keep_idx.extend(valid_idx)

            if not keep_idx:
                return df.iloc[[]].copy()

            return df.loc[sorted(keep_idx)].reset_index(drop=True)

        return _filter(target_df)

    @staticmethod
    def _build_link_metadata(df: pd.DataFrame) -> dict:
        """Extract per-link static metadata for TFT static covariates.

        Calculates link_length via Haversine, reconstructs route sequence 
        from link chain ([A]>[B], [B]>[C]),
        and assigns integer IDs.

        Returns:
            dict with keys:
                'link_names':          list[str]     — ordered link column names
                'link_ids':            ndarray int   — integer ID per link
                'link_lengths':        ndarray float — length in meters
                'num_links':           int
                'route_ids':           ndarray int
                'num_routes':          int
                'from_stop_ids':       ndarray int
                'num_from_stops':      int
                'to_stop_ids':         ndarray int
                'num_to_stops':        int
                'stop_sequence_norms': ndarray float32
        """
        import re
        from collections import defaultdict
        import numpy as np

        def haversine(lon1, lat1, lon2, lat2):
            R = 6371000  # radius of Earth in meters
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)
            a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
            return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        def extract_coords(point_str):
            if not isinstance(point_str, str):
                return np.nan, np.nan
            m = re.search(r"POINT\s*\(\s*([\d\.\-]+)\s+([\d\.\-]+)\s*\)", point_str)
            if m:
                return float(m.group(1)), float(m.group(2))
            return np.nan, np.nan

        def _order_links_by_chain(sub_df: pd.DataFrame) -> List[str]:
            """Sắp xếp các link theo chuỗi lộ trình liên tục [A]>[B] -> [B]>[C]."""
            edges = {}
            out_adj = defaultdict(list)
            in_degree = defaultdict(int)
            nodes = set()
            unparseable_links = []

            for _, row in sub_df.iterrows():
                lk = str(row["link"])
                # Ưu tiên tách trực tiếp từ tên link có định dạng A>B
                if ">" in lk:
                    u, v = lk.split(">", 1)
                    u, v = u.strip(), v.strip()
                elif (
                    "from_stop" in row
                    and "to_stop" in row
                    and pd.notna(row["from_stop"])
                    and pd.notna(row["to_stop"])
                ):
                    u, v = str(row["from_stop"]).strip(), str(row["to_stop"]).strip()
                else:
                    unparseable_links.append(lk)
                    continue

                edges[lk] = (u, v)
                out_adj[u].append((v, lk))
                in_degree[v] += 1
                nodes.add(u)
                nodes.add(v)

            if not edges:
                return sorted(sub_df["link"].astype(str).tolist())

            # Sắp xếp danh sách kề để đảm bảo tính tất định (deterministic) nếu có nhánh rẽ
            for u in out_adj:
                out_adj[u].sort(key=lambda x: x[1])

            visited_links = set()
            ordered_links = []

            def traverse(start_node):
                curr = start_node
                while True:
                    next_step = None
                    for nxt_node, lk in out_adj[curr]:
                        if lk not in visited_links:
                            next_step = (nxt_node, lk)
                            break
                    if next_step is None:
                        break
                    nxt_node, lk = next_step
                    visited_links.add(lk)
                    ordered_links.append(lk)
                    curr = nxt_node

            # 1. Tìm các điểm đầu tuyến (in_degree == 0) và duyệt trước
            start_nodes = sorted([u for u in nodes if in_degree[u] == 0 and u in out_adj])
            for s in start_nodes:
                traverse(s)

            # 2. Xử lý các đoạn bị đứt gãy hoặc tuyến vòng tròn (cycle) nếu còn sót
            remaining_links = sorted([lk for lk in edges if lk not in visited_links])
            for lk in remaining_links:
                if lk not in visited_links:
                    u, _ = edges[lk]
                    traverse(u)

            # 3. Ghép thêm các link không theo chuẩn A>B (nếu có) ở cuối
            ordered_links.extend(sorted(unparseable_links))
            return ordered_links

        if "link" not in df.columns:
            return {}

        # Deduplicate: one row per link
        cols_to_keep = [
            "link",
            "from_geometry",
            "to_geometry",
            "route",
            "from_stop",
            "to_stop",
        ]
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        link_info = df.drop_duplicates("link")[existing_cols].copy()

        # Compute lengths
        if "from_geometry" in link_info.columns and "to_geometry" in link_info.columns:
            lons1, lats1 = zip(*link_info["from_geometry"].apply(extract_coords))
            lons2, lats2 = zip(*link_info["to_geometry"].apply(extract_coords))
            link_info["length_m"] = haversine(
                np.array(lons1), np.array(lats1), np.array(lons2), np.array(lats2)
            )
            link_info["length_m"] = link_info["length_m"].fillna(0.0)
        else:
            link_info["length_m"] = 0.0

        # Tự động khôi phục thứ tự lộ trình và gán stop_sequence từ chuỗi liên kết [A]>[B]
        if "route" in link_info.columns:
            ordered_dfs = []
            for _, grp in link_info.groupby("route", sort=False):
                ordered_lks = _order_links_by_chain(grp)
                grp_ordered = grp.set_index("link").loc[ordered_lks].reset_index()
                grp_ordered["stop_sequence"] = np.arange(len(grp_ordered), dtype=np.float32)
                ordered_dfs.append(grp_ordered)
            link_info = pd.concat(ordered_dfs, ignore_index=True)
        else:
            ordered_lks = _order_links_by_chain(link_info)
            link_info = link_info.set_index("link").loc[ordered_lks].reset_index()
            link_info["stop_sequence"] = np.arange(len(link_info), dtype=np.float32)

        # Calculate normalized stop sequence (luôn hoạt động kể cả khi vừa khôi phục từ chuỗi)
        if "route" in link_info.columns:
            group_min = link_info.groupby("route")["stop_sequence"].transform("min")
            group_max = link_info.groupby("route")["stop_sequence"].transform("max")
        else:
            group_min = link_info["stop_sequence"].min()
            group_max = link_info["stop_sequence"].max()

        range_diff = group_max - group_min
        link_info["stop_sequence_norm"] = np.where(
            range_diff > 0,
            (link_info["stop_sequence"] - group_min) / range_diff,
            0.0,
        ).astype(np.float32)

        # Assign integer encode (theo đúng thứ tự lộ trình đã sắp xếp)
        link_vocab = {v: i for i, v in enumerate(link_info["link"].unique())}
        link_info["link_id"] = link_info["link"].map(link_vocab)

        nums = {}
        for col, col_id_name, num_name in [
            ("route", "route_id", "num_routes"),
            ("from_stop", "from_stop_id", "num_from_stops"),
            ("to_stop", "to_stop_id", "num_to_stops"),
        ]:
            if col in link_info.columns:
                vocab = {v: i for i, v in enumerate(link_info[col].unique())}
                link_info[col_id_name] = link_info[col].map(vocab)
                nums[num_name] = len(vocab)
            else:
                link_info[col_id_name] = 0
                nums[num_name] = 1

        return {
            "link_names": link_info["link"].tolist(),
            "link_ids": link_info["link_id"].values.astype(int),
            "link_lengths": link_info["length_m"].values.astype(float),
            "num_links": len(link_vocab),
            "route_ids": link_info["route_id"].values.astype(int),
            "num_routes": nums["num_routes"],
            "from_stop_ids": link_info["from_stop_id"].values.astype(int),
            "num_from_stops": nums["num_from_stops"],
            "to_stop_ids": link_info["to_stop_id"].values.astype(int),
            "num_to_stops": nums["num_to_stops"],
            "stop_sequence_norms": link_info["stop_sequence_norm"].values.astype(np.float32),
        }
