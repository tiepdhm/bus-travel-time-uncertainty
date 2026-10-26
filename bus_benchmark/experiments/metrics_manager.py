import torch
import pandas as pd
from typing import Literal, List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler
from .run_metrics import RunMetrics

from bus_benchmark.experiments.utils import geometric_mean


class MetricsManager:
    """
    Class to manage training metrics.
    """

    def __init__(self) -> None:
        self.run_metrics_objects: List[RunMetrics] = []

    def init_run_metrics(
        self,
        prefix: Literal["eval", "test"],
        absolut_matrix_train: Optional[pd.DataFrame] = None,
        x_list: Optional[list[torch.Tensor]] = None,
        y_list: Optional[list[torch.Tensor]] = None,
        scaler: Optional[StandardScaler] = None,
    ) -> RunMetrics:
        run_metrics = RunMetrics(
            prefix=prefix,
            absolut_matrix_train=absolut_matrix_train,
            x_list=x_list,
            y_list=y_list,
            scaler=scaler,
        )

        self.run_metrics_objects.append(run_metrics)

        return run_metrics

    @property
    def final_metrics(self) -> List[Dict[str, torch.Tensor]]:
        return [run_metrics.final_metrics for run_metrics in self.run_metrics_objects]

    @property
    def final_metrics_list(self) -> Dict[str, List[Union[torch.Tensor, float]]]:
        final_metrics = self.final_metrics
        final_metrics_list = {
            k: [d.get(k) for d in final_metrics] for k in set().union(*final_metrics)
        }
        return final_metrics_list  # type: ignore

    @property
    def test_geometric_means(self) -> Dict[str, float]:
        final_metrics_list = self.final_metrics_list
        result = {}
        for k, v in final_metrics_list.items():
            valid_v = [x for x in v if x is not None]
            if len(valid_v) > 0:
                new_key = k.replace("test", "test_GM").replace("best_eval", "best_eval_GM")
                result[new_key] = geometric_mean(valid_v)
        return result

    @property
    def final_log_metrics(
        self,
    ) -> Dict[str, Union[float, List[Union[torch.Tensor, float]]]]:
        return self.final_metrics_list | self.test_geometric_means
