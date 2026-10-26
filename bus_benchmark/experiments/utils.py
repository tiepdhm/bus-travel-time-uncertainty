import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Union
import torch


def geometric_mean(nums: List[Union[float, torch.Tensor]]) -> float:
    log_sum = sum(math.log(x) for x in nums)
    return math.exp(log_sum / len(nums))


def preprocess_csv(df: pd.DataFrame) -> pd.DataFrame:
    # date conversion
    df["date"] = pd.to_datetime(df["date"])

    # time conversion to UTC
    df["from_time"] = pd.to_datetime(df["from_time"], utc=True)
    df["to_time"] = pd.to_datetime(df["to_time"], utc=True)

    # compute travel_time
    df["travel_time"] = (df["to_time"] - df["from_time"]).dt.total_seconds()

    # create link column
    df["link"] = df["from_stop"] + ">" + df["to_stop"]

    # reorder columns
    columns_order = [
        "lau",
        "date",
        "line",
        "trip",
        "from_stop",
        "to_stop",
        "from_geometry",
        "to_geometry",
        "from_time",
        "to_time",
        "travel_time",
        "link",
        "route",
    ]
    df = df[columns_order]

    return df


# Hàm vẽ biểu đồ dự đoán so với thực tế trên tập test cho tất cả các link
def plot_test_predictions_all_links(trainer, horizon_idx=0, num_samples=200):
    """
    Vẽ biểu đồ so sánh y_true và y_pred cho tất cả các links (trạm) cùng một lúc.
    - horizon_idx: Chỉ số của horizon muốn vẽ (ví dụ: 0 tương ứng với horizon đầu tiên trong horizons).
    - num_samples: Số lượng điểm dữ liệu muốn hiển thị trên biểu đồ để dễ quan sát.
    """
    if not hasattr(trainer, 'test_y_true') or not hasattr(trainer, 'test_y_pred'):
        print("Không tìm thấy kết quả dự đoán. Vui lòng chạy lại trainer.run_experiment() sau khi cập nhật mã nguồn.")
        return
        
    y_true = trainer.test_y_true
    y_pred = trainer.test_y_pred
    
    # Kích thước của y_true và y_pred là: (num_samples, len(horizons), num_links)
    num_links = y_true.shape[2]
    horizon_val = trainer.horizons[horizon_idx]
    model_name = getattr(trainer, 'model_name', 'Model')
    
    # Tạo layout lưới cho các subplot (2 cột, số hàng tự động tính toán)
    cols = 2
    rows = (num_links + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if num_links > 1 else [axes]
    
    for link_idx in range(num_links):
        y_true_sample = y_true[:num_samples, horizon_idx, link_idx]
        y_pred_sample = y_pred[:num_samples, horizon_idx, link_idx]
        
        ax = axes[link_idx]
        ax.plot(y_true_sample, label='Thực tế (y_true)', color='blue', linewidth=2)
        ax.plot(y_pred_sample, label='Dự đoán (y_pred)', color='red', linestyle='--', linewidth=2)
        ax.set_title(f"Link: {link_idx}")
        ax.set_xlabel("Thời gian (Time steps)")
        ax.set_ylabel("Travel time (đã chuẩn hoá)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    # Xóa các ô trống nếu số links không lấp đầy lưới
    for i in range(num_links, len(axes)):
        fig.delaxes(axes[i])
        
    fig.suptitle(f"Dự đoán vs Thực tế - {model_name} (Horizon: {horizon_val}) - Tất cả các link", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_test_predictions_quantiles_all_links(trainer, horizon_idx=0, num_samples=200):
    """
    Vẽ biểu đồ so sánh y_true và 3 quantile của y_pred cho tất cả các links (trạm) cùng một lúc.
    """
    if not hasattr(trainer, "test_y_true") or not hasattr(trainer, "test_y_pred_quantiles"):
        print("Không tìm thấy kết quả dự đoán (test_y_pred_quantiles). Vui lòng chạy lại trainer.run_experiment() sau khi cập nhật mã nguồn tft_model_trainer.")
        return
        
    y_true = trainer.test_y_true
    y_pred = trainer.test_y_pred_quantiles
    
    num_links = y_true.shape[2]
    horizon_val = trainer.horizons[horizon_idx]
    model_name = getattr(trainer, "model_name", "TFT Model")
    
    cols = 2
    rows = (num_links + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if num_links > 1 else [axes]
    
    for link_idx in range(num_links):
        y_true_sample = y_true[:num_samples, horizon_idx, link_idx]
        y_pred_p10 = y_pred[:num_samples, horizon_idx, link_idx, 0]
        y_pred_p50 = y_pred[:num_samples, horizon_idx, link_idx, 1]
        y_pred_p90 = y_pred[:num_samples, horizon_idx, link_idx, 2]
        
        ax = axes[link_idx]
        ax.plot(y_true_sample, label="Actual (y_true)", color="blue", linewidth=2)
        ax.plot(y_pred_p50, label="Predicted (p50)", color="red", linestyle="--", linewidth=2)

        # MỚI THÊM: Vẽ rõ đường viền của p10 và p90 với các màu khác nhau 
        ax.plot(y_pred_p10, label="p10", color="green", linestyle="-", linewidth=1.5)
        ax.plot(y_pred_p90, label="p90", color="orange", linestyle="-", linewidth=1.5)
        
        # GIỮ NGUYÊN: Fill nền đỏ mờ
        ax.fill_between(range(len(y_pred_p50)), y_pred_p10, y_pred_p90, color="red", alpha=0.2, label="Prediction interval (p10-p90)")
        
        ax.set_title(f"Link: {link_idx}")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Travel time")
        ax.grid(True, alpha=0.3)
        
    for i in range(num_links, len(axes)):
        fig.delaxes(axes[i])
        
    # Tạo legend chung cho toàn bộ biểu đồ tại vị trí trên cùng
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5)
        
    fig.suptitle(f"Actual vs Predicted - {model_name} (Horizon: {horizon_val}) - All links", fontsize=16, y=1.02)
    # Điều chỉnh layout để có chỗ trống cho legend và title chung
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_test_predictions_mean_quantiles_all_links(trainer, horizon_idx=0, num_samples=200):
    """
    Vẽ biểu đồ so sánh y_true và dự đoán bằng MEAN, cùng với 2 quantile (p10, p90) cho tất cả các links.
    (Chỉ dành riêng cho ConvLSTM đã lưu test_y_pred_mean)
    """
    if not hasattr(trainer, "test_y_true") or not hasattr(trainer, "test_y_pred_mean") or not hasattr(trainer, "test_y_pred_quantiles"):
        print("Không tìm thấy kết quả dự đoán (test_y_pred_mean hoặc test_y_pred_quantiles). Vui lòng thử đánh giá lại mô hình.")
        return
        
    y_true = trainer.test_y_true
    y_pred_mean = trainer.test_y_pred_mean
    y_pred_quantiles = trainer.test_y_pred_quantiles
    
    num_links = y_true.shape[2]
    horizon_val = trainer.horizons[horizon_idx]
    model_name = getattr(trainer, "model_name", "ConvLSTM")
    
    cols = 2
    rows = (num_links + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if num_links > 1 else [axes]
    
    for link_idx in range(num_links):
        y_true_sample = y_true[:num_samples, horizon_idx, link_idx]
        y_pred_mean_sample = y_pred_mean[:num_samples, horizon_idx, link_idx]
        y_pred_p10 = y_pred_quantiles[:num_samples, horizon_idx, link_idx, 0]
        y_pred_p90 = y_pred_quantiles[:num_samples, horizon_idx, link_idx, 2] # 0 = p10, 1 = p50, 2 = p90
        
        ax = axes[link_idx]
        ax.plot(y_true_sample, label="Actual (y_true)", color="blue", linewidth=2)
        ax.plot(y_pred_mean_sample, label="Predicted (Mean)", color="red", linestyle="--", linewidth=2)

        ax.plot(y_pred_p10, label="p10", color="green", linestyle="-", linewidth=1.5)
        ax.plot(y_pred_p90, label="p90", color="orange", linestyle="-", linewidth=1.5)
        
        ax.fill_between(range(len(y_pred_mean_sample)), y_pred_p10, y_pred_p90, color="red", alpha=0.2, label="Prediction interval (p10-p90)")
        
        ax.set_title(f"Link: {link_idx}")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Travel time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    for i in range(num_links, len(axes)):
        fig.delaxes(axes[i])
        
    fig.suptitle(f"Predicted vs Actual - {model_name} (Using Mean - Horizon: {horizon_val}) - All links", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_test_predictions_calibrated_quantiles_all_links(trainer, horizon_idx=0, num_samples=200):
    """
    Vẽ biểu đồ so sánh y_true và 3 quantile của y_pred sau khi Conformal Calibration (p10_calibrated, p50, p90_calibrated)
    cho tất cả các links (trạm) cùng một lúc.
    - horizon_idx: Chỉ số của horizon muốn vẽ (ví dụ: 0 tương ứng với horizon đầu tiên trong horizons).
    - num_samples: Số lượng điểm dữ liệu muốn hiển thị trên biểu đồ để dễ quan sát.
    """
    if not hasattr(trainer, "test_y_true") or not hasattr(trainer, "test_y_pred_quantiles_calibrated"):
        print("Không tìm thấy kết quả dự đoán calibrated (test_y_pred_quantiles_calibrated). Vui lòng gọi trainer.use_conformal(nominal_coverage=...) rồi chạy trainer.evaluate_only().")
        return
        
    y_true = trainer.test_y_true
    y_pred = trainer.test_y_pred_quantiles_calibrated
    
    num_links = y_true.shape[2]
    horizon_val = trainer.horizons[horizon_idx]
    model_name = getattr(trainer, "model_name", "TFT Model")
    coverage_pct = f"{trainer.nominal_coverage * 100:.0f}%" if hasattr(trainer, "nominal_coverage") else "Calibrated"
    
    cols = 2
    rows = (num_links + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if num_links > 1 else [axes]
    
    for link_idx in range(num_links):
        y_true_sample = y_true[:num_samples, horizon_idx, link_idx]
        y_pred_p10 = y_pred[:num_samples, horizon_idx, link_idx, 0]
        y_pred_p50 = y_pred[:num_samples, horizon_idx, link_idx, 1]
        y_pred_p90 = y_pred[:num_samples, horizon_idx, link_idx, 2]
        
        ax = axes[link_idx]
        ax.plot(y_true_sample, label="Actual (y_true)", color="blue", linewidth=2)
        ax.plot(y_pred_p50, label="Predicted (p50)", color="red", linestyle="--", linewidth=2)

        ax.plot(y_pred_p10, label="Calibrated p10", color="green", linestyle="-", linewidth=1.5)
        ax.plot(y_pred_p90, label="Calibrated p90", color="orange", linestyle="-", linewidth=1.5)
        
        ax.fill_between(range(len(y_pred_p50)), y_pred_p10, y_pred_p90, color="teal", alpha=0.2, label=f"Prediction interval Calibrated ({coverage_pct})")
        
        ax.set_title(f"Link: {link_idx}")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Travel time")
        ax.grid(True, alpha=0.3)
        
    for i in range(num_links, len(axes)):
        fig.delaxes(axes[i])
        
    # Tạo legend chung cho toàn bộ biểu đồ tại vị trí trên cùng
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5)
        
    fig.suptitle(f"Predicted vs Actual (Conformal Calibrated - {coverage_pct}) - {model_name} (Horizon: {horizon_val}) - All links", fontsize=16, y=1.02)
    # Điều chỉnh layout để có chỗ trống cho legend và title chung
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Alias tiện dụng
plot_test_predictions_quantiles_calibrated_all_links = plot_test_predictions_calibrated_quantiles_all_links

