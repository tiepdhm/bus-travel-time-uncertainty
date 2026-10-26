# Bus Travel Time Prediction Benchmark (Rotterdam)

Mã nguồn thực nghiệm dự báo thời gian di chuyển xe buýt (Bus Travel Time) sử dụng các mô hình học sâu chuỗi thời gian: **Quantile ConvLSTM** và **Temporal Fusion Transformer (TFT)** kết hợp **Conformal Prediction**.

---

## 📁 Cấu trúc Notebooks

| File Notebook | Mô tả & Chức năng | Khuyến nghị Runtime |
| :--- | :--- | :--- |
| [`notebooks/plot_trajectory.ipynb`](notebooks/plot_trajectory.ipynb) | Trực quan hóa tuyến đường và trạm dừng trên bản đồ thực tế (OSRM API + Esri tiles). | CPU |
| [`notebooks/EDA_Benchmark_Dataset_Statistics.ipynb`](notebooks/EDA_Benchmark_Dataset_Statistics.ipynb) | Phân tích thống kê dữ liệu (EDA), phân bổ 96 slot 15-phút, độ dài link và chuỗi lộ trình. | CPU |
| [`notebooks/Rotterdam_ConvLSTM_quantiles.ipynb`](notebooks/Rotterdam_ConvLSTM_quantiles.ipynb) | Huấn luyện & đánh giá mô hình **Quantile ConvLSTM** đa bước (15, 30, 60, 120 phút) kèm dải phân vị (p10, p50, p90). | GPU (T4 / CUDA) |
| [`notebooks/Rotterdam_TFT_quantiles_Multi_entities.ipynb`](notebooks/Rotterdam_TFT_quantiles_Multi_entities.ipynb) | Huấn luyện mô hình **TFT Multi-entities**, trực quan hóa Attention weights và hiệu chuẩn độ tin cậy bằng **Conformal Prediction**. | GPU (T4 / CUDA) |

---

## ⚙️ Cài đặt môi trường

### 1. Dùng Conda / Local (env `ai_workspace`)
```bash
conda create -n ai_workspace python=3.10 -y
conda activate ai_workspace
pip install -r requirements.txt
```

### 2. Dùng trên Google Colab
Thêm 1 cell ở đầu notebook để clone repo, cài lib và chuẩn bị data:
```python
!git clone https://github.com/<your-username>/<your-repo-name>.git
%cd <your-repo-name>/notebooks

!pip install -q -r ../requirements.txt gdown -U
os.environ["WANDB_MODE"] = "offline"
```

---

## 📊 Chuẩn bị Dữ liệu

File dữ liệu: `GM0599.parquet` (~391 MB, dữ liệu xe buýt Rotterdam). Đặt tại đường dẫn:
```text
data/
└── parquet_data/
    └── GM0599.parquet
```

* **Trên Google Colab** (tải trực tiếp từ Google Drive bằng `gdown` không cần mount):
```bash
import os
os.makedirs('../data/parquet_data', exist_ok=True)
!gdown --id <GOOGLE_DRIVE_FILE_ID> -O ../data/parquet_data/GM0599.parquet
```

---

## 🚀 Hướng dẫn chạy

1. **Khám phá dữ liệu**: Mở `EDA_Benchmark_Dataset_Statistics.ipynb` hoặc `plot_trajectory.ipynb` và chọn `Run All`.
2. **Huấn luyện mô hình**:
   - Chạy `Rotterdam_ConvLSTM_quantiles.ipynb` hoặc `Rotterdam_TFT_quantiles_Multi_entities.ipynb`.
   - Các notebook sẽ tự động chia 5 split cross-validation theo chuỗi thời gian, huấn luyện mô hình, đánh giá các chỉ số (MAE, RMSE, Pinball Loss) và vẽ đồ thị dự đoán các phân vị.
