# Bus Link Travel Time Prediction with Uncertainty Quantification

Mã nguồn thực nghiệm Đồ án tốt nghiệp Thạc sĩ HK253: 

**Ứng dụng mô hình dựa trên Transformer trong dự báo thời gian di chuyển của xe buýt** (*Bus Travel Time Prediction Using a Transformer-Based Model*)

---

## 📁 Cấu trúc Notebooks

| File Notebook | Mô tả |
| --- | --- |
| [`notebooks/plot_trajectory.ipynb`](notebooks/plot_trajectory.ipynb)| Trực quan hóa tuyến đường sử dụng trong dataset và các trạm dừng trên bản đồ thực tế (sử dụng OSRM API + Esri tiles). |
| [`notebooks/EDA_Benchmark_Dataset_Statistics.ipynb`](notebooks/EDA_Benchmark_Dataset_Statistics.ipynb) | Phân tích thống kê dữ liệu (EDA). |
| [`notebooks/Rotterdam_ConvLSTM_quantiles.ipynb`](notebooks/Rotterdam_ConvLSTM_quantiles.ipynb) | Huấn luyện & đánh giá mô hình baseline **DQR-ConvLSTM**. |
| [`notebooks/Rotterdam_TFT_quantiles_Multi_entities.ipynb`](notebooks/Rotterdam_TFT_quantiles_Multi_entities.ipynb)| Huấn luyện & đánh giá thực nghiệm framework đề xuất: **A TFT-Based Framework with Conformalized Quantile Regression for Bus Link Travel Time Prediction with Uncertainty Quantification**. Notebook bao gồm huấn luyện mô hình **Temporal Fusion Transformer**, trực quan hóa kết quả dự đoán vs kết quả thực tế cũng như tính diễn giải (**Interpretability**) của mô hình, và hiệu chuẩn độ tin cậy bằng **Conformalized Quantile Regression**. |




