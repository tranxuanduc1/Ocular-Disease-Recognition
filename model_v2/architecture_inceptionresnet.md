# Dàn ý kiến trúc mô hình: Siamese Multimodal Network (InceptionResNetV2)

## 1. Tổng quan

Mô hình được thiết kế để phân loại đa nhãn 8 bệnh lý mắt (N, D, G, C, A, H, M, O) từ ảnh đáy mắt của cả hai mắt kết hợp với thông tin nhân khẩu học (tuổi, giới tính).

---

## 2. Đầu vào (Input)

| Nhánh | Dữ liệu | Kích thước |
|---|---|---|
| Ảnh mắt trái | Fundus image, RGB, chuẩn hóa ImageNet | 3 × 299 × 299 |
| Ảnh mắt phải | Fundus image, RGB, chuẩn hóa ImageNet | 3 × 299 × 299 |
| Đặc trưng bảng | Tuổi (z-score), giới tính (one-hot: Male/Female) | 3 |

---

## 3. Kiến trúc chi tiết

### 3.1 Nhánh trích xuất đặc trưng ảnh (Image Branch — Siamese)

- **Backbone**: InceptionResNetV2, pretrained trên ImageNet
- **Cơ chế Siamese**: Dùng **chung một bộ trọng số** cho cả hai ảnh (mắt trái và mắt phải)
- **Pooling**: Global Average Pooling (GAP) → vector đặc trưng chiều `feat_dim`
- **Projection head** (per eye):
  ```
  Linear(feat_dim → 128) → ReLU → Dropout(0.4)
  ```
- **Ghép đặc trưng hai mắt**:
  ```
  [f_left (128-D) ‖ f_right (128-D)] → 256-D
  ```

### 3.2 Nhánh mã hóa đặc trưng bảng (Tabular Branch)

- **Đầu vào**: `[age_norm, gender_Male, gender_Female]` — 3 chiều
- **Mã hóa**:
  ```
  Linear(3 → 16) → ReLU → 16-D
  ```

### 3.3 Tầng hợp nhất và phân loại (Fusion & Classifier)

- **Ghép đặc trưng**:
  ```
  [img_feat (256-D) ‖ tab_feat (16-D)] → 272-D
  ```
- **Bộ phân loại**:
  ```
  Linear(272 → 64) → ReLU → Dropout(0.4)
  → Linear(64 → 8)   ← logits
  ```
- **Kích hoạt đầu ra**: Sigmoid (áp dụng khi inference, không dùng khi training)

---

## 4. Sơ đồ luồng dữ liệu

```
Ảnh mắt trái (3×299×299)  ──┐
                              ├─ InceptionResNetV2 (shared) ─ GAP
Ảnh mắt phải (3×299×299)  ──┘   │                               │
                              Projector (128-D)         Projector (128-D)
                                   │                         │
                              f_left (128-D) ──────── f_right (128-D)
                                         └────┬────┘
                                        concat (256-D)
                                              │
[age_norm, gender_M, gender_F] ─ Linear(3→16) ─ ReLU ─ tab_feat (16-D)
                                              │
                                  concat (272-D)
                                              │
                              Linear(272→64) → ReLU → Dropout(0.4)
                                              │
                              Linear(64→8) → logits (8 nhãn)
```

---

## 5. Hàm mất mát và tối ưu hóa

| Thành phần | Cấu hình |
|---|---|
| Hàm mất mát | `BCEWithLogitsLoss` với `pos_weight` (xử lý mất cân bằng lớp) |
| `pos_weight` | Tính từ phân phối nhãn trong tập train (N: 2.07, D: 2.11, G: 15.25, C: 15.46, A: 20.33, H: 33.07, M: 19.11, O: 2.58) |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | patience=8 (theo val_loss) |

---

## 6. Tham số mô hình

| Chỉ số | Giá trị |
|---|---|
| Tổng số tham số | ~54.5 triệu |
| Tham số có thể huấn luyện | ~54.5 triệu (full fine-tuning) |
| Kích thước ảnh đầu vào | 299 × 299 |
| Batch size | 16 |
| Số epoch | 5 (+ early stopping) |

---

## 7. Điểm nổi bật về thiết kế

1. **Siamese backbone**: Chia sẻ trọng số giữa hai nhánh ảnh, đảm bảo tính đối xứng và giảm số tham số so với hai backbone độc lập.
2. **Multimodal fusion**: Kết hợp đặc trưng thị giác (ảnh đáy mắt) và đặc trưng phi thị giác (tuổi, giới tính) — phù hợp với thực tiễn lâm sàng.
3. **Xử lý mất cân bằng lớp**: Dùng `pos_weight` trong BCEWithLogitsLoss thay vì oversampling, tránh nhiễu dữ liệu.
4. **Phân loại đa nhãn**: Mỗi mắt có thể mắc đồng thời nhiều bệnh, sigmoid độc lập trên từng đầu ra phù hợp hơn softmax.

---

## 8. Đầu ra (Output)

- **Số lớp**: 8 nhãn (`N`, `D`, `G`, `C`, `A`, `H`, `M`, `O`)
- **Dạng**: Logit vector 8 chiều → sau sigmoid cho xác suất mỗi nhãn ∈ [0, 1]
- **Ngưỡng quyết định**: 0.5 (mặc định)
- **Độ đo đánh giá**: Macro AUC-ROC
