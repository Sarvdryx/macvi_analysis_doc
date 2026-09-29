# Phân tích Notebook `macvi-eda.ipynb` — MaCVi EURS

> Tài liệu này ghi lại toàn bộ phân tích về notebook EDA, tập trung vào 3 câu hỏi:
> 1. Dữ liệu video sẽ được huấn luyện như thế nào so với ảnh?
> 2. Giải thích cặn kẽ cell 4 và cell 10
> 3. Mục đích của các cell còn lại là gì?
>
> Nguồn: `C:\Users\ADMIN\Desktop\EURS\macvi-eda.ipynb` (22 cells, 10 code cells)

---

## Mục lục

- [0. Dataset này KHÔNG phải video](#0-dataset-này-không-phải-video)
- [1. Cell 4 — Lập chỉ mục dữ liệu](#1-cell-4--lập-chỉ-mục-dữ-liệu)
- [2. Cell 10 — Kiểm tra chất lượng dữ liệu](#2-cell-10--kiểm-tra-chất-lượng-dữ-liệu)
- [3. Các cell còn lại dùng để làm gì](#3-các-cell-còn-lại-dùng-để-làm-gì)
- [4. Tóm tắt những điều cần làm tiếp](#4-tóm-tắt-những-điều-cần-làm-tiếp)

---

## 0. Dataset này KHÔNG phải video

Có **một hiểu lầm cần sửa ngay**, vì nó ảnh hưởng mọi thứ sau đó.

### Bằng chứng từ chính output của cell 4 và cell 6

| Bằng chứng | Số liệu |
|---|---|
| File trong `degraded/` | `.png` — ảnh tĩnh, không có file video nào |
| Train | 4000 ảnh / **20 sequence** → 20 × 200 |
| Dev | 1200 ảnh / **6 sequence** → 6 × 200 |
| `param1` (từ cell 16) | min 1, max 10 → **10 giá trị** |
| `param2` (từ cell 16) | min 5, max 100, bước 5 → **20 giá trị** |
| 10 × 20 | **= 200 ảnh / sequence** ✓ khớp chính xác |

### Cấu trúc thật

```
fish1/  ← 1 ẢNH sạch duy nhất + 1 mask duy nhất
  fish1_1_5.png    (atten=1,  turb=5)     ← 200 biến thể
  fish1_1_10.png   (atten=1,  turb=10)
  ...                                      ← grid 10 × 20
  fish1_10_100.png (atten=10, turb=100)
```

**Cái bạn gọi là "video" thực chất là 20 ảnh gốc được render lại 200 lần với 200 tổ hợp tham số hình ảnh dưới nước khác nhau.**

### Hệ quả cực kỳ quan trọng

**Tập dữ liệu hiệu dụng chỉ có 20 scene, không phải 4000.** Train 4000 ảnh mà model vẫn chỉ "thấy" 20 nội dung khác nhau. Cell 5 gọi đây là "≈200 degraded frames derived from the same single clean image" — chính xác, nhưng diễn đạt nhẹ quá.

### Nếu sau này thật sự có video thì khác gì?

| | Ảnh độc lập | Video |
|---|---|---|
| Sample | 1 frame | 1 **clip** T frames |
| Model | 2D CNN / UNet | 3D CNN (C3D, R(2+1)D), Video Swin, TimeSformer |
| Loss | trên 1 frame | khớp frame + loss giữa các frame (temporal consistency) |
| Augment | flip, crop, color jitter | thêm: random frame sampling, temporal crop, motion augmentation |
| Batch | N frame | B × T, nhớ tăng rất nhiều |
| Chia tập | Chia theo ảnh | **Chia theo clip/cảnh quay**, không được cắt giữa 2 frame cùng cảnh |
| Nhãn mask | 1 frame | mask phải gán cho từng frame; class biến mất theo thời gian → cần xử lý mất nhãn |

Trong notebook này **không cần bất kỳ thứ nào ở cột phải**, vì dữ liệu là ảnh. Trọng tâm phải là: nhận dạng 20 scene, chống overfit lên 20 scene đó.

---

## 1. Cell 4 — Lập chỉ mục dữ liệu (trái tim của notebook)

Cell này làm 2 việc, và **cả 2 đều là nền tảng cho mọi cell sau**.

### 1.1. `parse_filename` — bóc tách tên file

```python
sharkFish_2_45.png
   ↓ stem = "sharkFish_2_45", parts = ["sharkFish", "2", "45"]
   ↓ seq    = "_".join(parts[:-2]) = "sharkFish"   ← tên scene
   ↓ param1 = parts[-2] = "2"  → int 2            ← mức hấp thụ quang học
   ↓ param2 = parts[-1] = "45" → int 45           ← mức đục nước
```

Ba nhánh xử lý: `len(parts) >= 3` (đủ seq + 2 tham số) → `== 2` (chỉ 1 tham số) → `else` (không tham số).

**Điểm hay:** `seq = "_".join(parts[:-2])` xử lý đúng cả tên scene có dấu `_`, ví dụ `shar_RayFish_3_75` → `shar_RayFish`, không bị cắt nhầm.

**Điểm cần lưu ý:** `int(x) if x.isdigit() else x` — nếu tham số không phải số thì giữ chuỗi, dẫn tới cột `param1` có kiểu object. Hiện tại dữ liệu sạch nên không sao, nhưng nên ép kiểu một lần ở cuối:

```python
df_all["param1"] = pd.to_numeric(df_all["param1"], errors="coerce")
```

**Vì sao quan trọng:** 2 tham số này là **nhãn miền của augmentation**. Khi train, bạn có thể lấy `param1, param2` của một ảnh rồi random dịch chuyển trong khoảng `[1,10] × [5,100]` để tạo ảnh nhiễu mới *giữ nguyên cảnh*. Đây chính là augmentation mạnh nhất mà dataset này cho bạn, vì nó tạo ra dữ liệu đa dạng thật sự từ 20 scene gốc.

### 1.2. Ghép cặp 3 thành phần (cascade 4 tầng)

Với mỗi ảnh degraded, code tìm ảnh GT sạch và mask theo thứ tự ưu tiên:

```
1.  gt_images/{tên file y hệt}          e.g. sharkFish_2_45.png
2.  gt_images/{sequence}.png            e.g. sharkFish.png          ← thường thắng
3.  gt_images/{folder_sequence}.png     e.g. sharkFish.png
4.  glob("**/{sequence}*.png") → lấy [0]                            ← dự phòng
```

Tương tự cho mask. Kết quả: **4000/4000 ghép được ảnh GT, 4000/4000 ghép được mask.**

**Điểm mạnh:** cascade thay vì giả định một quy ước đặt tên — chịu được nhiều cách tổ chức thư mục.

**Điểm yếu nghiêm trọng ở tầng 4:** `nested_gt[0]` lấy **phần tử đầu tiên** của glob, không có tiêu chí xác định đúng. Nếu 1 sequence có nhiều ảnh sạch, code sẽ **ghép nhầm im lặng** — không báo lỗi, chỉ cho kết quả model tệ. Cần thêm kiểm chứng:

```python
# Mỗi sequence chỉ được dùng đúng 1 ảnh GT / 1 mask?
print(df_train.groupby("folder_sequence")["gt_image_path"].nunique().value_counts())
print(df_train.groupby("folder_sequence")["gt_mask_path"].nunique().value_counts())
# Mọi dòng trong cùng sequence có cùng đường dẫn GT không?
print(df_train.groupby("folder_sequence")["gt_image_path"].nunique().max())  # phải == 1
```

### 1.3. Hai cột then chốt: `sequence` vs `folder_sequence`

- `sequence` — suy ra từ **tên file** (logic, có thể sai nếu đặt tên lệch chuẩn)
- `folder_sequence` — lấy từ `f.parent.name`, tức **cấu trúc thư mục thật**

Cell 5 yêu cầu GroupKFold theo `folder_sequence` — đây là lựa chọn đúng, vì nó là tín hiệu vật lý không thể sai. Hai cột này phải **luôn bằng nhau**; nếu không, đó là dấu hiệu đường dẫn bị lệch:

```python
mismatch = df_train[df_train["sequence"] != df_train["folder_sequence"]]
print(f"Sai khớp sequence vs folder_sequence: {len(mismatch)} dòng")
```

### 1.4. Dev split

Code chỉ index `degraded` của dev, để `gt_image_path = None` — đúng, vì dev được chấm ở server. Bảng ở cell 6 cho thấy **6 sequence dev (fish3, fish6, fish19, starfish1, sharRayFish, fish10) không xuất hiện trong train** → tách theo sequence ở cấp scene. Đây là tin tốt, nhưng nên `assert` để chắc chắn:

```python
overlap = set(df_train["folder_sequence"]) & set(df_dev["folder_sequence"])
assert len(overlap) == 0, f"LEAK: {overlap}"
```

Nếu có overlap, toàn bộ điểm leaderboard sẽ ảo.

### 1.5. Kết quả thực tế từ output

```
=== Metadata Indexing Summary ===
• Train Degraded Images : 4000 (with 4000 paired GT images & 4000 paired GT masks)
• Dev Degraded Images   : 1200
• Combined Dataset Rows : 5200
```

---

## 2. Cell 10 — Kiểm tra chất lượng dữ liệu (cell quan trọng nhất)

Đây là cell **"nhìn tận mắt"** — trước khi huấn luyện, bạn phải tin rằng nhãn đúng. Không có cell này, mọi thứ bạn train có thể sai mà không ai biết.

### 2.1. Ba hàm helper

**`load_rgb`** — `cv2.imread` trả BGR → `cvtColor(BGR2RGB)`. Cần thiết vì matplotlib hiển thị theo RGB. Đây là chỗ **hay sai nhất trong toàn bộ notebook**: bỏ `cvtColor` thì ảnh hiển thị bị đảo kênh xanh-đỏ, và bạn sẽ tưởng dữ liệu bị lỗi màu.

**`load_mask`** — `IMREAD_GRAYSCALE`. Đúng, vì mask là nhãn chỉ số nguyên 0–4, không phải ảnh màu.

**`colorize_5class_mask`** — vòng lặp gán màu:

```python
for class_idx in range(5):
    colorized[mask == class_idx] = CLASS_COLORS_RGB[class_idx]
```

Pixel có giá trị **> 4 sẽ thành đen** và bị bỏ qua âm thầm. Nên kiểm tra:

```python
u = np.unique(cv2.imread(sample_mask, cv2.IMREAD_GRAYSCALE))
print(u)   # nếu có giá trị 5, 255 → mask khác định dạng 5 lớp
```

### 2.2. `plot_5class_triplets` — 4 cột × N hàng

| Cột | Nội dung | Vai trò kiểm tra |
|---|---|---|
| **0** | Ảnh degraded + tên file + `p1, p2` | Xác nhận mức độ nhiễu đã render đúng không |
| **1** | Ảnh GT sạch | Xác nhận target có thật sự sạch, đúng cảnh, không bị nhầm |
| **2** | Mask 5 lớp, có liệt kê class hiện diện | Xác nhận mask **cùng kích thước** và **cùng vị trí** với ảnh |
| **3** | Overlay mask lên ảnh GT | Xác nhận mask **đúng nội dung**, không lệch vị trí |

Vài chi tiết kỹ thuật đáng chú ý trong code:

**a) Lọc cặp hợp lệ trước khi vẽ:**

```python
valid_rows = df_samples[df_samples["gt_image_path"].notna() & df_samples["gt_mask_path"].notna()]
```

Bỏ dòng này thì `load_rgb(None)` trả `None` và ô trống trống — dễ tưởng dữ liệu hỏng.

**b) Xử lý trường hợp 1 hàng:**

```python
if len(selected) == 1:
    axes = np.expand_dims(axes, axis=0)
```

Không có dòng này, `axes` thành vector 1 chiều → `axes[i, 0]` báo lỗi.

**c) Resize mask bằng `INTER_NEAREST`:**

```python
gt_mask_res = cv2.resize(gt_mask, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)
```

Đúng — chỉ `INTER_NEAREST` giữ được nhãn nguyên vẹn 0–4. Dùng `INTER_LINEAR` sẽ sinh giá trị 2.7 → nhãn hỏng. Đây là nguyên tắc bạn phải giữ trong cả pipeline train.

**d) Chỉ blend vùng foreground:**

```python
fg_bool = gt_mask_res > 0
overlay[fg_bool] = cv2.addWeighted(gt_img[fg_bool], 0.45, color_mask[fg_bool], 0.55, 0)
```

Background giữ nguyên ảnh gốc để bạn thấy bối cảnh, còn vùng có vật thể mới tô màu.

**e) Tự động liệt kê class có mặt thật:**

```python
present_classes = [CLASS_NAMES[c] for c in np.unique(gt_mask) if c < 5]
```

Cái này rất hữu ích: cho biết ảnh đó chứa những lớp nào, phục vụ phân tích mất cân bằng.

### 2.3. Ba lỗi/điểm yếu của cell 10 cần sửa

**Lỗi 1 — `random_state=42` cố định.** Cả cell 4, 8, 14, 16, 20 đều dùng `random_state=42`, nên **4 mẫu vẽ ra luôn y hệt nhau mọi lần chạy**. Bạn sẽ không bao giờ thấy được cảnh giống `turtles1` hay `seahorse1` mà không phải sửa code. Nên lọc theo sequence để xem đủ 5 loại:

```python
plot_5class_triplets(df_train[df_train["folder_sequence"].isin(
    ["fish1", "sharkFish", "ray1", "turtles1", "seahorse1"])], num_samples=5)
```

**Lỗi 2 — Thiếu ảnh kiểm chứng (sanity check) bằng đại số.** Nhìn bằng mắt dễ bỏ sót lệch 5 pixel. Thêm:

```python
# Ảnh GT và ảnh degraded có cùng kích thước không?
a = cv2.imread(row["path"]).shape
b = cv2.imread(row["gt_image_path"]).shape
# Ảnh GT có thật sự sạch hơn ảnh degraded không? (Laplacian variance)
print(compute_laplacian_var(load_rgb(row["gt_image_path"])),
      compute_laplacian_var(load_rgb(row["path"])))
```

**Lỗi 3 — Overlay 55% màu là quá đậm.** Bạn không nhìn rõ ranh giới vật thể. Giảm xuống `0.6 ảnh / 0.4 màu`, và thêm một panel crop phóng to 256×256 ở vùng biên để đánh giá chất lượng annotation.

---

## 3. Các cell còn lại dùng để làm gì

### Cell 2 — Thiết lập môi trường

Nạp thư viện, chọn gốc dữ liệu (Kaggle nếu có, không thì local), in ra 4 đường dẫn để xác nhận. Định nghĩa **bảng màu 5 lớp dùng xuyên suốt notebook** — cần giữ nhất quán giữa EDA, loss visualization và bài báo.

### Cell 6 — Phân bố sequence & cảnh báo rò rỉ

`extract_category` gán nhãn sinh học từ tên (`shark`/`ray` → Elasmobranch, `fish` → Bony Fish, `star` → Starfish...). Có chú ý thứ tự `elif`: `"fish" in "starfish"` đúng, nhưng điều kiện `and "star" not in seq_lower` phải đứng trước — nếu đảo thứ tự, mọi starfish sẽ bị gán nhầm là Bony Fish.

Kết quả: **20 sequence train, 6 sequence dev, mỗi sequence 200 ảnh.** Đây là cell chứng minh tính hiệu dụng của dataset chỉ bằng 20, và là nơi phát ra cảnh báo leakage — mọi frame của `fish1` đều dùng chung một ảnh GT, nên random split sẽ đưa 200 ảnh của cùng một cảnh vào cả train lẫn val → mô hình học thuộc, điểm ảo, fail trên leaderboard. Bắt buộc GroupKFold theo `folder_sequence`.

### Cell 8 — Kích thước & độ phân giải

Đo trên 200 ảnh mẫu. Kết quả:

| Chỉ số | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| width | 4281.9 | 1228.7 | 2268 | 4000 | 4000 | 4032 | 9710 |
| height | 3720.5 | 1260.9 | 3000 | 3000 | 3024 | 4032 | 8000 |
| aspect_ratio | 1.2086 | 0.3143 | 0.562 | 0.750 | 1.333 | 1.333 | 1.978 |
| mean_val | 39.70 | 29.44 | **0.0** | 13.68 | 38.75 | 57.75 | 135.35 |

Rộng trung vị **4000px**, cao trung vị **3024px**, tỉ lệ khung 0.562–1.978 (có cả ảnh dọc lẫn ngang), cường độ trung bình dao động **0 → 135** (có ảnh gần như đen tuyệt đối).

**Quyết định thiết kế:** ảnh gốc to 12 MP, nhưng Track 2 giới hạn 12 GFLOPs và nền tham chiếu chạy 384×768. Vậy phải resize về cỡ đó khi train. Nhưng vì tỉ lệ khung rất khác nhau, **không thể resize cứng 384×768 toàn bộ** — sẽ méo. Nên dùng **random crop 384×768** (giữ tỉ lệ) khi train, và resize giữ tỉ lệ khi suy luận. Sai lầm phổ biến nhất ở đây là bỏ qua bước này.

### Cell 12 — Quét phổ suy giảm

Giữ cố định 1 sequence, sắp xếp theo `param1, param2`, hiển thị 10 frame đầu. Mục đích: xác nhận trực quan rằng `param1` tăng → mất kênh đỏ dần (hấp thụ), `param2` tăng → đục và nhòe tăng (khuếch tán ngược). Cơ sở để thiết kế augmentation theo tham số. Lưu ý: ý nghĩa chính xác của `param1`/`param2` là **giả định của notebook**, phải đối chiếu `OVERVIEW.md` xác nhận.

### Cell 14 — Hồ sơ màu RGB / LAB / HSV

Cộng dồn histogram 256 bin trên 50 ảnh. Mục đích: chứng minh vật lý hóa nước ngầm. Theo mô hình

$$I_c = J_c e^{-\beta_c d} + B_c(1 - e^{-\beta_c d}), \quad c \in \{R, G, B\}$$

với $\beta_R \gg \beta_G > \beta_B$: đỏ bị hấp thụ gần như hết, nền lục lam do backscatter. Hệ quả trực tiếp cho loss: mô hình phải **tái tạo kênh đỏ gần như từ không** → đây là lý do `L1` thuần dễ cho ra ảnh xanh/xám, và vì sao cell 21 đề xuất thêm loss tần số.

### Cell 16 — PSNR / SSIM / Sharpness ratio

Đo trên 100 cặp. **Đây là con số đáng sốc nhất của notebook:**

| Chỉ số | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| PSNR (dB) | 12.6453 | 4.5015 | 4.47 | 9.10 | 11.725 | 15.42 | 27.00 |
| `psnr_norm` | 0.0485 | 0.1101 | 0.0 | 0.0 | **0.0** | 0.021 | 0.600 |
| SSIM | 0.4266 | 0.2583 | 0.0084 | 0.2025 | 0.4274 | 0.6548 | 0.9127 |
| `sharpness_ratio` | 0.0207 | 0.0235 | 0.0 | 0.0023 | **0.0113** | 0.0328 | 0.0822 |

Nghĩa là:

- **Hơn 75% mẫu có `psnr_norm = 0`** (vì PSNR < 15 dB bị clip về 0). Trong composite metric, PSNR đóng 0.3 × 0.45 = **13.5% trọng số** — nếu mô hình không đạt 15 dB, mất sạch phần này.
- **`sharpness_ratio` trung vị chỉ 0.011** — ảnh đầu vào giữ được **1,1% năng lượng tần số cao** của ảnh sạch. Nghĩa là bản chất input gần như là một ảnh làm mờ. Model phải **sinh chế texture**, không phải tìm lại texture.

Đây chính là bằng chứng định lượng cho cell 18, và là lý do `L_Laplacian` trong loss ở cell 21 là bắt buộc chứ không phải tuỳ chọn.

### Cell 18 — Phổ Fourier

2D FFT của ảnh xám, hiển thị dạng `20·log|magnitude|`. So sánh phổ ảnh nhiễu với ảnh sạch: phổ ảnh nhiễu bị **dập tần số cao** thành một vùng tròn tối tập trung. Cơ sở vật lý cho `Laplacian` loss và cho kiến trúc phải dùng skip connection mạnh (giữ được tần số cao qua đường tắt).

### Cell 20 — Phân bố pixel 5 lớp & heatmap không gian

Tính tỉ lệ pixel từng lớp trên 100 mask, vẽ thanh + heatmap mật độ 5 lớp ở kích thước chuẩn 256×256.

Quan trọng vì $Q_{\text{seg}} = \text{F1} \times \text{mIoU} \times \text{coverage\_penalty}$ chiếm **60%** điểm, và `coverage_penalty` phạt nặng khi bỏ sót lớp hiếm. Nếu `Background` chiếm 90% pixel thì mô hình dự đoán toàn background vẫn đạt 90% accuracy — nhưng `mIoU` và `coverage_penalty` sẽ kéo $Q_{\text{seg}}$ về gần 0. Đây là lý do cell 21 đề xuất `FocalTversky` bên cạnh CrossEntropy. Heatmap không gian còn cho biết vị trí đặc trưng của từng lớp (cá thường giữa khung, coral ở đáy) — có thể dùng cho loss có trọng số vị trí.

### Cell 21 — Chiến lược mô hình

Kiến trúc 2 đầu dùng chung encoder, ràng buộc Track 2 (≤ 12 GFLOPs, ≤ 1.0M params, ONNX tĩnh 384×768), và công thức loss đa nhiệm vụ. Đây là cell kết luận, không phải EDA.

```
                               ┌─► Restoration Decoder ──► restored_image (1, 3, H, W) in [0, 1]
Input (1, 3, H, W) ──► Shared ─┤
                       Encoder └─► Segmentation Decoder ─► seg_logits (1, 5, H, W)
```

$$\mathcal{L}_{\text{total}} = \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{rest}} \mathcal{L}_{\text{rest}}$$

- $\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{CrossEntropy}} + \mathcal{L}_{\text{FocalTversky}}$ (xử lý mất cân bằng background vs elasmobranch/coral)
- $\mathcal{L}_{\text{rest}} = \alpha \mathcal{L}_{1} + \beta (1 - \text{SSIM}) + \gamma \mathcal{L}_{\text{Laplacian/Sharpness}}$

---

## 4. Tóm tắt những điều cần làm tiếp

```python
# 1) Chứng minh effective dataset chỉ có 20 scene
print(df_train.groupby("folder_sequence")["gt_image_path"].nunique().max())   # phải = 1

# 2) Chắc chắn không rò rỉ train/dev
assert set(df_train["folder_sequence"]) & set(df_dev["folder_sequence"]) == set()

# 3) GroupKFold thật, không dùng train_test_split
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for tr, va in gkf.split(df_train, groups=df_train["folder_sequence"]):
    assert not (set(df_train.iloc[tr]["folder_sequence"]) &
                set(df_train.iloc[va]["folder_sequence"]))

# 4) Augmentation mạnh nhất: dịch chuyển (param1, param2) giữ nguyên scene
#    biến [1,10] × [5,100] → sinh ảnh nhiễu mới từ 20 ảnh gốc
```

### Ba kết luận cốt lõi

1. Dataset là **ảnh**, không phải video; và chỉ có **20 scene thật** — mọi chiến lược phải xoay quanh chống overfit 20 scene.
2. Cell 4 xây nền tảng (ghép cặp + tham số), cell 10 xác minh chất lượng — hai cell này phải **đúng** thì phần còn lại mới có ý nghĩa.
3. Đầu vào cực suy biến (`psnr_norm` trung vị = 0, `sharpness_ratio` trung vị = 0.011) → đây là bài toán **sinh texture hơn là sửa màu**, loss Laplacian tần số không phải tuỳ chọn.

---

## Phụ lục — Bảng tra nhanh

| Cell | Markdown | Mục đích | Số liệu quan trọng |
|---|---|---|---|
| 0 | Title | Tổng quan bài toán + công thức metric | Composite = 0.6·Q_seg + 0.3·(...) + 0.1·NRquality |
| 1 | Mục 1 | Tiêu đề | — |
| 2 | — | Setup, đường dẫn, bảng màu 5 lớp | Kaggle root: `/kaggle/input/datasets/thnhtt/macvi-eurs` |
| 3 | Mục 2 | Tiêu đề | — |
| **4** | — | **Index dataset, ghép cặp GT/mask** | **4000 train / 1200 dev / 5200 tổng** |
| 5 | Mục 3 | Cảnh báo leakage | — |
| 6 | — | Phân bố sequence, đếm theo category | 20 seq train × 200, 6 seq dev × 200 |
| 7 | Mục 4 | Tiêu đề | — |
| 8 | — | Kích thước, aspect ratio, cường độ | 4000×3000, AR 0.562–1.978, mean_val 0–135 |
| 9 | Mục 5 | Tiêu đề | — |
| **10** | — | **Vẽ bộ 4 ảnh: input / GT / mask / overlay** | — |
| 11 | Mục 6 | Tiêu đề | — |
| 12 | — | Quét phổ suy giảm theo param | — |
| 13 | Mục 7 | Tiêu đề | — |
| 14 | — | Histogram RGB / LAB / HSV | — |
| 15 | Mục 8 | Tiêu đề | — |
| 16 | — | PSNR / SSIM / sharpness ratio | PSNR 11.7 dB, norm 0.0, sharp 0.011 |
| 17 | Mục 9 | Tiêu đề | — |
| 18 | — | 2D FFT phổ tần số | — |
| 19 | Mục 10 | Tiêu đề | — |
| 20 | — | Phân bố pixel 5 lớp + heatmap | Q_seg = 60% trọng số |
| 21 | Mục 11 | Kiến trúc 2 đầu + loss + ONNX | ≤12 GFLOPs, ≤1.0M params |

### Bảng màu 5 lớp

| ID | Tên | RGB |
|---|---|---|
| 0 | Background | `[15, 23, 42]` — Dark Slate |
| 1 | Coral | `[249, 115, 22]` — Orange |
| 2 | Fish | `[234, 179, 8]` — Gold |
| 3 | Elasmobranch | `[6, 182, 212]` — Cyan |
| 4 | Other | `[236, 72, 153]` — Magenta |

### Công thức metric chính thức

$$\text{Composite} = 0.6 \times Q_{\text{seg}} + 0.3 \times (0.45 \times \text{PSNR}_{\text{norm}} + 0.45 \times \text{SSIM} + 0.10 \times \text{sharpness\_ratio}) + 0.1 \times \text{NRquality}$$

$$\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15}{35 - 15}, 0, 1\right)$$

$$\text{sharpness\_ratio} = \frac{\text{Var}(\nabla^2 I_{\text{degraded}})}{\text{Var}(\nabla^2 I_{\text{clean\_gt}})}$$
