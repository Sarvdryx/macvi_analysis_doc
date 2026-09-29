# Báo cáo lỗi tiềm ẩn — `macvi-eda.ipynb`

> Tài liệu tổng hợp toàn bộ lỗi và rủi ro phát hiện được khi phân tích notebook EDA của MaCVi-EURS.
>
> **Phạm vi:** lỗi *trong notebook*, lỗi *thiết kế pipeline*, và rủi ro *phương pháp luận*.
> Lỗi * cú pháp Python / code không chạy: **không có**. Tất cả code đều chạy được.
>
> **Nguồn:** `macvi-eda.ipynb` (22 cells) + output thực tế + kết quả kiểm chứng đã chạy.

---

## Mục lục

| Mức độ | Lỗi | Cell |
| :--- | :--- | :--- |
| 🔴 **Nghiêm trọng** | [E-01](#e-01--cell-20-lấy-mẫu-sai-vì-chỉ-có-20-mask-duy-nhất) | 20 |
| 🔴 **Nghiêm trọng** | [E-02](#e-02--cell-20-có-thể-bỏ-sót-hẳn-một-lớp-khỏi-biểu-đồ) | 20 |
| 🔴 **Nghiêm trọng** | [E-03](#e-03--df_all-gộp-cả-dev--rủi-ro-rò-rỉ-thống-kê) | 4 |
| 🔴 **Nghiêm trọng** | [E-04](#e-04--thiếu-khẳng-định-không-rò-rỉ-train--dev) | 4 |
| 🟠 **Cao** | [E-05](#e-05--cell-16-đo-metric-trên-100-cặp-nhưng-chỉ-có-20-ảnh-gốc) | 16 |
| 🟠 **Cao** | [E-06](#e-06--tầng-4-cascade-ghép-cặp-đường-dẫn-gt-có-thể-ghép-nhầm-im-lặng) | 4 |
| 🟠 **Cao** | [E-07](#e-07--random_state42-cố-định--4-mẫu-luôn-giống-nhau) | 4,8,10,14,16,20 |
| 🟠 **Cao** | [E-08](#e-08--cell-10-kiểm-tra-mắt-thuần-quy-nhưng-không-có-sanity-check-đại-số) | 10 |
| 🟡 **Trung bình** | [E-09](#e-09--parse_filename-nhánh-lenparts--2-không-thể-phân-biệt) | 4 |
| 🟡 **Trung bình** | [E-10](#e-10--parse_filename-có-thể-trả-về-cột-kiểu-object-gây-lỗi-khi-sort) | 4 |
| 🟡 **Trung bình** | [E-11](#e-11--df_all-không-được-copy--side-effect-khi-không-có-dev) | 4 |
| 🟡 **Trung bình** | [E-12](#e-12--df_train-thiếu-cột-category--vi-chỉ-gán-trên-df_all) | 4, 6 |
| 🟡 **Trung bình** | [E-13](#e-13--cell-8--kết-luận-resize-cứng-384×768-sẽ-làm-méo-ảnh) | 8 |
| 🟡 **Trung bình** | [E-14](#e-14--thiếu-kiểm-chứng-mask-chỉ-chứa-giá-trị-04) | 10, 20 |
| 🟡 **Trung bình** | [E-15](#e-15--giả-thuyết-ý-nghĩa-param1--param2-chưa-được-xác-minh) | 12 |
| 🟡 **Trung bình** | [E-16](#e-16--cell-16-đo-chỉ-hành-động--không-có-baseline-mô-hình-học) | 16 |
| 🟢 **Thấp** | [E-17](#e-17--cell-6--điều-kiện-extract_category-dễ-đảo-ngược-thứ-tự) | 6 |
| 🟢 **Thấp** | [E-18](#e-18--cell-8--sns.boxplot-palette-sẽ-bị-lỗi-ở-seaborn-014) | 8 |
| 🟢 **Thấp** | [E-19](#e-19--cell-16--thiếu-handle-vi-gọi-metrics-ở-cấp-module) | 16 |
| 🟢 **Thấp** | [E-20](#e-20--cell-10--overlay-55-màu-che-khuất-biên-vật-thể) | 10 |
| 🟢 **Thấp** | [E-21](#e-21--cell-12--rò-rỉ-hành-vi-để-ảnh-tạm-vào-dataframe-chung) | 12 |

---

## Tóm tắt

**22 lỗi** phân bố theo mức độ:

```
🔴 Nghiêm trọng   4   (E-01 → E-04)
🟠 Cao           4   (E-05 → E-08)
🟡 Trung bình    8   (E-09 → E-16)
🟢 Thấp          6   (E-17 → E-22)
```

### Nhóm theo bản chất

| Bản chất | Lỗi |
| :--- | :--- |
| **Thiếu kiểm chứng / assertion** | E-02, E-04, E-06, E-14, E-21 |
| **Chống rò rí dữ liệu** | E-03, E-05, E-12 |
| **Lỗi thống kê do giả định sai** | E-01, E-02, E-13, E-15 |
| **Đánh đổi độ ổn định ↔ tính đại diện** | E-07 |
| **Thiếu biến cục bộ** | E-11, E-19 |
| **Chất lượng trực quan / API** | E-17, E-18, E-20 |

### Sai lầm nền tảng nhất

> **Số ảnh đầu vào là 4000, nhưng số nội dung thật chỉ là 20.**
> Hầu hết lỗi nghiêm trọng bên dưới đều là hệ quả trực tiếp của điều này.

---

## 🔴 E-01 — Cell 20: Lấy mẫu sai vì chỉ có 20 mask duy nhất

**Mức độ:** Nghiêm trọng · **Cell:** 20

### Vị trí

```python
def analyze_5class_masks(df_train_sub, target_shape=(256, 256), max_samples=80):
    valid_masks = df_train_sub[df_train_sub["gt_mask_path"].notna()].copy()
    if len(valid_masks) == 0:
        print("No GT mask paths found for analysis.")
        return

    samples = valid_masks.sample(min(len(valid_masks), max_samples), random_state=42)
    #              ^^^^^ lấy 100 DÒNG, nhưng chỉ có 20 mask DUY NHẤT
```

### Vấn đề

Đã kiểm chứng: `df_train["gt_mask_path"].nunique() == 20`.

Lệnh `sample(100)` trên 4000 dòng sẽ lấy 100 dòng — nhưng 100 dòng đó chỉ là **20 mask bị lặp lại khoảng 5 lần**. Vòng lặp sau đó đọc cùng một file PNG 5 lần:

```python
for _, row in tqdm(samples.iterrows(), ...):
    mask = load_mask(row["gt_mask_path"])   # đọc lại file đã đọc ở vòng trước
    for c in range(5):
        c_mask = (mask == c).astype(np.float32)
        class_pixel_counts[c] += c_mask.sum()      # nhân đôi số liệu
        class_heatmaps[c] += resized_c
    count += 1                                    # count = 100, thực chất = 20
```

### Hậu quả

1. **Số liệu chỉ là xấp xỉ, không chính xác.** Biểu đồ thanh "Overall Pixel Percentage by Semantic Class" được tính từ 20 mask bị gán trọng số ngẫu nhiên theo số lần xuất hiện, không phải tỷ lệ thật của dataset.
2. **Lãng phí ~5× thời gian tính toán.** Chỉ 20 file nhưng phải đọc 100 lần.
3. **Sai lệch ngẫu nhiên giữa các lần chạy.** Đổi `random_state` là biểu đồ nhảy số.

### Cách sửa

```python
samples = valid_masks.drop_duplicates(subset="gt_mask_path")
# → còn đúng 20 dòng, đúng 20 mask, kết quả CHÍNH XÁC TUYỆT ĐỐI
```

Và bỏ tham số `max_samples` vì không còn ý nghĩa — chỉ có 20 mask, cứ tính hết.

### Xác minh lại

```python
# Cách sửa cho kết quả chuẩn
tot = np.zeros(5, dtype=np.int64)
for p in df_train["gt_mask_path"].unique():
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    for c in range(5):
        tot[c] += int((m == c).sum())
for c, n in enumerate(CLASS_NAMES):
    print(f"  {c}: {n:13} {100*tot[c]/tot.sum():6.2f}%")
```

---

## 🔴 E-02 — Cell 20: Có thể bỏ sót hẳn một lớp khỏi biểu đồ

**Mức độ:** Nghiêm trọng · **Cell:** 20

### Vấn đề

Kiểm chứng trên mask đầu tiên cho ra:

```
Giá trị trong mask: [0 1 2 3]      ← không có lớp 4 (Other)
```

Nếu lớp 4 chỉ tồn tại ở `starfish2` / `seahorse1` / `turtles1` (2–3 trong 20 mask), thì với `random_state=42` cố định, xác suất **không lấy mẫu** tới những mask đó là có thật. Khi đó:

- Biểu đồ thanh hiển thị 4 cột thay vì 5 — **người đọc kết luận "dataset không có lớp 4"**
- Heatmap `axes[4]` gần như đen, dễ bị bỏ qua
- Nếu dùng luôn heatmap này để thiết kế loss có trọng số vị trí → mất hoàn toàn thông tin về lớp `Other`

### Tại sao nghiêm trọng

Metric chính thức là:

$$Q_{\text{seg}} = \text{F1} \times \text{mIoU} \times \text{coverage\_penalty}$$

`coverage_penalty` phạt nặng khi model bỏ sót lớp hiếm. Nếu lớp `Other` vốn đã hiếm (vài scene), thì bất kỳ sai sót nào trong EDA này sẽ thành sai lầm thiết kế loss, rồi thành điểm 0 trên leaderboard.

### Cách sửa

Bắt buộc kiểm tra toàn cục **trước khi vẽ bất kỳ biểu đồ nào**:

```python
all_vals = set()
for p in df_train["gt_mask_path"].unique():
    all_vals |= set(np.unique(cv2.imread(p, cv2.IMREAD_GRAYSCALE)).tolist())
print("Tất cả giá trị có trong 20 mask train:", sorted(all_vals))

# Lớp 4 xuất hiện ở sequence nào?
for p in df_train["gt_mask_path"].unique():
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    n4 = int((m == 4).sum())
    print(f"  {Path(p).name:20} lớp4 = {n4:>10,} px  ({100*n4/m.size:5.2f}%)")
```

Sau khi sửa E-01 (dùng `drop_duplicates`), cả 20 mask đều được đưa vào phân tích → lớp 4 không thể bị bỏ sót.

---

## 🔴 E-03 — `df_all` gộp cả dev → rủi ro rò rỉ thống kê

**Mức độ:** Nghiêm trọng · **Cell:** 4, dùng lại ở 6, 8, 14

### Vị trí

```python
df_all = pd.concat([df_train, df_dev], ignore_index=True) if len(df_dev) > 0 else df_train
#                         ^^^^^^^^^^    ^^^^^^
#                     5200 dòng = 4000 train + 1200 dev
```

### Cách dùng hiện tại

| Cell | Dùng `df_all` cho | Đánh giá |
| :--- | :--- | :--- |
| 6 | Thống kê số lượng, phân bố category | ✅ hợp lệ |
| 8 | Thống kê kích thước ảnh | ✅ hợp lệ |
| 14 | Histogram màu | ⚠️ chấp nhận được |

### Vấn đề

`df_all` chứa **1200 ảnh dev**. Dev là tập mà bạn **không được phép dùng để huấn luyện hay chọn siêu tham số**.

Nếu bạn dùng `df_all` cho các mục đích sau thì đã vi phạm nguyên tắc:

| Mục đích sai | Hậu quả |
| :--- | :--- |
| Tính `PIXEL_MEAN` / `PIXEL_STD` để normalize | Rò rỉ thống kê phân phối dev |
| Chọn ngưỡng soát tệp | Tối ưu trên dev → điểm không khớp test |
| Chọn augmentation dựa trên phân phối màu | Tương tự rò rỉ |
| Tính class weight cho CrossEntropy | Cân bằng theo cả dev |

Đây là loại rò rỉ dễ xảy ra nhất vì `df_all` là biến "tiện tay" được tạo ra ở cell 4 và rất dễ bị dùng nhầm.

### Cách sửa

```python
# Mô tả EDA → df_all OK
# Mọi thứ liên quan tới huấn luyện → df_train
PIXEL_MEAN = df_train_mean_value      # chỉ train
PIXEL_STD  = df_train_std_value
CLASS_WEIGHT = compute_from(df_train) # chỉ train

# An toàn hơn: đặt tên rõ ràng
df_for_stats = df_train              # dùng cho normalize / class weight
```

Nên thêm một dòng cảnh báo ngay dưới chỗ tạo `df_all`:

```python
print("[WARNING] df_all chỉ dùng để EDA mô tả. KHÔNG dùng để normalize/class-weight.")
```

---

## 🔴 E-04 — Thiếu khẳng định không rò rỉ train / dev

**Mức độ:** Nghiêm trọng · **Cell:** 4 (thiếu), liên quan 5, 6

### Vấn đề

Cell 4 index dev mà **không hề kiểm tra** dev có trùng sequence với train hay không. Bảng ở cell 6 *mới hiển thị* ra 6 sequence dev không nằm trong train — nhưng đó chỉ là quan sát bằng mắt, không có `assert`.

Nếu tổ chức cuộc thi thay đổi cách chia tập (rất dễ xảy ra ở các đợt cập nhật dataset), hoặc nếu bạn chạy lại notebook trên phiên bản dataset khác, một sequence dev có thể trùng train mà không ai phát hiện.

### Hậu quả

Nếu có overlap:
- 200 ảnh dev đó **đã có ảnh sạch trong tập train**
- Model đã học thuộc chính xác cảnh đó
- Điểm leaderboard **hoàn toàn ảo**
- Toàn bộ kết luận trong notebook mất giá trị

### Cách sửa

Chèn ngay sau `df_all`:

```python
# ===== KIỂM TRA RÒ RỈ: bắt buộc =====
overlap = set(df_train["folder_sequence"]) & set(df_dev["folder_sequence"])
assert len(overlap) == 0, f"LEAK — dev trùng train: {overlap}"
print(f"[OK] Không rò rỉ: train {df_train['folder_sequence'].nunique()} seq, "
      f"dev {df_dev['folder_sequence'].nunique()} seq, chồng lấn 0")

# ===== KIỂM TRA CẤU TRÚC: cũng nên assert =====
assert df_train.groupby("folder_sequence")["gt_image_path"].nunique().max() == 1, \
    "Một sequence có nhiều ảnh GT → ghép cặp chưa đúng"
assert (df_train["sequence"] == df_train["folder_sequence"]).all(), \
    "Tên file và tên thư mục không khớp"
print(f"[OK] Cấu trúc hợp lệ: {df_train['gt_image_path'].nunique()} ảnh GT cho "
      f"{df_train['folder_sequence'].nunique()} sequence")
```

Kết quả kiểm chứng hiện tại: 20 ảnh GT / 20 sequence — đạt.

---

## 🟠 E-05 — Cell 16: Đo metric trên 100 cặp nhưng chỉ có 20 ảnh gốc

**Mức độ:** Cao · **Cell:** 16

### Vị trí

```python
def compute_image_metrics(df_train_pairs, max_pairs=80):
    valid_pairs = df_train_pairs[df_train_pairs["gt_image_path"].notna()].copy()
    samples = valid_pairs.sample(min(len(valid_pairs), max_pairs), random_state=42)
```

### Vấn đề

Lệnh gọi: `df_metrics = compute_image_metrics(df_train, max_pairs=100)` → lấy 100 dòng.

Nhưng đã xác nhận chỉ có **20 ảnh sạch duy nhất**. Nên 100 cặp = **20 cảnh × 5 mức nhiễu khác nhau**.

### Hệ quả: hiểu sai ý nghĩa số liệu

Bảng kết quả trong notebook:

```
           psnr_db   psnr_norm     ssim  sharpness_ratio
count         100.000000    0.0000  0.426607         0.020702
mean           12.645300    0.0485  0.258306         0.023492
```

Cột `count = 100` gợi ý "100 mẫu độc lập" — **không phải**. Có chỉ 20 mức độ khó thực sự.

| Tình huống | Hệ quả lên độ tin cậy |
| :--- | :--- |
| 20 cảnh × 5 nhiễu | Độ rộng do **mức nhiễu**, không do nội dung |
| 100 cảnh ngẫu nhiên | Độ rộng phản ánh **cả** nội dung lẫn nhiễu |

Nếu bạn báo cáo "trên 100 mẫu, PSNR trung vị 11.7 dB" thì người đọc hiểu sai — độ lệch thực tế giữa các cảnh là rất lớn nhưng bị giấu.

### Cách sửa

```python
# Báo cáo rõ cấu trúc mẫu
df_metrics = compute_image_metrics(df_train, max_pairs=100)
print(f"[INFO] {len(df_metrics)} cặp, nhưng chỉ "
      f"{df_metrics['sequence'].nunique()} ảnh sạch duy nhất "
      f"→ std phản ánh biến thiên theo mức nhiễu, không phải theo nội dung")
```

Và khi muốn so sánh mô hình, nên **cố định tập 20 cảnh × đủ 20 mức nhiễu** thay vì lấy mẫu ngẫu nhiên:

```python
# Đánh giá xác định (deterministic), không ngẫu nhiên
eval_set = (df_train.drop_duplicates(subset="gt_image_path")
                      .groupby("gt_image_path")
                      .apply(lambda g: g.nsmallest(5, "param2")))  # 5 mức nhiễu cố định
```

---

## 🟠 E-06 — Tầng 4 cascade ghép cặp GT: có thể ghép nhầm im lặng

**Mức độ:** Cao · **Cell:** 4

### Vị trí

```python
gt_img_exact      = TRAIN_GT_IMG_DIR / f.name                    # tầng 1
gt_img_seq        = TRAIN_GT_IMG_DIR / f"{rec['sequence']}.png"  # tầng 2
gt_img_folder_seq = TRAIN_GT_IMG_DIR / f"{f.parent.name}.png"    # tầng 3

if   gt_img_exact.exists():      rec["gt_image_path"] = str(gt_img_exact)
elif gt_img_seq.exists():        rec["gt_image_path"] = str(gt_img_seq)
elif gt_img_folder_seq.exists(): rec["gt_image_path"] = str(gt_img_folder_seq)
else:
    nested_gt = list(TRAIN_GT_IMG_DIR.glob(f"**/{rec['sequence']}*.png"))
    rec["gt_image_path"] = str(nested_gt[0]) if len(nested_gt) > 0 else None
    #                                     ^^^^^^^^^^^^ đây
```

### Vấn đề

`nested_gt[0]` là **phần tử đầu tiên** của kết quả glob, không có bất kỳ tiêu chí nào để chọn đúng. Thứ tự do hệ điều hành quyết định, không đảm bảo ổn định giữa các lần chạy.

### Kịch bản hỏng

Nếu `gt_images/` chứa nhiều hơn một ảnh cho một sequence:

```
gt_images/
  ├── sharkFish.png
  ├── sharkFish_clean.png        ← cũng khớp glob "**/sharkFish*.png"
  └── sharkFish_2_45.png
```

thì tầng 4 có thể chọn `sharkFish_clean.png` thay vì `sharkFish.png`.

### Hậu quả

**Ghép nhầm mà không có bất kỳ tín hiệu nào:**
- Không exception
- Không cảnh báo
- `gt_image_path.notna()` vẫn = 4000/4000
- Chỉ lộ ra khi điểm số thấp bất thường — rất khó truy nguyên

Đây là loại lỗi nguy hiểm nhất vì nó **im lặng**.

### Cách sửa

```python
else:
    nested_gt = list(TRAIN_GT_IMG_DIR.glob(f"**/{rec['sequence']}*.png"))
    if len(nested_gt) == 1:
        rec["gt_image_path"] = str(nested_gt[0])
    elif len(nested_gt) == 0:
        rec["gt_image_path"] = None
    else:
        rec["gt_image_path"] = None   # hoặc báo lỗi
        warnings.warn(f"GT mơ hồ cho {f.name}: {len(nested_gt)} ứng viên "
                      f"{[p.name for p in nested_gt]} — cần kiểm tra thủ công")
```

### Kiểm chứng

```python
print("Số ảnh GT duy nhất:", df_train["gt_image_path"].nunique())          # → 20
print("Số mask duy nhất:", df_train["gt_mask_path"].nunique())             # → 20
print("Max GT per sequence:",
      df_train.groupby("folder_sequence")["gt_image_path"].nunique().max()) # → 1
```

Kết quả hiện tại: **20 / 20 / 1** → tầng 4 chưa từng được dùng. Nhưng vẫn nên thêm cảnh báo, vì chỉ cần dataset đổi là hỏng.

---

## 🟠 E-07 — `random_state=42` cố định: 4 mẫu luôn giống nhau

**Mức độ:** Cao · **Cell:** 4 (không), 8, 10, 14, 16, 20

### Vị trí

| Cell | Dòng | Ảnh hưởng |
| :--- | :--- | :--- |
| 8 | `df_sample.sample(..., random_state=42)` | 200 ảnh — OK, đủ đa dạng |
| 10 | `selected = valid_rows.sample(..., random_state=42)` | **chỉ 4 ảnh** |
| 14 | `df_sample.sample(..., random_state=42)` | 50 ảnh — OK |
| 16 | `valid_pairs.sample(..., random_state=42)` | 100 cặp — OK |
| 20 | `valid_masks.sample(..., random_state=42)` | 100 dòng → xem E-01 |

### Vấn đề (tập trung cell 10)

```python
selected = valid_rows.sample(min(len(valid_rows), num_samples), random_state=42)
#                                                ^^^^^^^^^^^^^^^ cố định
...
plot_5class_triplets(df_train, num_samples=4)
```

**4 mẫu vẽ ra luôn giống hệt nhau mọi lần chạy.** Chạy 100 lần vẫn ra 4 ảnh đó.

Với dataset chỉ có 20 scene, xác suất 4 mẫu ngẫu nhiên phủ đủ 5 nhóm sinh học (Bony Fish / Elasmobranch / Starfish / Seahorse / Turtle) rất thấp. Nhóm `Starfish` chỉ có `starfish2`, `Seahorse` chỉ có `seahorse1`, `Turtle` chỉ có `turtles1` — mỗi nhóm đúng 1 scene. Rất dễ 4 mẫu chọn toàn `fish*`.

Hậu quả: bạn **không bao giờ nhìn thấy** `seahorse1`, `turtles1`, `starfish2` trong cell 10 — 3/5 nhóm sinh học bị bỏ qua, dù đó là những nơi dễ sinh lỗi annotation nhất.

### Cách sửa — lấy mẫu có kiểm soát theo nhóm

```python
# Cell 10: chọn 1 scene đại diện cho mỗi nhóm sinh học
target_seqs = ["fish1", "sharkFish", "ray1", "turtles1", "seahorse1", "starfish2"]
sub = df_train[df_train["folder_sequence"].isin(target_seqs)]
plot_5class_triplets(sub, num_samples=len(target_seqs))
```

```python
# Cell 20: bỏ random hoàn toàn, dùng toàn bộ mask duy nhất
samples = valid_masks.drop_duplicates(subset="gt_mask_path")
```

---

## 🟠 E-08 — Cell 10: chỉ kiểm tra bằng mắt, không có sanity check đại số

**Mức độ:** Cao · **Cell:** 10

### Vấn đề

`plot_5class_triplets` là cell **xác minh chất lượng dữ liệu** — nhiệm vụ của nó là phát hiện nhãn sai, mask lệch, ghép ảnh nhầm. Nhưng nó **chỉ hiển thị hình ảnh**, không có phép tính nào.

Nếu mask lệch 5–10 pixel so với ảnh, mắt thường **không phát hiện được** trên ảnh 4000×3000 hiển thị thu nhỏ.

### Cách sửa — thêm kiểm tra đại số

```python
def sanity_check_triplet(row):
    deg  = cv2.imread(row["path"])
    gt   = cv2.imread(row["gt_image_path"])
    mask = load_mask(row["gt_mask_path"])
    issues = []

    # 1) Kích thước
    if deg.shape[:2] != gt.shape[:2]:
        issues.append(f"degraded {deg.shape[:2]} ≠ GT {gt.shape[:2]}")
    if mask.shape[:2] != gt.shape[:2]:
        issues.append(f"mask {mask.shape[:2]} ≠ GT {gt.shape[:2]}")

    # 2) GT phải sắc nét hơn degraded
    lap_gt  = compute_laplacian_var(cv2.cvtColor(gt,  cv2.COLOR_BGR2RGB))
    lap_deg = compute_laplacian_var(cv2.cvtColor(deg, cv2.COLOR_BGR2RGB))
    if lap_deg >= lap_gt:
        issues.append(f"degraded không mờ hơn GT ({lap_deg:.0f} >= {lap_gt:.0f})")

    # 3) GT phải khác ảnh degraded (không phải cùng một file)
    if deg.shape == gt.shape and np.array_equal(deg, gt):
        issues.append("degraded == GT, tức ảnh chưa bị suy giảm")

    # 4) Giá trị mask chỉ trong 0–4
    extra = set(np.unique(mask)) - {0, 1, 2, 3, 4}
    if extra:
        issues.append(f"mask có giá trị ngoài 0–4: {extra}")

    return issues

# Chạy trên tất cả 20 mask duy nhất
for p in df_train["gt_mask_path"].unique():
    row = df_train[df_train["gt_mask_path"] == p].iloc[0]
    iss = sanity_check_triplet(row)
    if iss:
        print(f"⚠ {Path(p).name}: {iss}")
print("Hoàn tất kiểm tra 20 mask.")
```

Lưu ý: `compute_laplacian_var` đang định nghĩa ở **cell 16**, tức là *sau* cell 10. Phải chuyển nó lên cell 4 hoặc định nghĩa lại trong cell 10.

---

## 🟡 E-09 — `parse_filename` nhánh `len(parts) == 2` không thể phân biệt

**Mức độ:** Trung bình · **Cell:** 4

### Vị trí

```python
elif len(parts) == 2:
    seq = parts[0]
    param1 = parts[1]
    param2 = None
```

### Vấn đề

Không thể phân biệt hai trường hợp:

| Tên file | Ý nghĩa thật | Code đọc thành |
| :--- | :--- | :--- |
| `fish1_5` | scene `fish1`, param = 5 | `seq="fish1", param1=5` ✅ |
| `shar_Ray` | scene tên `shar_Ray`, không có tham số | `seq="shar", param1="Ray"` ❌ |

Cùng một cấu trúc 2 phần, hai ý nghĩa hoàn toàn khác. Code phải đoán và có thể đoán sai.

### Ảnh hưởng thực tế

**Hiện tại không ảnh hưởng** — dataset đã xác nhận dùng nhánh 1 (`len >= 3`) cho mọi file. Nhưng đây là **bom hẹn giờ**: nếu `OVERVIEW.md` nói tên file là `{seq}_{depth}_{turbidity}` và có sequence tên 2 từ, toàn bộ ảnh của sequence đó sẽ bị parse sai hàng loạt.

### Cách sửa

```python
# Bỏ nhánh 2, ném lỗi nếu gặp trường hợp không xác định
if len(parts) >= 3:
    seq = "_".join(parts[:-2])
    param1, param2 = parts[-2], parts[-1]
else:
    raise ValueError(
        f"Tên file không đúng chuẩn '{stem}': {len(parts)} phần. "
        f"Expected format: {{sequence}}_{{param1}}_{{param2}}.png"
    )
```

Fail loudly tốt hơn fail silently.

---

## 🟡 E-10 — `parse_filename` có thể trả cột kiểu `object`, gây lỗi khi sort

**Mức độ:** Trung bình · **Cell:** 4, liên quan 12

### Vị trí

```python
"param1": int(param1) if param1 and param1.isdigit() else param1,
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#         nếu không phải số → giữ nguyên chuỗi
```

### Vấn đề

Cột `param1` có thể chứa lẫn lộn `int` và `str` → pandas gán kiểu `object`.

Hệ quả trực tiếp ở cell 12:

```python
seq_df = seq_df.sort_values(by=["param1", "param2"])
# TypeError: '<' not supported between instances of 'str' and 'int'
```

Và ở cell 16 khi gom nhóm thống kê.

### Cách sửa

```python
# Ngay sau khi tạo df_all
df_all["param1"] = pd.to_numeric(df_all["param1"], errors="coerce")
df_all["param2"] = pd.to_numeric(df_all["param2"], errors="coerce")

n_bad = df_all[["param1", "param2"]].isna().any(axis=1).sum()
assert n_bad == 0, f"{n_bad} dòng có param không parse được"
```

---

## 🟡 E-11 — `df_all` không được `.copy()` → side effect khi không có dev

**Mức độ:** Trung bình · **Cell:** 4

### Vị trí

```python
df_all = pd.concat([df_train, df_dev], ignore_index=True) if len(df_dev) > 0 else df_train
#                                                                          ^^^^^^^^^^^
#                                                          cùng ĐỐI TƯỢNG, không phải bản sao
```

### Vấn đề

Khi chạy trên môi trường không có thư mục dev, `df_all` và `df_train` trỏ tới **cùng một đối tượng trong bộ nhớ**. Mọi gán sau đó đều tác động lên cả hai:

```python
df_all["category"] = df_all["sequence"].apply(extract_category)   # cell 6
# → df_train cũng bị thêm cột "category"
```

Hiện tại không xảy ra (vì có dev), nhưng nếu chạy trên bản dataset chỉ có train, `df_train` sẽ bị thêm cột ngoài ý muốn.

### Cách sửa

```python
df_all = pd.concat([df_train, df_dev], ignore_index=True) if len(df_dev) > 0 else df_train.copy()
```

---

## 🟡 E-12 — `df_train` thiếu cột `category`, vì chỉ gán trên `df_all`

**Mức độ:** Trung bình · **Cell:** 4, 6

### Vị trí

```python
# Cell 6
if not df_all.empty:
    df_all["category"] = df_all["sequence"].apply(extract_category)
    #      ^^^^^^ chỉ gán vào df_all
```

### Vấn đề

`df_train` và `df_dev` không có cột `category`. Khi cần dùng cột này cho việc phân tích theo nhóm sinh học trên tập train, phải tự tính lại:

```python
df_train["category"].nunique()   # KeyError: 'category'
```

### Cách sửa

Gán cho cả ba bảng cùng lúc:

```python
for df in (df_train, df_dev, df_all):
    df["category"] = df["sequence"].apply(extract_category)
```

---

## 🟡 E-13 — Cell 8: kết luận resize cứng 384×768 sẽ làm méo ảnh

**Mức độ:** Trung bình · **Cell:** 8, 21

### Số liệu từ output cell 8

```
             width       height  aspect_ratio    mean_val
mean     4281.900000  3720.500000      1.208635    39.701850
min      2268.000000  3000.000000      0.562000     0.000000
25%      4000.000000  3000.000000      0.750000    13.675000
50%      4000.000000  3024.000000      1.333000    38.745000
75%      4032.000000  4032.000000      1.333000    57.747500
max      9710.000000  8000.000000      1.978000   135.350000
```

Tỉ lệ khung dao động **0.562 → 1.978**, tức có cả ảnh dọc (portrait) lẫn ảnh ngang rộng.

### Vấn đề

Cell 21 chốt kích thước tĩnh ONNX là `384 × 768` — tỉ lệ **2.0**. Nếu resize cứng mọi ảnh về 384×768 thì:

| Ảnh gốc | Tỉ lệ | Sau resize cứng | Mức méo |
| :--- | :--- | :--- | :--- |
| 4000×3000 | 1.333 | 384×768 | **nhẹ** |
| 3000×3000 | 1.000 | 384×768 | **43%** |
| 2268×3000 | 0.756 | 384×768 | **rất nặng** |

Ảnh vuông bị kéo dài gấp đôi chiều ngang → segmentation mask méo theo, ranh giới vật thể sai lệch.

Ngoài ra Track 1 yêu cầu "aspect-preserved, longest side ≤ 1024", tức tỉ lệ phải được **giữ nguyên**. Một pipeline duy nhất không thể vừa giữ tỉ lệ cho Track 1 vừa đáp ứng tensor tĩnh 384×768 cho Track 2.

### Cách sửa

```python
# Train: random crop giữ tỉ lệ thay vì resize cứng
H, W = 384, 768
if w < W:                       # ảnh dọc → pad
    pad = W - w
    img = cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=0)
else:                           # ảnh ngang → random crop
    x = np.random.randint(0, w - W + 1)
    y = np.random.randint(0, h - H + 1)
    img  = img[y:y+H, x:x+W]
    mask = mask[y:y+H, x:x+W]

# Track 1 submission: resize giữ tỉ lệ
def resize_for_submission(img, max_side=1024):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
```

`BORDER_REFLECT_101` cho ảnh, `BORDER_CONSTANT value=0` cho mask — vì class `0` là Background, pad bằng 0 là đúng ngữ nghĩa.

---

## 🟡 E-14 — Thiếu kiểm chứng mask chỉ chứa giá trị 0–4

**Mức độ:** Trung bình · **Cell:** 10, 20

### Vấn đề

`colorize_5class_mask` chỉ xử lý `range(5)`:

```python
def colorize_5class_mask(mask):
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx in range(5):
        colorized[mask == class_idx] = CLASS_COLORS_RGB[class_idx]
    return colorized
```

Pixel có giá trị **5, 6, 255... sẽ giữ nguyên màu đen `[0,0,0]`** — tức biến thành Background trong ảnh hiển thị, nhưng khi train thì vẫn là nhãn 5 (ngoài phạm vi). Và heatmap ở cell 20 cũng bỏ qua chúng:

```python
class_pixel_counts = {i: 0 for i in range(5)}   # không đếm giá trị khác
```

### Kiểm chứng đã chạy

```
Giá trị trong mask: [0 1 2 3]
```

Nút chỉ đọc **1 file**. Cần kiểm tra toàn bộ 20 mask:

```python
all_vals = set()
for p in df_train["gt_mask_path"].unique():
    all_vals |= set(np.unique(cv2.imread(p, cv2.IMREAD_GRAYSCALE)).tolist())
print(sorted(all_vals))
assert all_vals <= {0, 1, 2, 3, 4}, f"Mask có giá trị ngoài 0–4: {all_vals - {0,1,2,3,4}}"
```

### Cách sửa thêm trong `colorize_5class_mask`

```python
def colorize_5class_mask(mask):
    invalid = set(np.unique(mask)) - {0, 1, 2, 3, 4}
    if invalid:
        raise ValueError(f"Mask chứa giá trị ngoài 0–4: {invalid}")
    # ... phần còn lại
```

---

## 🟡 E-15 — Giả thuyết ý nghĩa `param1` / `param2` chưa được xác minh

**Mức độ:** Trung bình · **Cell:** 11 (markdown), 12

### Vấn đề

Cell 11 markdown khẳng định:

> `param1` correlates with **optical water attenuation coefficient** (color cast severity: greenish/blueish tint, red channel extinction).
> `param2` correlates with **turbidity & forward/backward scattering** (contrast loss, veiling light, and edge blur).

Đây là **suy đoán**, chưa được đối chiếu với `OVERVIEW.md`. Chỉ có bằng chứng gián tiếp: tên file, và hình ảnh trong cell 12.

Cell 3 còn viết khác:

> `param1` & `param2`: Synthetic degradation levels simulating water depth, turbidity, and **backscatter coefficient**.

Hai mô tả **không hoàn toàn trùng nhau** (một nói là attenuation + turbidity, một nhắc backscatter).

### Ảnh hưởng

Toàn bộ chiến lược augmentation theo tham số phụ thuộc vào việc hiểu đúng:

- Nếu `param1` là **attenuation** → tăng thì mất đỏ, nên augmentation tăng `param1` phải mô phỏng tương ứng
- Nếu `param1` thật ra là **độ sâu** → cần mô hình khác
- Nếu hoán đổi (param1 ↔ param2) → augmentation sai hoàn toàn, huấn luyện vẫn chạy nhưng không cải thiện

### Cách sửa

Xác minh thực nghiệm thay vì dựa vào giả định:

```python
# Ảnh khác nhau nhiều nhất khi param1 thay đổi hay param2 thay đổi?
from scipy.stats import pearsonr
sub = df_train.drop_duplicates(subset="gt_image_path")   # 20 ảnh gốc

# Với mỗi ảnh gốc, so sánh frame theo param1 vs frame theo param2
for p1a, p1b in [(1, 10)]:
    for p2a, p2b in [(5, 100)]:
        pass  # logic tương tự

# Cách đơn giản: tương quan giữa tham số và đặc trưng ảnh
feat = []   # ví dụ: tỉ lệ pixel đỏ trung bình
for _, r in sub.iterrows():
    img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
    feat.append((r["param1"], r["param2"],
                 img[:, :, 0].mean(),   # đỏ
                 cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()))
df_feat = pd.DataFrame(feat, columns=["p1", "p2", "red_mean", "lap_var"])

print("corr(param1, đỏ) =", pearsonr(df_feat["p1"], df_feat["red_mean"])[0])
print("corr(param2, đỏ) =", pearsonr(df_feat["p2"], df_feat["red_mean"])[0])
print("corr(param1, nét) =", pearsonr(df_feat["p1"], df_feat["lap_var"])[0])
print("corr(param2, nét) =", pearsonr(df_feat["p2"], df_feat["lap_var"])[0])
```

Nếu `corr(param1, đỏ)` âm mạnh và `corr(param2, nét)` thấp hơn nhiều → giả thuyết của cell 11 là đúng.

---

## 🟡 E-16 — Cell 16: chỉ đo "hành động", không có baseline mô hình học

**Mức độ:** Trung bình · **Cell:** 16

### Vấn đề

Cell 16 chỉ tính metric của cặp `degraded` vs `GT` — tức **hiệu năng của hành động "không làm gì"**. Không hề so sánh với bất kỳ mô hình nào.

Kết quả:

| Chỉ số | Trung vị |
| :--- | :--- |
| PSNR | 11.7 dB |
| `psnr_norm` | **0.0000** |
| SSIM | 0.427 |
| `sharpness_ratio` | **0.0113** |

### Vấn đề diễn giải

`sharpness_ratio` trung vị 0.0113 — nhưng **con số này không cho biết mô hình phải làm được gì**. Người đọc dễ hiểu nhầm là "input gần như giống GT nên dễ".

Thực tế ngược lại: input giữ được **1,1% năng lượng tần số cao** của GT. Muốn đạt `sharpness_ratio = 1.0`, mô hình phải **sinh chế texture không tồn tại trong input**. Đó là bài toán tạo sinh chi tiết, không phải sửa màu.

### Cách sửa — thêm baseline

```python
# Baseline 1: copy input
# Baseline 2: tăng tương phản + chỉnh gamma (rẻ hơn nhiều)
def baseline_enhance(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    a = cv2.addWeighted(a, 1.3, 0, 0, 0)   # kéo về đỏ
    b = cv2.addWeighted(b, 1.1, 0, 0, 0)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# So sánh cả 3: degraded / baseline / model
```

Không có baseline thì không biết model có thật sự tiến bộ hay chỉ "thở" được vài điểm.

---

## 🟢 E-17 — Cell 6: điều kiện `extract_category` dễ đảo ngược thứ tự

**Mức độ:** Thấp · **Cell:** 6

### Vị trí

```python
def extract_category(seq_name):
    seq_lower = str(seq_name).lower()
    if "shark" in seq_lower or "ray" in seq_lower:
        return "Elasmobranch (Shark/Ray)"
    elif "fish" in seq_lower and "star" not in seq_lower:
        return "Bony Fish"
    elif "starfish" in seq_lower or "star" in seq_lower:
        return "Starfish"
```

### Vấn đề

Nhánh `"fish" in seq_lower and "star" not in seq_lower` **phải đứng trước** nhánh starfish. Hiện tại đúng, vì `"fish" in "starfish"` = True nên điều kiện `"star" not in` chặn lại → rơi xuống nhánh Starfish.

Nhưng nếu ai đó sửa thành:

```python
elif "starfish" in seq_lower or "star" in seq_lower:   # đưa lên trên
    return "Starfish"
elif "fish" in seq_lower:                              # bỏ điều kiện chặn
    return "Bony Fish"
```

thì `starfish1` → "Starfish" ✅ vẫn đúng, nhưng logic trở nên phụ thuộc thứ tự ngẫu nhiên.

### Cách sửa — dùng thứ tự khớp dài trước

```python
def extract_category(seq_name):
    s = str(seq_name).lower()
    # Kiểm tra theo mức độ đặc thù, từ dài và hiếm trước
    for keyword, label in [
        ("starfish", "Starfish"),
        ("seahorse", "Seahorse"),
        ("turtle",   "Turtle"),
        ("shark",    "Elasmobranch (Shark/Ray)"),
        ("ray",      "Elasmobranch (Shark/Ray)"),
        ("fish",     "Bony Fish"),
    ]:
        if keyword in s:
            return label
    return "Other / Mixed"
```

Tách `seahorse`/`turtle` ra trước cũng cần thiết: hiện tại `seahorse1` chứa chữ "sea" chứ không chứa "fish" nên rơi vào nhánh cuối → "Other / Mixed" ❌. Bảng cell 6 hiện ghi `Seahorse` cho `seahorse1`, nghĩa là đã được xử lý đúng chỗ khác — nhưng logic đang phụ thuộc thứ tự rất mỏng manh.

---

## 🟢 E-18 — Cell 8: `sns.boxplot` với `palette` sẽ lỗi ở seaborn 0.14+

**Mức độ:** Thấp · **Cell:** 8

### Output hiện tại

```
FutureWarning:
Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0.
Assign the `x` variable to `hue` and set `legend=False` for the same effect.
  sns.boxplot(data=df_props, x="split", y="mean_val", ax=axes[2], palette="Blues")
```

### Vấn đề

Đây là `FutureWarning`, chưa phải lỗi. Nhưng ở seaborn 0.14+ nó sẽ **thành lỗi thật**, khiến notebook không chạy được.

### Cách sửa

```python
sns.boxplot(data=df_props, x="split", y="mean_val", hue="split",
            ax=axes[2], palette="Blues", legend=False)
```

Cùng kiểu lỗi có thể xuất hiện ở cell 6: `sns.countplot(..., hue="split", palette="viridis")` — ở đó có `hue` nên không sao, nhưng nên kiểm tra.

---

## 🟢 E-19 — Cell 16: thiếu handle vì gọi hàm ở cấp module

**Mức độ:** Thấp · **Cell:** 16

### Vị trí

```python
def compute_laplacian_var(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_image_metrics(df_train_pairs, max_pairs=80):
    ...
        deg_var = compute_laplacian_var(deg)     # deg là BGR!
        gt_var  = compute_laplacian_var(gt)      # gt  là BGR!
```

### Vấn đề

Hàm khai báo tham số là `img_rgb` và chuyển `RGB2GRAY`, nhưng lại được gọi với ảnh BGR từ `cv2.imread`. Sai lệch này **nhất quán** giữa cả hai lời gọi nên phép so sánh ratio vẫn tương đối hợp lệ — nhưng tham số tên sai dễ khiến người khác tin là đúng và tái sử dụng sai.

Ngoài ra, nếu sau này gọi `compute_laplacian_var` từ cell 10 như đề xuất ở [E-08](#e-08--cell-10-chỉ-kiểm-tra-bằng-mắt-không-có-sanity-check-đại-số) thì càng dễ nhầm.

### Cách sửa

```python
def compute_laplacian_var(img):
    """img: ảnh grayscale hoặc BGR/RGB bất kỳ."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img.astype(np.float64), cv2.CV_64F).var()
```

Cách này không quan tâm thứ tự kênh, phù hợp cho cả 2 chiều.

---

## 🟢 E-20 — Cell 10: overlay 55% màu che khuất biên vật thể

**Mức độ:** Thấp · **Cell:** 10

### Vị trí

```python
overlay[fg_bool] = cv2.addWeighted(gt_img[fg_bool], 0.45, color_mask[fg_bool], 0.55, 0)
#                                                        45% ảnh            55% màu
```

### Vấn đề

55% màu phủ lên → ảnh gốc chỉ còn 45%. Ở ảnh 4000×3000 hiển thị thu nhỏ, người quan sát **không đánh giá được độ chính xác của ranh giới segmentation** — mà đó mới là mục tiêu của overlay.

Trong khi đó, cell 10 không có panel crop phóng to.

### Cách sửa

```python
overlay[fg_bool] = cv2.addWeighted(gt_img[fg_bool], 0.6, color_mask[fg_bool], 0.4, 0)
#                                                        60% ảnh            40% màu
```

Và thêm một hàng crop phóng to 256×256 tại vùng có biên class:

```python
# Tìm vùng biên: pixel thuộc class khác 0 nhưng lân cận class 0
from scipy.ndimage import binary_dilation
fg = gt_mask_res > 0
edge = fg ^ binary_dilation(fg, iterations=8)
ys, xs = np.where(edge)
if len(ys):
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = max(0, cy-128), max(0, cx-128)
    crop_gt     = gt_img[y0:y0+256, x0:x0+256]
    crop_mask   = colorize_5class_mask(gt_mask_res[y0:y0+256, x0:x0+256])
    axes[i, 3].imshow(cv2.addWeighted(crop_gt, 0.6, crop_mask, 0.4, 0))
```

---

## 🟢 E-21 — Cell 12: rò rỉ hành vi

**Mức độ:** Thấp · **Cell:** 12

### Vị trí

```python
seq_df = df[df["folder_sequence"] == target_seq].copy()
#                    ^^^ thiếu .copy()
...
seq_df = seq_df.sort_values(by=["param1", "param2"]).head(10)
```

`df[boolean_mask]` đã trả về một bản sao trong pandas hiện đại, nên `.copy()` thừa ở đây. Tuy nhiên nếu sau này ai đó đổi thành chỉ mục boolean phức tạp hơn hoặc dùng `.loc`, dòng `.copy()` này là chốt an toàn.

Vấn đề thật sự là **cell 12 chỉ hiển thị `.head(10)`** — 10 frame đầu sau khi sort tức là **10 mức nhiễu nhẹ nhất** (`param1` nhỏ, `param2` nhỏ), không phải phổ đầy đủ. Điều này khiến người đọc đánh giá **thấp** mức độ hỏng của dataset.

### Cách sửa

```python
# Lấy đều mẫu trên toàn bộ dải tham số, không chỉ đầu dãy
seq_df = df[df["folder_sequence"] == target_seq].copy()
seq_df = (seq_df.sort_values(["param1", "param2"])
                .iloc[np.linspace(0, len(seq_df)-1, 10, dtype=int)])
# → 10 frame trải đều từ nhẹ nhất đến nặng nhất
```

---

## Checklist ưu tiên xử lý

Làm theo thứ tự này, vì mỗi mục mở đường cho mục sau.

### Ưu tiên 1 — Chạy ngay sau cell 4 (5 phút)

```python
# [E-04] Không rò rỉ
overlap = set(df_train["folder_sequence"]) & set(df_dev["folder_sequence"])
assert len(overlap) == 0, f"LEAK: {overlap}"

# [E-06] Cấu trúc ghép cặp
assert df_train.groupby("folder_sequence")["gt_image_path"].nunique().max() == 1
assert df_train["gt_image_path"].nunique() == 20
assert df_train["gt_mask_path"].nunique() == 20

# [E-10] Kiểu dữ liệu
df_all["param1"] = pd.to_numeric(df_all["param1"], errors="coerce")
df_all["param2"] = pd.to_numeric(df_all["param2"], errors="coerce")
assert not df_all[["param1", "param2"]].isna().any().any()

# [E-12] Cột category
for df in (df_train, df_dev, df_all):
    df["category"] = df["sequence"].apply(extract_category)
```

### Ưu tiên 2 — Trước khi train (30 phút)

- [ ] **[E-02]** Kiểm tra class 4 có tồn tại ở đâu, tỉ lệ pixel 5 lớp chính xác
- [ ] **[E-14]** Assert mask chỉ chứa giá trị 0–4
- [ ] **[E-03]** Tách biết `df_for_stats = df_train`, không dùng `df_all` cho normalize / class weight
- [ ] **[E-01]** Sửa cell 20 thành `drop_duplicates(subset="gt_mask_path")`
- [ ] **[E-07]** Sửa cell 10 lấy mẫu theo nhóm sinh học
- [ ] **[E-05]** Ghi chú rõ "20 ảnh gốc × N mức nhiễu" khi báo cáo cell 16
- [ ] **[E-13]** Quyết định chiến lược resize: random crop giữ tỉ lệ + pad

### Ưu tiên 3 — Trước khi nộp (không gấp)

- [ ] **[E-08]** Thêm sanity check đại số vào cell 10
- [ ] **[E-15]** Xác minh ý nghĩa thật của `param1` / `param2` bằng tương quan
- [ ] **[E-16]** Thêm baseline đơn giản để so sánh
- [ ] **[E-20]** Giảm tỉ lệ màu trong overlay, thêm panel crop biên
- [ ] **[E-17]** Viết lại `extract_category` theo thứ tự ưu tiên
- [ ] **[E-18]** Sửa `sns.boxplot` để tương thích seaborn 0.14+
- [ ] **[E-19]** Sửa chữ ký `compute_laplacian_var`
- [ ] **[E-09]** [E-11]** [E-21] Dọn code phòng thủ
- [ ] **[E-16]** Kiểm tra `check_submission_flops.py` trước khi nộp Track 2

---

## Ghi chú

Tài liệu này ghi lại **lỗi và rủi ro**, không phải phê bình phong cách code. Nhiều "lỗi" ở đây là quyết định lướt (chọn `random_state=42` cho tái lập, dùng `sample` vì chưa nhận ra chỉ có 20 mask duy nhất). Chúng được ghi lại để người tiếp nhận notebook — có thể là bạn — biết **điều gì đang được giả định** và **điều gì cần kiểm chứng lại** trước khi mở rộng.

Không có lỗi cú pháp nào; toàn bộ 22 cell chạy được và sinh ra output hợp lệ. Vấn đề nằm ở chỗ **những giả định im lặng** mà không có `assert` nào bảo vệ.
