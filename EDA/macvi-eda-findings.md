# Kết quả đo và Phân tích — MaCVi EURS

> Tài liệu ghi lại **số liệu thực nghiệm** đo được trên dataset, kèm giải thích ý nghĩa từng con số.
>
> **Nguồn dữ liệu:** chạy trên Kaggle notebook với `df_train` đã được index bởi cell 4.
> **Ngày đo:** phiên phân tích EDA.
>
> **Vấn đề được kiểm chứng:** E-01 trong [`macvi-eda-bugs.md`](./macvi-eda-bugs.md) — cell 20 lấy mẫu 100 dòng trong khi chỉ có 20 file mask duy nhất.

---

## Tóm tắt nhanh

| # | Phát hiện | Mức độ |
| :-- | :--- | :--- |
| 1 | Phân bố pixel thật của 5 lớp: **Background 55.31% · Coral 33.73% · Fish 7.80% · Other 2.21% · Elasmobranch 0.94%** | ⭐ Quan trọng nhất |
| 2 | Lớp hiếm nhất là **Elasmobranch (0.94%)**, không phải class 4 | Đính chính |
| 3 | **Coral không hiếm** (33.73%) — nhận định trong cell 21 gây hiểu nhầm | Đính chính |
| 4 | Class 4 **có tồn tại** và chiếm 2.21% — loại bỏ lo lắng E-02 | Xác nhận |
| 5 | Cell 20 (seed 42) sai lệch tới **6.5 điểm %**; thứ hạng 2 lớp hiếm nhất bị **đảo ngược** | Lỗi thật |
| 6 | Lấy mẫu 100 dòng **không thiên lệch** (sai số TB < 1 điểm) nhưng **đuôi nặng** (max ~10 điểm) | Giải thích cơ chế |

---

## 1. Cách đo

### 1.1. Vấn đề cần đo

Cell 20 dùng:

```python
samples = valid_masks.sample(100, random_state=42)
```

Cần biết: lấy 100 dòng như vậy làm sai lệch bao nhiêu so với tính chính xác trên cả 20 mask?

### 1.2. Vì sao không thể dùng cách thông thường

Bản kiểm chứng "lặp 50 lần với seed khác nhau" viết theo kiểu thông thường sẽ:

- `pixel_pct` gọi `cv2.imread` **5000 lần** (50 seed × 100 dòng)
- Mỗi lần giải nã 1 file PNG 12 megapixel
- **Thời gian: 15–30 phút**

Nguyên nhân: 5000 lần đọc nhưng chỉ có 20 file khác nhau → 4980 lần đọc thừa. Và `for c in range(5): (m==c).sum()` quét 12 triệu pixel **5 lần** thay vì 1 lần.

### 1.3. Cách tối ưu: tách I/O khỏi tính toán

Đọc 20 mask đúng một lần, chỉ giữ số pixel mỗi lớp, rồi mọi phép thống kê chạy trên mảng numpy nhỏ.

```python
import time, cv2, numpy as np

# --- BƯỚC 1: đọc 20 mask, chỉ giữ số pixel mỗi lớp ---
t = time.perf_counter()
counts = {}
for p in df_train["gt_mask_path"].unique():
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    counts[p] = np.bincount(m.ravel(), minlength=5)[:5].astype(np.int64)
print(f"Đọc {len(counts)} mask trong {time.perf_counter()-t:.1f}s")

# --- BƯỚC 2: đưa về dạng mảng 4000 × 5 để vectorize ---
k = np.stack([counts[p] for p in df_train["gt_mask_path"]])
total = k.sum(axis=0)
exact = 100 * total / total.sum()

# --- BƯỚC 3: 50 lần lấy mẫu — tức thì, không đọc lại file ---
errs = []
for s in range(50):
    idx = df_train.sample(100, random_state=s).index
    errs.append(100 * k[idx].sum(axis=0) / k[idx].sum() - exact)
errs = np.array(errs)

# --- In kết quả ---
seed42 = 100 * k[df_train.sample(100, random_state=42).index].sum(axis=0) / \
         k[df_train.sample(100, random_state=42).index].sum()
print(f"{'Lớp':<16}{'Chính xác':>11}{'cell20':>10}{'Sai số TB':>12}{'Max|sai số|':>13}")
for c, n in enumerate(CLASS_NAMES):
    print(f"{c}: {n:<13}{exact[c]:>10.2f}%{seed42[c]:>9.2f}%"
          f"{errs[:,c].mean():>11.2f}{np.abs(errs[:,c]).max():>12.2f}")
```

### 1.4. Ba kỹ thuật tối ưu đã dùng

| Kỹ thuật | Thay cho | Tác dụng |
| :--- | :--- | :--- |
| Đọc 20 file một lần, cache kết quả | `imread` mỗi lần | 5000 lần đọc → **20 lần** |
| `np.bincount(m.ravel())` | `for c in range(5): (m==c).sum()` | quét 12 MP **5 lần → 1 lần** |
| `k[idx].sum(axis=0)` | vòng lặp Python | 50 lần lấy mẫu → phép cộng mảng |

**Kết quả thời gian: 2.1 giây toàn bộ** (thay vì 15–30 phút).

> Bài học: cell 20 hiện tại đọc 100 lần để làm việc chỉ cần 20 lần. Khi chạy notebook từ đầu, đó là vài chục giây lãng phí mỗi lần — không nhiều, nhưng hoàn toàn vô nghĩa.

---

## 2. Kết quả đo

```
Đọc 20 mask trong 2.1s
50 lần lấy mẫu xong trong 2.1s

Lớp               Chính xác    cell20   Sai số TB  Max|sai số|
0: Background        55.31%    49.67%      -0.62        9.25
1: Coral             33.73%    40.25%       0.66        9.81
2: Fish               7.80%     7.05%       0.14        2.33
3: Elasmobranch       0.94%     1.60%      -0.06        0.66
4: Other              2.21%     1.44%      -0.11        1.33
```

Tổng: 55.31 + 33.73 + 7.80 + 0.94 + 2.21 = **99.99%** ✓

---

## 3. Ý nghĩa từng cột

### 3.1. Cột "Chính xác" — con số quan trọng nhất

Đây là **phân bố pixel thật của toàn bộ dataset phân đoạn**, tính trên đủ 20 mask.

| Lớp | % pixel | Xếp hạng | Nhận xét |
| :--- | ---: | :--: | :--- |
| 0 Background | **55.31%** | 1 | Hơn một nửa khung hình |
| 1 Coral | **33.73%** | 2 | Gần 1/3 — **không hề hiếm** |
| 2 Fish | 7.80% | 3 | |
| 4 Other | 2.21% | 4 | |
| 3 Elasmobranch | **0.94%** | 5 | **Hiếm nhất** |

Mức mất cân bằng: Background ÷ Elasmobranch = 55.31 ÷ 0.94 ≈ **59 lần**.

### 3.2. Cột "Sai số TB" → lấy mẫu KHÔNG thiên lệch có hệ thống

| Lớp | Sai số TB |
| :--- | ---: |
| Background | −0.62 |
| Coral | +0.66 |
| Fish | +0.14 |
| Elasmobranch | −0.06 |
| Other | −0.11 |

Tất cả dưới 1 điểm %, xoay quanh 0.

**Ý nghĩa:** nếu bạn lặp lại lấy mẫu nhiều lần, sai số trung bình triệt tiêu. Cell 20 **không** ưu tiên sai một hướng cụ thể — nó chỉ nhiễu.

Đây là tín hiệu tốt về mặt phương pháp: `sample(100)` về nguyên tắc là một ước lượng không lệch.

### 3.3. Cột "Max|sai số|" → đây mới là vấn đề thật

| Lớp | Max sai số (trên 50 seed) |
| :--- | ---: |
| Background | **9.25** |
| Coral | **9.81** |
| Fish | 2.33 |
| Elasmobranch | 0.66 |
| Other | 1.33 |

**Sai số cực đại lên tới gần 10 điểm phần trăm.**

Hình dung: nếu đổi `random_state` sang giá trị khác, biểu đồ thanh trong cell 20 có thể hiển thị Coral là 43.5% thay vì 33.73% thật — chênh nhau gần 10 điểm.

**Ý nghĩa:** trung kỳ đúng, nhưng **đơn lẻ thì sai rất nhiều**. Và notebook chỉ chạy **một** lần lấy mẫu.

### 3.4. Cột "cell20" → seed 42 rơi vào nhóm xấu

| Lớp | Chính xác | cell20 (seed 42) | Sai số | Sai số tương đối |
| :--- | ---: | ---: | ---: | ---: |
| 0 Background | 55.31% | 49.67% | −5.64 | −10.2% |
| 1 Coral | 33.73% | 40.25% | **+6.52** | **+19.3%** |
| 2 Fish | 7.80% | 7.05% | −0.75 | −9.6% |
| 3 Elasmobranch | 0.94% | 1.60% | +0.66 | **+70.2%** |
| 4 Other | 2.21% | 1.44% | −0.77 | **−34.8%** |

**Ý nghĩa:** biểu đồ đang được in ra trong notebook sai lệch tới 6.5 điểm %. Sai số tương đối lớn nhất nằm ở **hai lớp hiếm** — đúng nhóm quan trọng nhất.

---

## 4. Tại sao sai số lại lớn như vậy

### 4.1. Công thức

Gọi $p_i(c)$ là tỉ lệ pixel lớp $c$ trong mask thứ $i$ (đúng đắn):

$$P_{\text{thật}}(c) = \frac{1}{20}\sum_{i=1}^{20} p_i(c)$$

Cell 20 cho ra, với $k_i$ = số lần mask $i$ bị lấy mẫu và $\sum k_i = 100$:

$$P_{\text{cell20}}(c) = \sum_{i=1}^{20} \frac{k_i}{100}\, p_i(c)$$$

Vì $k_i$ trung bình 5, trung kỳ là đúng (khớp cột "Sai số TB"). Nhưng độ lệch chuẩn là:

$$\text{Std}[P_{\text{cell20}}(c)] = 0.0215 \times \sqrt{\sum_{i=1}^{20} p_i(c)^2}$$

Trong đó 0.0215 là độ lệch chuẩn của $k_i/100$ (do $k_i$ có trung bình 5, độ lệch chuẩn 2.15).

### 4.2. Tính cho từng lớp

**Background** (55.31%, giả sử đều ở cả 20 mask):

$$\sqrt{20 \times 0.5531^2} = 2.47 \quad\Longrightarrow\quad \text{Std} = 0.0215 \times 2.47 = \mathbf{5.3 \text{ điểm }\%}$$

Max trên 50 lần rút ≈ 1.86 × 5.3 ≈ 9.9 → quan sát được 9.25 ✓

### 4.3. Nhận xét quan trọng: sai số càng LỚN khi lớp càng phổ biến và đều

Điều này ngược với trực giác. Giải thích:

Khi lớp chiếm tỉ lệ lớn và **đều ở mọi mask**, việc một mask bị đánh trọng số 9× thay vì 5× kéo toàn bộ trung bình dịch chuyển mạnh → sai số lớn.

Ngược lại, lớp tập trung ở ít mask thì dù có đánh trọng số sai, ảnh hưởng tới tổng vẫn nhỏ vì $p_i(c)$ của các mask khác gần 0.

### 4.4. Một quan sát từ dữ liệu: các lớp không phân bố đều

Giá trị `Max|sai số|` **luôn lớn hơn** dự đoán "giả sử phân bố đều":

| Lớp | Max quan sát | Max dự đoán (đều) |
| :--- | ---: | ---: |
| Background | 9.25 | ~9.9 |
| Coral | **9.81** | ~6.0 |
| Fish | **2.33** | ~1.4 |
| Elasmobranch | **0.66** | ~0.2 |
| Other | **1.33** | ~0.4 |

Ngoại trừ Background, mọi lớp đều cao hơn dự đoán → **các lớp tập trung ở một nhóm scene cụ thể**, không phân bố đều. Điều này hoàn toàn hợp lý:

- `Coral` tập trung ở các scene `fish*`
- `Elasmobranch` chỉ ở `sharkFish`, `ray1`
- `Other` chỉ ở `starfish2`, `seahorse1`, `turtles1`

**Hệ quả:** chỉ một nhóm scene bị đánh trọng số 9× có thể làm sai lệch lớn — và nhóm đó chính là nhóm quyết định giá trị của lớp.

---

## 5. Hậu quả cụ thể: thứ hạng hai lớp hiếm nhất bị đảo ngược

### 5.1. Cơ sở: `class_weight` tỉ lệ nghịch với tỉ lệ pixel

$$w_c \propto \frac{1}{p_c}$$

Để dễ so sánh, chuẩn hoá sao cho **Background = 1.00** (tức lớp đó nhận trọng số 1 phần).

### 5.2. Với số đúng

| Lớp | $p_c$ | $w_c$ | Trọng số so với Background |
| :--- | ---: | ---: | ---: |
| 0 Background | 55.31% | 1.808 | **1.00×** |
| 1 Coral | 33.73% | 2.965 | **1.64×** |
| 2 Fish | 7.80% | 12.821 | **7.09×** |
| 3 Elasmobranch | 0.94% | 106.38 | **58.83×** |
| 4 Other | 2.21% | 45.25 | **25.02×** |

### 5.3. Với số của cell 20

| Lớp | $p_c$ | $w_c$ | Trọng số so với Background |
| :--- | ---: | ---: | ---: |
| 0 Background | 49.67% | 2.014 | **1.00×** |
| 1 Coral | 40.25% | 2.484 | **1.23×** |
| 2 Fish | 7.05% | 14.184 | **7.04×** |
| 3 Elasmobranch | 1.60% | 62.50 | **31.03×** |
| 4 Other | 1.44% | 69.44 | **34.48×** |

### 5.4. So sánh

| Lớp | Trọng số đúng | cell 20 | Sai số |
| :--- | ---: | ---: | ---: |
| Background | 1.00× | 1.00× | — |
| Coral | 1.64× | 1.23× | −25% |
| Fish | 7.09× | 7.04× | −1% |
| **Elasmobranch** | **58.83×** | **31.03×** | **−47%** |
| **Other** | **25.02×** | **34.48×** | **+38%** |

### 5.5. Đảo ngược thứ hạng — hậu quả nghiêm trọng nhất

Tỉ số trọng số giữa hai lớp hiếm nhất:

$$\text{Đúng:}\quad \frac{w_{\text{Elasmobranch}}}{w_{\text{Other}}} = \frac{58.83}{25.02} = 2.35$$

$$\text{Cell 20:}\quad \frac{w_{\text{Elasmobranch}}}{w_{\text{Other}}} = \frac{31.03}{34.48} = 0.90$$

| | Thứ tự đúng | Thứ tự cell 20 |
| :--- | :--- | :--- |
| Lớp hiếm hơn | **Elasmobranch** | **Other** |
| Lớp khan hiếp hơn | Elasmobranch (2.35×) | Other (1.11×) |

**Cell 20 tin rằng `Other` khan hiếp hơn `Elasmobranch` — hoàn toàn ngược với thực tế.**

Đây là lý do E-01 là lỗi **thật**, không phải lỗi hình thức. Vì:

1. `Other` và `Elasmobranch` là hai lớp khác hẳn về hình ảnh (sao biển/rùa/cá ngựa/cá ngựa vs cá mập/cá đuối)
2. Mô hình bị phân bổ trọng số ngược → ưu tiên sai lớp cần học
3. `Q_seg = F1 × mIoU × coverage_penalty` chiếm **60%** tổng điểm, mà `mIoU` trung bình **đều trên 5 lớp** → sai một lớp hiếm là mất một phần lớn điểm

---

## 6. Phát hiện quan trọng nhất: phân bố lớp thật

Bảng ở mục 3.1 lật ngược ba giả định đang có trong notebook.

### 6.1. Class 4 CÓ tồn tại, chiếm 2.21%

Câu hỏi "lớp 4 có xuất hiện ở đâu không?" được giải đáp: **có**, và chiếm 2.21% pixel — nhiều hơn 2 lần so với Elasmobranch.

→ Loại bỏ hoàn toàn lo lắng E-02 trong báo cáo lỗi (lo lắng class 4 biến mất khỏi biểu đồ).

### 6.2. Lớp hiếm nhất là Elasmobranch, KHÔNG phải class 4

| | Tỉ lệ | Xếp hạng hiếm |
| :--- | ---: | :--: |
| 3 Elasmobranch | **0.94%** | **1 (hiếm nhất)** |
| 4 Other | 2.21% | 2 |

Lớp cần được ưu tiên tuyệt đối khi thiết kế loss là **Elasmobranch**.

### 6.3. Coral KHÔNG hiếm — nhận định trong cell 21 gây hiểu nhầm

Cell 21 ghi:

> `FocalTversky` *(handles severe background vs **elasmobranch/coral** class imbalance)*

Cụm từ này **gợi ý rằng Coral là lớp hiếm**. Nhưng Coral chiếm **33.73%** — lớp phổ biến thứ hai, chỉ kém Background.

**Vấn đề thật** là:

$$\text{Background (55.31\%)} \quad \gg \quad \text{Elasmobranch (0.94\%)}$$

Chỉ `Elasmobranch` mới cần trọng số mạnh. Đặt trọng số cao cho Coral sẽ phá hoại kết quả vì Coral vốn không hiếm và dễ học.

### 6.4. Mức mất cân bằng "chỉ" 59 lần, không phải 90 lần

Background 55.31% không phải 90%. Hình ảnh khá "đầy" — cá và san hô chiếm gần một nửa khung hình, khác với trực giác mặc định về ảnh đại dương nhiều nước.

Mức mất cân bằng thực tế: 55.31 ÷ 0.94 ≈ **59×** (thay vì 90× nếu Background chiếm 85%).

---

## 7. Hành động cần thực hiện

### 7.1. Dùng số liệu chính xác cho class weight

```python
# Tỉ lệ pixel chính xác (đo trên đủ 20 mask)
PIXEL_PCT = [0.5531, 0.3373, 0.0780, 0.0094, 0.0221]   # thứ tự 0,1,2,3,4

class_weight = torch.tensor([1/p for p in PIXEL_PCT])
class_weight = class_weight / class_weight[0]           # Background = 1.0
# → tensor([1.00, 1.64, 7.09, 58.83, 25.02])
```

### 7.2. Sửa cell 20

```python
samples = valid_masks.drop_duplicates(subset="gt_mask_path")
# → 20 dòng, 20 file, kết quả chính xác tuyệt đối
```

Và cập nhật lời gọi:

```python
analyze_5class_masks(df_train, max_samples=100)   # tham số này không còn tác dụng
```

### 7.3. Sửa nhận định trong cell 21

Thay ghi chú về mất cân bằng bằng:

```markdown
**Phân bố pixel (đo trên 20 mask):**
| Lớp | % pixel | weight ∝ 1/p |
| :--- | ---: | ---: |
| 0 Background | 55.31% | 1.00× |
| 1 Coral | 33.73% | 1.64× |
| 2 Fish | 7.80% | 7.09× |
| 3 Elasmobranch | **0.94%** | **58.83×** |
| 4 Other | 2.21% | 25.02× |

- Lớp hiếm nhất là **Elasmobranch** (59× khan hiếp hơn Background)
- **Coral không hiếm** — chỉ kém Background, không cần trọng số mạnh
- `Q_seg` chiếm 60% điểm, `mIoU` trung bình đều 5 lớp → lớp hiếm quyết định thắng thua
```

### 7.4. Kiểm tra bổ sung: mask có cùng kích thước không?

Bảng ở mục 3.1 giả định 20 mask có cùng số pixel. Cần xác nhận:

```python
for p in df_train["gt_mask_path"].unique():
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    print(f"{Path(p).name:16} {m.shape}")
```

Nếu kích thước khác nhau, cần quyết định: tính trung bình theo mask hay theo tổng pixel.

### 7.5. Xác nhận class 4 bằng trực quan

```python
for p in df_train["gt_mask_path"].unique():
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    n4 = int((m == 4).sum())
    if n4:
        print(f"{Path(p).name:16} class 4 = {n4:>10,} px ({100*n4/m.size:.2f}%)")
```

Kỳ vọng: chỉ `starfish2`, `seahorse1`, `turtles1` có class 4.

---

## 8. Nhật ký đính chính

Ghi lại các dự đoán của tôi đã sai trong quá trình phân tích, để tiếp nhận file này không bị nhầm lẫn.

| # | Tôi đã nói | Thực tế | Ảnh hưởng |
| :-- | :--- | :--- | :--- |
| 1 | "Sai số khoảng **±1–2 điểm %**" | Sai số chuẩn ~**5 điểm %**, max **9.8** | Đánh giá nhẹ E-01. Đã sửa lại mức 🔴 |
| 2 | "Cell 20 cho ra đúng kết luận, sai số không làm hỏng quyết định thiết kế loss" | **Sai** — thứ hạng 2 lớp hiếm nhất bị đảo ngược | Đảo ngược kết luận |
| 3 | "E-01 không nghiêm trọng, có thể hạ xuống 🟡" | **Sai** — có bằng chứng định lượng, giữ 🔴 | Giữ nguyên 🔴 |
| 4 | "Kết quả không có tính tái lập" | **Sai** — `random_state=42` cố định nên **tái lập được** | Vấn đề thật là "sai nhưng trông rất chắc chắn" |
| 5 | "Class 4 có thể bị bỏ sót, xác suất đáng kể" | Xác suất ~1.7×10⁻⁷, gần như không thể | Loại bỏ E-02 |
| 6 | "Coral là lớp hiếm cần trọng số" | Coral chiếm **33.73%**, lớp phổ biến thứ 2 | Đính chính cell 21 |
| 7 | "Lớp hiếm nhất có thể là class 4" | Lớp hiếm nhất là **Elasmobranch (0.94%)** | Định hướng loss |
| 8 | Ví dụ minh hoạ "sai số 39%" (3 sequence tự chế) | Con số đúng về cơ chế nhưng tỉ lệ sai (xem #1) | Giữ phần giải thích, bỏ con số 39% |

**Nguyên nhân chung:** tôi đã ước lượng thống kê thay vì chờ đo, và công thức sai số chuẩn có phần tử $\sqrt{\sum p_i^2}$ mà tôi đánh giá nhầm. May mắn là khi đo được, dữ liệu đã lộ ra phân bố lớp thật — giá trị lớn hơn nhiều so với câu hỏi ban đầu.

---

## 9. Tóm tắt

| Câu hỏi | Trả lời |
| :--- | :--- |
| Lấy mẫu 100 dòng sai bao nhiêu? | Trung kỳ không sai lệch (<1 điểm), nhưng từng lần có thể sai tới **~10 điểm** |
| Seed 42 của notebook sai bao nhiêu? | **+6.52 điểm %** trên Coral, **+70% tương đối** trên Elasmobranch |
| Hậu quả? | **Đảo ngược thứ hạng 2 lớp hiếm nhất** → class weight sai 38–47% |
| Phân bố lớp thật? | **Background 55.31% · Coral 33.73% · Fish 7.80% · Other 2.21% · Elasmobranch 0.94%** |
| Class 4 có tồn tại? | **Có**, 2.21% pixel |
| Lớp nào hiếm nhất? | **Elasmobranch (0.94%)**, khan hiếp 59× so với Background |
| Có cần sửa cell 20? | **Có** — 1 dòng, thêm thông tin chính xác, bỏ nguy cơ hẹn giờ |
| Có cần đo trước khi kết luận? | **Luôn luôn** — 3 trong 8 nhận định của tôi đã bị số liệu thực nghiệm bác bỏ |
