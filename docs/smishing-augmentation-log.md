# Nhật ký bổ sung mẫu smishing vào train

Mục tiêu: thêm mẫu smishing hiện đại (kiểu bank-alert) vào `train.csv`,
giữ nguyên `val.csv`/`test.csv` đã khóa để vẫn so sánh được với số cũ.
Nguồn: `electricsheepafrica/africa-smishing-sms-phishing` (Hugging Face),
lọc tin tiếng Anh, kiểm tra trùng lặp với test trước khi trộn.

Quy tắc áp dụng: AGENTS.md (ngưỡng 0.5 cố định, train từ đầu, không pretrained).

## Tiến độ

- [x] Bước 0: chốt nguồn dataset (africa-smishing, có cột `sms_text` + `label`, loại bank-alert/account-suspension).
- [x] Bước 1: kiểm tra công cụ đọc dữ liệu (parquet/API).
  - Kết quả: venv thiếu `pandas`/`pyarrow`, không đọc được parquet trực tiếp.
    Chuyển sang tải qua `datasets-server` API (JSON, 100 dòng/trang).
  - Script tải: `C:\Users\letha\AppData\Local\Temp\fetch_smish.py`
    (lỗi format chuỗi `%` trong URL đã sửa sang `.format()`).
- [x] Bước 2: tải toàn bộ mẫu tiếng Anh + thống kê nhãn/loại.
- [x] Bước 3: kiểm tra trùng lặp với test hiện tại (loại mẫu trùng).
- [x] Bước 4: trộn vào train, train lại 3 model (baseline trên `train_base.csv`, augmented trên `train.csv`).
- [x] Bước 5: benchmark + so sánh F1 trước/sau. Demo: đã sửa lỗi thiếu `normalize_text`; `app.py`/`test_model.py` chuyển sang
  `models/mlp/checkpoint`. Thử 16 tin tự viết (`compare_demo_models.py`): MLP cũ 13/16, MLP mới 15/16.
  Cả hai cùng báo nhầm tin ham "Your Chase card ending 4821 was used for $23.50 ... Not you? Call ...".

### Bước 5: kết quả (ngưỡng 0.5, seed 42, cùng chuẩn hóa mới)

Spam F1, baseline -> augmented. Artifact: `eval/results/{baseline,augmented}_{test,test_smishing}.{json,md}`.

| Model | test.csv (836) | test_smishing.csv (689) |
|---|---|---|
| MLP | 0.968 -> 0.954 | 0.343 -> 1.000 |
| LSTM | 0.949 -> 0.925 | 0.474 -> 1.000 |
| Transformer | 0.937 -> 0.935 | 0.384 -> 1.000 |

- Test cũ giảm nhẹ ở cả 3 model, chủ yếu do recall spam giảm (MLP FN 7->9, LSTM 10->14, Transformer 8->12).
- Baseline gần như vô dụng trên smishing kiểu mới (precision 0.21-0.33, báo nhầm hàng trăm tin ham là spam).
- F1 = 1.000 trên test_smishing là quá dễ: dữ liệu tổng hợp, template khác nhau chỉ ở tên ngân hàng/tiền tệ.
  Không coi là bằng chứng tổng quát hóa; cần tin smishing thật để đánh giá.

## Test ngoài (tin thật)

- Nguồn: Mishra & Soni (2022), Mendeley Data, DOI 10.17632/f45bkkt8pr.1, CC BY 4.0. Script: `data/build_external_test.py`.
- Loại 5.035 dòng trùng UCI, 447 gần trùng UCI (difflib >= 0.8), 65 quá ngắn, 5 lỗi mã hóa, 32 trùng nội bộ.
- Còn 387 tin: 221 ham, 166 spam (102 smishing + 64 spam gốc). Trùng với train/val/test/test_smishing = 0.
- Spam F1 baseline -> augmented (`eval/results/{baseline,augmented}_test_external.*`):

| Model | F1 | Recall spam | FP / FN |
|---|---|---|---|
| MLP | 0.839 -> 0.890 | 0.723 -> 0.807 | 0/46 -> 1/32 |
| LSTM | 0.946 -> 0.932 | 0.898 -> 0.873 | 0/17 -> 0/21 |
| Transformer | 0.976 -> 0.950 | 0.964 -> 0.910 | 2/6 -> 1/15 |

- Bổ sung africa-smishing chỉ giúp MLP; LSTM và Transformer giảm. Tập nhỏ (387), chênh lệch 4-14 tin.
- Demo chuyển sang Transformer baseline: `models/transformer/demo_checkpoint` (bản sao của
  `eval/results/baseline/transformer`). 36 tin tự viết: MLP cũ 27/36, MLP mới 30/36, Transformer 33/36.

## Chi tiết từng bước

### Bước 1 (bổ sung)

- Lần tải đầu dừng ở 2.500/10.000 dòng. Script đã thêm resume (đọc số dòng đã lưu, ghi nối)
  và retry có backoff khi gặp HTTP 429/500.
- `.venv` gốc của repo hỏng (trỏ tới Python của máy khác). Dùng `eval/.venv`
  (tạo bằng `py -3.11 eval/setup_env.py`, torch 2.15.0, scikit-learn 1.7.2).

### Bước 2: tải và thống kê

- File: `C:\Users\letha\AppData\Local\Temp\africa_smish_raw.jsonl`, đủ 10.000 dòng (API báo split `train` = 10.000).
- Nhãn: 5.000 spam (`label=1`) + 5.000 ham (`label=0`, `smishing_type=none`). 100% `is_synthetic=1`.
- Loại spam: bank_alert_fake 978, mobile_money_pin_theft 907, prize_lottery_scam 628,
  account_suspension_threat 494, loan_approval_scam 407, tax_refund_scam 387,
  delivery_notification 333, airtime_reward_scam 298, government_impersonation 248,
  job_offer_scam 192, investment_scam 128.
- Cột `language` không phản ánh ngôn ngữ thật: xem mẫu của mọi nhãn (Zulu, Swahili, Yoruba, ...)
  đều là câu tiếng Anh theo template. Lọc theo `language == "English"` chỉ còn 2.104 dòng
  (1.702 spam / 402 ham) và bỏ phí phần lớn dữ liệu cùng loại.
- Trùng lặp cao do sinh từ template: 10.000 dòng còn 8.207 văn bản khác nhau sau `normalize_text`;
  nhóm English-tagged 2.104 dòng còn 1.866.

### Bước 3: trùng lặp với split hiện tại

- So khớp văn bản sau `normalize_text` (cùng hàm của `data/prepare_data.py`) với cột `text` của split:
  trùng với train = 0, val = 0, test = 0 (đo trên nhóm English-tagged).

### Vấn đề cần quyết định trước bước 4

1. Lọc tiếng Anh theo cột `language` hay lấy toàn bộ (thực tế đều là tiếng Anh)?
2. Chỉ thêm spam hay thêm cả ham? Chỉ thêm spam thì mọi tin có mã tiền tệ châu Phi
   (`ngn`, `kes`, ...), tên ngân hàng (`mtn`, `opay`, ...) hay `ref: txn...` đều là spam trong train,
   model dễ học shortcut theo thương hiệu/tiền tệ thay vì nội dung lừa đảo.
3. `normalize_text` chưa bắt các dạng của dataset này: URL không có `http`/`www`
   (`bit.ly/...`, `secure-mtn.net/...`, `t.me/...`) không thành `<URL>`; tiền dạng mã ISO
   dính số (`ngn25000`, `usd250000`) không thành `<MONEY>`. Sửa hàm chuẩn hóa thì cột `text`
   của cả val/test cũng phải sinh lại (thành viên và nhãn không đổi) — đụng split đã khóa, cần duyệt.
4. Test hiện tại là UCI SMS cũ, không có tin kiểu này: thêm dữ liệu khó làm tăng F1 trên test cũ.
   Muốn đo được hiệu quả cần thêm một tập test smishing riêng (giữ lại một phần nhóm template
   không đưa vào train, chia theo nhóm văn bản chuẩn hóa để không rò rỉ).
