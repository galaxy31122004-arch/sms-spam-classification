# Kế hoạch benchmark SMS spam: MLP / LSTM / Transformer train từ đầu

## 0. Mục tiêu

- Train ba kiến trúc MLP, LSTM và Transformer encoder nhỏ từ đầu trên cùng bộ SMS spam tiếng Anh, rồi so sánh bằng một quy trình đánh giá chung.
- Ưu tiên phương pháp so sánh và khả năng tái lập hơn hiệu suất tối đa.
- Không dùng checkpoint pretrained, fine-tuning hoặc số liệu Tier-1/cascade.

## 1. Dữ liệu

1. SMS Spam Collection gồm 5.572 mẫu (4.825 ham, 747 spam), lưu tại `data/raw/spam.csv`.
2. Chuẩn hóa nhãn thành `ham` và `spam`.
3. Chia stratified với seed 42 theo tỷ lệ train/val/test 70/15/15: 3.900 / 836 / 836 mẫu. Cả ba model dùng cùng các file split.
4. Thành viên `data/splits/test.csv` đã khóa. Không tạo lại split bằng `--force` khi chưa có yêu cầu rõ ràng.
5. Ngày 05/11/2026 (đã duyệt), cột `text` của train/val/test được sinh lại từ `raw_text` bằng `normalize_text` hiện tại; thành viên và nhãn không đổi. Bản trước đó: `data/splits/backup_20261105_173206/`.
6. Split vẫn còn duplicate giữa các tập (84 dòng test trùng train theo `text`). Phương án gom nhóm theo văn bản sau chuẩn hóa đã có trong `prepare_data.py` (`--dry-run` để xem), chưa áp dụng; cần được duyệt trước khi thay split.
7. Dữ liệu bổ sung (`data/augment_smishing.py`, `data/build_external_test.py`):
   - `train_base.csv`: 3.900 SMS UCI — train của benchmark chính.
   - `train.csv`: `train_base` + 3.906 SMS tổng hợp africa-smishing — train của thí nghiệm augmented.
   - `test_smishing.csv`: 689 SMS africa-smishing, chia theo template, không trùng train.
   - `test_external.csv`: 387 SMS thật từ Mendeley SMS Phishing, loại mọi tin trùng/gần trùng UCI.
   - Nguồn thô trong `data/raw/`; tải lại africa-smishing bằng `data/fetch_africa_smishing.py`.

## 2. Chuẩn hóa và vector hóa

- Chuẩn hóa văn bản chung: lowercase, Unicode NFKC, thay URL bằng `<URL>`, số điện thoại bằng `<PHONE>`, số tiền bằng `<MONEY>`; giữ dấu `!`, `$`, `%`.
- Vector hóa phù hợp với từng kiến trúc: MLP dùng TF-IDF từ 1-2gram (tối đa 10.000 feature) cộng ký tự 3-5gram (tối đa 5.000 feature). LSTM và Transformer dùng cùng bộ token ID, vocab tối đa 10.000 mục và `max_len=100`.
- Chỉ fit TF-IDF và xây vocab trên train split. Validation và test chỉ dùng bộ biến đổi đã fit.
- Ghi rõ kiểu đầu vào, số feature/vocab và `max_len` trong cấu hình của từng model. Sự khác nhau về vector hóa là một phần của kiến trúc, không phải bước chuẩn hóa chung.

Kiểm tra trực quan 100 mẫu sau chuẩn hóa khi thay đổi quy trình dữ liệu.

## 3. MLP

- Đầu vào TF-IDF như mục 2; Dense 256 + ReLU + Dropout 0,4 -> Dense 128 + ReLU + Dropout 0,4 -> một logit spam.
- BCEWithLogitsLoss có trọng số lớp; Adam với learning rate 1e-3; dừng sớm theo spam F1 trên validation.

## 4. LSTM

- Embedding 128 học từ đầu; BiLSTM hidden 128 -> Dropout 0,4 -> Dense 64 -> một logit spam.
- `max_len=100`; Adam với learning rate 1e-3; dừng sớm theo spam F1 trên validation.

## 5. Transformer nhỏ

- Encoder hai lớp, hidden 128, bốn attention head; embedding token và vị trí đều học từ đầu. Không dùng trọng số pretrained.
- Dùng cùng vocab và `max_len=100` với LSTM; AdamW và lịch learning rate warmup/linear decay; dừng sớm theo spam F1 trên validation.

## 6. Đánh giá chung

- Benchmark chính: train trên `train_base.csv`, chấm `test.csv`. Đánh giá phụ: chấm thêm `test_smishing.csv` và `test_external.csv`, và lặp lại với train `train.csv` (augmented) để so trước/sau.
- Dùng ngưỡng xác suất spam cố định 0,5 cho cả ba model khi chọn checkpoint trên validation và chấm test. Ngưỡng này không được tối ưu riêng cho từng model; vì dữ liệu lệch lớp, Precision/Recall/F1 có thể khác nếu dùng ngưỡng khác. ROC-AUC không phụ thuộc ngưỡng.
- Trên cùng `test.csv` đã khóa, báo cáo Accuracy, spam Precision/Recall/F1, Confusion Matrix (ham/spam), ROC-AUC, latency CPU trung bình ở batch size 1 và 32, số tham số và thời gian train.
- Latency CPU tính cả vector hóa (TF-IDF hoặc token ID) và inference, không phải thời gian inference thuần. Warmup riêng cho batch 1 và batch 32 (tối đa 32 mẫu) trước khi đo; các lượt warmup không tính vào kết quả thời gian hay metric.
- Đặt ba model cạnh nhau trong một bảng. Mỗi config có cùng các trường: kiểu đầu vào, vocab/feature, `max_len`, embedding, hidden, optimizer, learning rate, batch size, epoch yêu cầu/đã chạy/tốt nhất, patience, seed, ngưỡng và tiêu chí chọn checkpoint (`max_val_spam_f1_first_tie`: giữ epoch đầu tiên khi F1 bằng nhau). Ghi `n/a` cho trường không áp dụng hoặc artifact cũ chưa lưu.

## 7. Cấu trúc thư mục

```text
data/raw/                     dữ liệu gốc (UCI, africa-smishing, Mendeley)
data/splits/                  train_base/train/val/test + test_smishing/test_external
data/prepare_data.py          chuẩn hóa CSV, chia stratified với seed 42
data/augment_smishing.py      sinh lại text, thêm africa-smishing
data/build_external_test.py   tạo test ngoài nguồn từ Mendeley
data/fetch_africa_smishing.py tải africa-smishing từ Hugging Face
models/common.py              dữ liệu, vector hóa, model và quy trình train chung
models/mlp/                   script train MLP từ đầu
models/lstm/                  script train LSTM từ đầu
models/transformer/           script train Transformer nhỏ từ đầu
models/transformer/demo_checkpoint/  Transformer baseline dùng cho website (được commit)
outputs/                      checkpoint và kết quả lượt Colab cũ (chuẩn hóa cũ), chỉ để đối chiếu
legacy/bert-tiny-sms-spam/    checkpoint pretrained cũ, không nằm trong benchmark
eval/                         đánh giá chung và báo cáo so sánh
tests/                        test dữ liệu, model, metric và report
```

## 8. Script

- `data/prepare_data.py --input data/raw/spam.csv`: chuẩn hóa và chia dữ liệu; từ chối ghi đè split nếu không có `--force`.
- `models/{mlp,lstm,transformer}/train.py`: train từ đầu trên train split (`--train-file`, mặc định `train.csv`), chọn checkpoint theo validation, lưu trọng số và config JSON.
- `eval/evaluate.py`: nạp ba checkpoint từ `--checkpoint-root` (không tự rơi về thư mục khác), chấm cùng test split (`--test`) bằng metric chung, lưu JSON.
- `eval/report.py`: tạo bảng so sánh Markdown từ JSON.
- `eval/setup_env.py`: tạo `eval/.venv` và cài dependencies (gồm Flask cho website).
- `app.py` / `start_website.bat`: website demo dùng Transformer baseline.
- `eval/results/`: artifact JSON và Markdown tạo khi chạy, được Git ignore để tránh dùng nhầm số liệu cũ.
