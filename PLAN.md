# Kế hoạch benchmark SMS spam: MLP / LSTM / Transformer train từ đầu

## 0. Mục tiêu

- Train ba kiến trúc MLP, LSTM và Transformer encoder nhỏ từ đầu trên cùng bộ SMS spam tiếng Anh, rồi so sánh bằng một quy trình đánh giá chung.
- Ưu tiên phương pháp so sánh và khả năng tái lập hơn hiệu suất tối đa.
- Không dùng checkpoint pretrained, fine-tuning hoặc số liệu Tier-1/cascade.

## 1. Dữ liệu

1. SMS Spam Collection gồm 5.572 mẫu (4.825 ham, 747 spam), lưu tại `data/raw/spam.csv`.
2. Chuẩn hóa nhãn thành `ham` và `spam`.
3. Chia stratified với seed 42 theo tỷ lệ train/val/test 70/15/15: 3.900 / 836 / 836 mẫu. Cả ba model dùng cùng các file split.
4. `data/splits/test.csv` đã khóa. Không tạo lại split bằng `--force` khi chưa có yêu cầu rõ ràng.

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

- Dùng ngưỡng xác suất spam cố định 0,5 cho cả ba model khi chọn checkpoint trên validation và chấm test. Ngưỡng này không được tối ưu riêng cho từng model; vì dữ liệu lệch lớp, Precision/Recall/F1 có thể khác nếu dùng ngưỡng khác. ROC-AUC không phụ thuộc ngưỡng.
- Trên cùng `test.csv` đã khóa, báo cáo spam Precision/Recall/F1, Confusion Matrix (ham/spam), ROC-AUC, latency CPU trung bình ở batch size 1 và 32, số tham số và thời gian train.
- Latency CPU tính cả vector hóa (TF-IDF hoặc token ID) và inference, không phải thời gian inference thuần. Warmup một lượt trên tối đa 32 mẫu trước khi đo; lượt warmup không tính vào kết quả thời gian hay metric.
- Đặt ba model cạnh nhau trong một bảng. Mỗi config có cùng các trường: kiểu đầu vào, vocab/feature, `max_len`, embedding, hidden, optimizer, epoch đã chạy, epoch tốt nhất, seed và ngưỡng. Ghi `n/a` cho trường không áp dụng.

## 7. Cấu trúc thư mục

```text
data/raw/                     dữ liệu gốc
data/splits/                  train/val/test đã khóa
data/prepare_data.py          chuẩn hóa CSV, chia stratified với seed 42
models/common.py              dữ liệu, vector hóa, model và quy trình train chung
models/mlp/                   script train MLP từ đầu
models/lstm/                  script train LSTM từ đầu
models/transformer/           script train Transformer nhỏ từ đầu
legacy/bert-tiny-sms-spam/    checkpoint pretrained cũ, không nằm trong benchmark
eval/                         đánh giá chung và báo cáo so sánh
tests/                        test dữ liệu, model, metric và report
```

## 8. Script

- `data/prepare_data.py --input data/raw/spam.csv`: chuẩn hóa và chia dữ liệu; từ chối ghi đè split nếu không có `--force`.
- `models/{mlp,lstm,transformer}/train.py`: train từ đầu trên train split, chọn checkpoint theo validation, lưu trọng số và config JSON.
- `eval/evaluate.py`: nạp ba checkpoint, chấm cùng test split bằng metric chung, lưu JSON.
- `eval/report.py`: tạo bảng so sánh Markdown từ JSON.
- `eval/setup_env.py`: tạo `eval/.venv` và cài dependencies.
- `eval/results/`: artifact JSON và Markdown tạo khi chạy, được Git ignore để tránh dùng nhầm số liệu cũ.
