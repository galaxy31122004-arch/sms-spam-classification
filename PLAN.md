# Plan: SMS Spam Detection (Classical Tier-1 + Transformer)

## 0. Mục tiêu (đã chốt)
- Model chính: Transformer. MLP/LSTM chỉ giữ làm baseline so sánh.
- Tier-1 classical chỉ làm pre-filter, không thay thế model chính.
- Ngôn ngữ: tiếng Anh. Dataset chính: SMS Spam Collection + bổ sung phishing mới.

## 1. Dữ liệu
1. Tải SMS Spam Collection (5574 mẫu, nhãn ham/spam).
2. Bổ sung SmishTank / PhishTank / mẫu mới (fake delivery, job scam, crypto).
3. Chuẩn hóa nhãn về `ham / spam`.
4. Chia stratified: train 70 / val 15 / test 15. Test set khóa lại, mọi model dùng chung.

## 2. Tiền xử lý chung
- Lowercase, chuẩn hóa unicode.
- Thay thế: URL -> `<URL>`, SĐT -> `<PHONE>`, số tiền -> `<MONEY>`.
- Giữ dấu câu (!, $, %) vì có tín hiệu spam.
- Không stemming mạnh với nhánh Transformer.

Verify: script kiểm tra 100 mẫu sau normalize bằng mắt.

## 3. Tier-1: Classical filter
- Checkpoint: `jngb-labs/sms-spam-classical` (TF-IDF word + char + LinearSVC).
- Không fine-tune checkpoint này. Nếu cần threshold riêng thì fit trên train split bằng `decision_function`.
- Ngưỡng đề xuất:
  - score > 1.0 -> spam chắc chắn
  - score < -1.0 -> ham chắc chắn
  - còn lại -> chuyển Tier-2
- Rule ép xuống Tier-2 (xem `tier1_rules.yaml`):
  - chứa URL lạ + Tier-1 bảo ham -> vẫn chuyển Tier-2
  - chứa từ khóa trúng thưởng / vay / job offer + số điện thoại -> chuyển Tier-2

Verify: đo precision/recall Tier-1 độc lập, tỉ lệ lọt Tier-2 khoảng 10-20%.

## 4. Baseline 1: MLP (tham khảo)
- Input: TF-IDF word 1-2gram (max 10k) + char 3-5gram (max 5k).
- Kiến trúc: Dense 256 + ReLU + Dropout 0.4 -> Dense 128 -> Sigmoid.
- Loss: BCE với class_weight. Adam lr 1e-3. Early stopping theo F1 val.
- Chỉ dùng để so sánh, không triển khai chính.

## 5. Baseline 2: LSTM (tham khảo)
- Embedding 128 (học từ đầu) hoặc GloVe 100d.
- BiLSTM 128 -> Dropout 0.4 -> Dense 64 -> Sigmoid.
- max_len 100. Adam lr 1e-3, early stopping F1 val.
- Chỉ dùng để so sánh, không triển khai chính.

## 6. Model chính: Transformer (đã chốt)
- Checkpoint: `mrm8488/bert-tiny-finetuned-sms-spam-detection` (BERT-tiny, đã clone sẵn).
- Đường dẫn local: `models/transformer/bert-tiny-sms-spam/` (config, safetensors, vocab, tokenizer).
- Dùng trực tiếp cho Tier-2, không train lại. Chỉ fine-tune thêm nếu cần số liệu riêng.
- Inference: pipeline `text-classification`, max_length 128.

## 7. Đánh giá
- Transformer: Precision / Recall / F1 spam, Confusion Matrix, ROC-AUC.
- MLP/LSTM chỉ đo F1 tham khảo trên cùng test set.
- Cascade: F1 cuối, false positive rate ham, latency trung bình, tỉ lệ gọi Tier-2.
- Target: F1 spam > 0.95, ham chặn nhầm < 1%.

## 8. Báo cáo
- Bảng 1: baseline classical và MLP/LSTM tham khảo.
- Bảng 2: Transformer chạy độc lập (không qua lọc).
- Bảng 3: cascade Tier-1 + Transformer.
- Ghi rõ Tier-1 là phụ trợ, không dùng để thay số liệu model chính.

## 9. Cấu trúc thư mục
```
data/raw/          SMS Spam Collection + phishing bổ sung
data/processed/    dữ liệu sau normalize
data/splits/       train/val/test đã khóa
data/prepare_data.py  chuẩn hóa CSV và tạo split stratified
tier1/             code load jngb-labs + áp thresholds/rules
tier1_rules.yaml   ngưỡng + rule ép xuống Tier-2
models/mlp/        baseline tham khảo
models/lstm/       baseline tham khảo
models/transformer/bert-tiny-sms-spam/  checkpoint Tier-2 đã clone
eval/              evaluate.py, report.py
PLAN.md            kế hoạch này
README.md          chỉ dẫn file
```

## 10. Scripts hiện có

- `data/prepare_data.py`: đọc CSV Kaggle (`v1`/`v2`) hoặc CSV tương đương, chuẩn hóa Unicode, URL, số điện thoại và số tiền; tạo `train.csv`, `val.csv`, `test.csv` theo tỷ lệ 70/15/15 với seed mặc định 42. Script từ chối ghi đè split đã có trừ khi truyền `--force`. Chạy `python data/prepare_data.py --input data/raw/spam.csv`.
- `eval/evaluate.py`: chạy checkpoint BERT local trên test split, tính precision/recall/F1 spam, accuracy, ROC-AUC và confusion matrix; lưu JSON ở `eval/results/transformer.json`. Chạy `python eval/evaluate.py`.
- `eval/requirements.txt` và `eval/setup_env.py`: dependency đánh giá được giữ riêng; script tạo `eval/.venv` và cài thư viện vào môi trường đó. Chạy `python eval/setup_env.py`.

Các split chỉ được tạo khi có file dữ liệu đầu vào. Checkpoint hiện không khai báo rõ `id2label`; trước khi đánh giá cần xác nhận nhãn spam bằng `--spam-label-id` nếu script không tự nhận diện được. Không có kết quả metric nào được giả định trước khi chạy trên test split.
