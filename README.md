# sms-spam-classification

Phát hiện SMS spam tiếng Anh: Tier-1 classical lọc nhanh + Tier-2 Transformer (BERT-tiny).

## Kế hoạch

Xem [PLAN.md](PLAN.md): dữ liệu, Tier-1, Transformer đã chốt, đánh giá, báo cáo.
Quy tắc chung cho mọi AI: [AGENTS.md](AGENTS.md).

## Chỉ dẫn file

| File / thư mục | Nội dung |
|---|---|
| [PLAN.md](PLAN.md) | Kế hoạch đầy đủ |
| [AGENTS.md](AGENTS.md) | Quy tắc chung (nguồn duy nhất) |
| [tier1_rules.yaml](tier1_rules.yaml) | Ngưỡng + rule ép ca nghi ngờ xuống Tier-2 |
| `tier1/` | Loader classifier sklearn, rule engine và cascade router |
| `models/transformer/bert-tiny-sms-spam/` | Checkpoint local tùy chọn; mặc định evaluator tải từ Hugging Face |
| `models/mlp/`, `models/lstm/` | Baseline tham khảo |
| `data/raw/`, `data/processed/`, `data/splits/` | Dữ liệu gốc, sau xử lý, splits đã khóa |
| `eval/` | Script đánh giá và báo cáo |

## Scripts

| Script | Chức năng |
|---|---|
| `data/prepare_data.py` | Đọc CSV SMS Spam Collection, chuẩn hóa tin nhắn và tạo train/validation/test stratified, seed mặc định 42. |
| `tier1/model.py`, `tier1/cascade.py`, `tier1/rules.py` | Tải classifier sklearn, tính score và áp rule điều hướng tới Tier-2. |
| `eval/evaluate.py` | Đánh giá Transformer độc lập, Tier-1 và cascade trên cùng test split. |
| `eval/report.py` | Tạo báo cáo Markdown từ kết quả JSON. |
| `eval/setup_env.py` | Tạo môi trường Python riêng tại `eval/.venv` và cài dependency đánh giá. |

## Chạy pipeline

1. Tạo môi trường đánh giá: `python eval/setup_env.py`.
2. Dataset Kaggle hiện có ở `data/raw/spam.csv`. Split hiện tại: 3,900 train / 836 val / 836 test. Chỉ tạo lại có chủ ý bằng `python data/prepare_data.py --force`.
3. Chạy đánh giá: `eval/.venv/Scripts/python.exe eval/evaluate.py` trên Windows hoặc `eval/.venv/bin/python eval/evaluate.py` trên macOS/Linux. Lần đầu cần mạng để tải Tier-1 và BERT từ Hugging Face vào cache.
4. Tạo báo cáo: `eval/.venv/Scripts/python.exe eval/report.py` trên Windows hoặc `eval/.venv/bin/python eval/report.py` trên macOS/Linux.

Label mapping: `0=HAM`, `1=SPAM`; evaluator mặc định dùng ID 1. Mapping được đối chiếu trên validation split (spam F1 0.8976 với ID 1 so với 0.0468 với ID 0). Kết quả JSON và Markdown mặc định nằm trong `eval/results/`.

Kết quả test hiện tại (836 SMS): BERT spam F1 0.8600; cascade spam F1 0.9439, ham false-positive rate 0.1%, Tier-2 rate 11.2%, latency trung bình 0.16 ms/message trên môi trường CPU hiện tại. Đây là kết quả trên tập SMS Spam Collection, không bao gồm dữ liệu phishing mới.

Xem [PLAN.md](PLAN.md) để biết phạm vi và các bước tiếp theo.
