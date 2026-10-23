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
| `tier1/` | Code load checkpoint classical |
| `models/transformer/bert-tiny-sms-spam/` | Checkpoint Tier-2 đã clone |
| `models/mlp/`, `models/lstm/` | Baseline tham khảo |
| `data/raw/`, `data/processed/`, `data/splits/` | Dữ liệu gốc, sau xử lý, splits đã khóa |
| `eval/` | Script đánh giá và báo cáo |

## Scripts

| Script | Chức năng |
|---|---|
| `data/prepare_data.py` | Đọc CSV SMS Spam Collection, chuẩn hóa tin nhắn và tạo train/validation/test stratified, seed mặc định 42. |
| `eval/evaluate.py` | Đánh giá checkpoint BERT local trên test split; ghi metric và confusion matrix thành JSON. |
| `eval/setup_env.py` | Tạo môi trường Python riêng tại `eval/.venv` và cài dependency đánh giá. |

## Chạy

1. Tạo môi trường đánh giá: `python eval/setup_env.py`.
2. Đặt file Kaggle `spam.csv` tại `data/raw/spam.csv` (hoặc truyền đường dẫn bằng `--input`).
3. Tạo split: `python data/prepare_data.py`. Split hiện có không bị ghi đè trừ khi chủ động truyền `--force`.
4. Đánh giá bằng interpreter riêng: `eval/.venv/Scripts/python.exe eval/evaluate.py` trên Windows hoặc `eval/.venv/bin/python eval/evaluate.py` trên macOS/Linux.

Nếu checkpoint không khai báo nhãn `spam` trong `id2label`, script dừng và yêu cầu xác nhận thứ tự nhãn qua `--spam-label-id 0` hoặc `--spam-label-id 1`. Báo cáo mặc định được ghi ở `eval/results/transformer.json`.

Xem [PLAN.md](PLAN.md) để biết phạm vi và các bước tiếp theo.
