# sms-spam-classification

Benchmark công bằng 3 kiến trúc MLP / LSTM / small Transformer, tất cả train từ đầu trên cùng dữ liệu SMS spam tiếng Anh.

## Kế hoạch

Xem [PLAN.md](PLAN.md): dữ liệu, 3 model train từ đầu, metric benchmark chung, báo cáo so sánh.
Quy tắc chung cho mọi AI: [AGENTS.md](AGENTS.md).

## Chỉ dẫn file

| File / thư mục | Nội dung |
|---|---|
| [PLAN.md](PLAN.md) | Kế hoạch benchmark đầy đủ |
| [AGENTS.md](AGENTS.md) | Quy tắc chung (nguồn duy nhất) |
| `models/mlp/` | Code train MLP từ đầu |
| `models/lstm/` | Code train LSTM từ đầu |
| `models/transformer/` | Code train small Transformer từ đầu |
| `legacy/bert-tiny-sms-spam/` | Checkpoint pretrained cũ, tách khỏi thư mục model và không dùng trong benchmark |
| `data/raw/`, `data/processed/`, `data/splits/` | Dữ liệu gốc, sau xử lý, splits đã khóa |
| `eval/` | Benchmark chung 3 model + báo cáo so sánh |

`eval/results/` là artifact local được Git ignore; chạy lại `eval/evaluate.py` và `eval/report.py` để tạo số liệu cho checkpoint hiện tại.

## Scripts

| Script | Chức năng |
|---|---|
| `data/prepare_data.py` | Chuẩn hóa SMS Spam Collection, tạo train/validation/test stratified, seed 42. |
| `models/{mlp,lstm,transformer}/train.py` | Train từng model từ đầu, lưu checkpoint và config. |
| `eval/evaluate.py` | Benchmark 3 model trên cùng test split. |
| `eval/report.py` | Tạo báo cáo Markdown so sánh 3 model. |
| `eval/setup_env.py` | Tạo `eval/.venv` và cài dependency. |

## Chạy pipeline

1. Tạo môi trường: `python eval/setup_env.py`.
2. Dataset ở `data/raw/spam.csv`. Split hiện tại: 3,900 train / 836 val / 836 test. Chỉ tạo lại có chủ ý bằng `python data/prepare_data.py --force`.
3. Chạy lần lượt `eval/.venv/Scripts/python.exe models/mlp/train.py`, `models/lstm/train.py`, `models/transformer/train.py` (dùng cùng Python trong môi trường ảo).
4. Chạy `eval/.venv/Scripts/python.exe eval/evaluate.py`, rồi `eval/.venv/Scripts/python.exe eval/report.py`. Trên Linux/macOS dùng `eval/.venv/bin/python`.

Xem [PLAN.md](PLAN.md) để biết phạm vi và các bước tiếp theo.
