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
| `legacy/bert-tiny-sms-spam/` | Checkpoint pretrained cũ (nếu còn trong checkout), tách khỏi thư mục model và không dùng trong benchmark |
| `data/raw/`, `data/processed/`, `data/splits/` | Dữ liệu gốc, sau xử lý, splits đã khóa |
| `eval/` | Benchmark chung 3 model + báo cáo so sánh |
| `app.py`, `templates/` | Web demo Flask, dùng Transformer ở `models/transformer/demo_checkpoint` |
| `compare_demo_models.py` | So các checkpoint trên 66 tin tự viết ngoài dataset (4 bộ, `all` để chạy hết) |
| [docs/kiem-thu-tin-nhan-thuc-te.md](docs/kiem-thu-tin-nhan-thuc-te.md) | Quy trình và kết quả kiểm thử thủ công, phân tích từng lỗi |
| `docs/smishing-augmentation-log.md` | Nhật ký bổ sung smishing, test ngoài và kết quả trước/sau |

`eval/results/` là artifact local được Git ignore; chạy lại `eval/evaluate.py` và `eval/report.py` để tạo số liệu cho checkpoint hiện tại.

## Scripts

| Script | Chức năng |
|---|---|
| `data/prepare_data.py` | Chuẩn hóa SMS Spam Collection, tạo train/validation/test stratified, seed 42. |
| `data/fetch_africa_smishing.py` | Tải africa-smishing từ Hugging Face vào `data/raw/africa_smish_raw.jsonl` (đã có sẵn trong repo). |
| `data/augment_smishing.py` | Chuẩn hóa lại `text` của split, thêm africa-smishing vào train và tạo `test_smishing.csv` (đã chạy một lần, có manifest + backup; từ chối chạy lại khi đã có manifest). |
| `data/build_external_test.py` | Tạo `test_external.csv` từ `data/raw/mendeley_sms_phishing_5971.zip`, loại mọi tin trùng/gần trùng UCI. |
| `models/{mlp,lstm,transformer}/train.py` | Train từng model từ đầu, lưu checkpoint và config. `--train-file` chọn file train (mặc định `train.csv`). |
| `eval/evaluate.py` | Benchmark 3 model trên một test split (`--test`). |
| `eval/report.py` | Tạo báo cáo Markdown so sánh 3 model. |
| `eval/setup_env.py` | Tạo `eval/.venv` và cài dependency. |

## Chạy pipeline

1. Tạo môi trường: `py -3.11 eval/setup_env.py` (cài cả Flask cho website). Chạy website: `start_website.bat` hoặc `eval/.venv/Scripts/python.exe app.py`.
2. Dữ liệu thô trong `data/raw/`: `spam.csv` (UCI), `africa_smish_raw.jsonl` (africa-smishing, tổng hợp), `mendeley_sms_phishing_5971.zip` (Mendeley, CC BY 4.0). Split hiện tại trong `data/splits/`:
   - `train.csv`: 7.806 tin = 3.900 UCI + 3.906 africa-smishing (3.187 ham / 719 spam, dữ liệu tổng hợp).
   - `train_base.csv`: 3.900 tin UCI gốc (baseline).
   - `val.csv` 836 / `test.csv` 836: UCI, thành viên không đổi; cột `text` sinh lại bằng `normalize_text` mới.
   - `test_smishing.csv`: 689 tin africa-smishing (562 ham / 127 spam), chia theo template, không trùng train.
   - `test_external.csv`: 387 tin thật từ Mendeley (221 ham / 166 spam), không trùng/gần trùng UCI.
   - Split trước khi bổ sung: `data/splits/backup_20261105_173206/`; thống kê: `augmentation_manifest.json`.
   Chỉ tạo lại có chủ ý bằng `python data/prepare_data.py --force`.
3. Train mặc định 10 epoch trên `train.csv` (augmented), lưu vào `models/<kind>/checkpoint`. Benchmark chính (baseline) dùng `--train-file data/splits/train_base.csv --output-dir eval/results/baseline/<kind>`. Lệnh đầy đủ ở Phụ lục C của báo cáo.
4. Chạy `eval/.venv/Scripts/python.exe eval/evaluate.py --checkpoint-root <thư mục> --test <split>`, rồi `eval/report.py`. `evaluate.py` chỉ tìm trong `<root>/<kind>/` và `<root>/<kind>/checkpoint/`; thiếu thì báo lỗi, không tự dùng checkpoint Colab trong `outputs/` (chuẩn hóa cũ). Muốn chấm checkpoint Colab thì truyền `--checkpoint-root outputs`. Trên Linux/macOS dùng `eval/.venv/bin/python`.

Split hiện tại còn tin nhắn trùng giữa train/val/test. Code chuẩn bị dữ liệu mới chia theo nhóm `text` đã chuẩn hóa, giữ toàn bộ bản ghi và cố gắng đạt 70/15/15 theo từng nhãn. Chạy `python data/prepare_data.py --dry-run` để xem duplicate và phương án chia trong bộ nhớ, không ghi file. Chỉ tạo split mới sau khi được duyệt; thay đổi test set cần train lại cả ba model để so sánh.

Các CSV trong `data/splits/` đã có cột `text` chuẩn hóa (lowercase, `<URL>`, `<PHONE>`, `<MONEY>`). Split hiện tại vẫn còn tin nhắn trùng giữa train/val/test theo kiểm tra trước đó. Benchmark/report đã hỗ trợ Accuracy, learning rate, batch size và tiêu chí chọn checkpoint; artifact cũ thiếu trường nào sẽ hiển thị `n/a`. Chạy unit test bằng `eval/.venv/Scripts/python.exe -m unittest discover -s tests -t .`.

## Kết quả hiện tại (spam F1, ngưỡng 0.5)

Baseline = train trên `train_base.csv`; augmented = train trên `train.csv` (có africa-smishing).

| Model | test.csv | test_smishing.csv | test_external.csv (tin thật) |
|---|---|---|---|
| MLP | 0.968 / 0.954 | 0.343 / 1.000 | 0.839 / 0.890 |
| LSTM | 0.949 / 0.925 | 0.474 / 1.000 | 0.946 / 0.932 |
| Transformer | 0.937 / 0.935 | 0.384 / 1.000 | **0.976** / 0.950 |

- `test_smishing` = 1.000 vì dữ liệu tổng hợp theo template, không phản ánh tổng quát hóa.
- Trên tin thật, bổ sung dữ liệu tổng hợp chỉ giúp MLP. Demo dùng Transformer baseline (tốt nhất trên `test_external`, 33/36 tin tự viết).
- Model vẫn bỏ sót kiểu lừa không có link (giả làm người thân) và đôi khi báo nhầm tin giao dịch ngân hàng thật.

Chi tiết: [docs/smishing-augmentation-log.md](docs/smishing-augmentation-log.md). Xem [PLAN.md](PLAN.md) để biết phạm vi và các bước tiếp theo.
