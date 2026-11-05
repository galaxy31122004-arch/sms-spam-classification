# PHỤ LỤC

## Phụ lục A. Cấu trúc dự án liên quan đến báo cáo

```text
AGENTS.md, CLAUDE.md              Quy tắc chung và tệp trỏ
README.md, PLAN.md                Chỉ dẫn và kế hoạch benchmark
data/
  raw/spam.csv                   SMS Spam Collection cục bộ
  raw/africa_smish_raw.jsonl     10.000 SMS tổng hợp africa-smishing [17]
  raw/mendeley_sms_phishing_5971.zip  Bộ Mendeley SMS Phishing [18]
  splits/train_base.csv          3.900 SMS UCI, train của kết quả chính
  splits/train.csv               7.806 SMS = train_base + 3.906 tin tổng hợp
  splits/val.csv                 836 SMS
  splits/test.csv                836 SMS, test UCI
  splits/test_smishing.csv       689 SMS tổng hợp, test cùng nguồn bổ sung
  splits/test_external.csv       387 SMS thật, test ngoài nguồn
  splits/augmentation_manifest.json   Thống kê lần bổ sung dữ liệu
  splits/backup_20261105_173206/      Split trước khi sinh lại cột text
  prepare_data.py                Chuẩn hóa, đọc CSV, chia nhóm
  augment_smishing.py            Sinh lại text, thêm dữ liệu tổng hợp
  build_external_test.py         Tạo test ngoài nguồn từ bộ Mendeley
  fetch_africa_smishing.py       Tải lại africa-smishing từ Hugging Face
models/
  common.py                      Biểu diễn, mạng, train, load checkpoint
  {mlp,lstm,transformer}/train.py
  {mlp,lstm,transformer}/checkpoint/   Checkpoint augmented
  transformer/demo_checkpoint/   Checkpoint Transformer baseline cho website
outputs/                         Checkpoint và kết quả lượt Colab (đối chiếu)
eval/
  evaluate.py                    Đánh giá và đo latency
  report.py                      Ghép bảng benchmark Markdown
  requirements.txt               Phụ thuộc đánh giá
  setup_env.py                   Tạo môi trường eval/.venv
  results/baseline/<kind>/       Checkpoint baseline của kết quả chính
  results/{baseline,augmented}_{test,test_smishing,test_external}.json
tests/
  test_prepare_data.py           Test chuẩn hóa, đọc CSV, chia nhóm
  test_benchmark.py              Test model, metric, latency, report
  test_augmentation.py           Test chia template, đọc zip, --train-file
app.py                           Ứng dụng Flask dùng Transformer
templates/index.html             Biểu mẫu và vùng kết quả Jinja
start_website.bat                Khởi chạy bằng eval/.venv hoặc .venv
test_model.py                    Hai SMS ví dụ chạy qua checkpoint demo
compare_demo_models.py           So các checkpoint trên 36 SMS tự soạn
docs/smishing-augmentation-log.md    Nhật ký bổ sung dữ liệu
docs/bao-cao/                    Tài liệu báo cáo
```

*Nguồn: danh mục tệp trong workspace. `eval/results/` và các thư mục checkpoint được Git bỏ qua; chúng là artifact cục bộ, tạo lại được bằng các lệnh ở Phụ lục C.*

## Phụ lục B. Cấu hình đầy đủ của artifact thực nghiệm

**Bảng PL.B.1. Toàn bộ các trường trong ba config baseline.**

| Trường | MLP | BiLSTM | Transformer |
|---|---|---|---|
| `kind` | `mlp` | `lstm` | `transformer` |
| `input_type` | `tfidf_word_char` | `token_ids` | `token_ids` |
| `vocab_size` | 15.000 | 6.856 | 6.856 |
| `max_len` | n/a | 100 | 100 |
| `embedding_dim` | n/a | 128 | 128 |
| `hidden_dim` | n/a | 128 | 128 |
| `optimizer` | Adam | Adam | AdamW |
| `learning_rate` | 0,001 | 0,001 | 0,001 |
| `batch_size` | 64 | 64 | 64 |
| `seed` | 42 | 42 | 42 |
| `epochs_requested` | 10 | 10 | 10 |
| `patience` | 3 | 3 | 3 |
| `decision_threshold` | 0,5 | 0,5 | 0,5 |
| `train_file` | `data/splits/train_base.csv` | `data/splits/train_base.csv` | `data/splits/train_base.csv` |
| `train_examples` | 3.900 | 3.900 | 3.900 |
| `val_file` | `data/splits/val.csv` | `data/splits/val.csv` | `data/splits/val.csv` |
| `checkpoint_selection` | `max_val_spam_f1_first_tie` | `max_val_spam_f1_first_tie` | `max_val_spam_f1_first_tie` |
| `best_epoch` | 8 | 8 | 4 |
| `epochs_trained` | 10 | 10 | 7 |
| `train_seconds` | 8,909 | 29,752 | 104,944 |
| `val_spam_f1`, dạng tỷ lệ | 0,9541284403669725 | 0,944954128440367 | 0,9107142857142857 |
| `parameters` | 3.873.281 | 1.158.273 | 1.155.457 |

*Nguồn: `eval/results/baseline/{mlp,lstm,transformer}/config.json`. Dấu phẩy dùng cho thập phân của bảng tiếng Việt; JSON gốc dùng dấu chấm. `n/a` tương ứng `null` vì không áp dụng. Với MLP, `vocab_size` là tổng số feature TF-IDF. Dropout, số tầng và head không có trường riêng trong config; các giá trị đó được dẫn từ mã tại Chương 3.*

## Phụ lục C. Các lệnh đã thực thi

Mọi lệnh dưới đây đã được chạy thành công trên máy nêu ở mục 3.8, từ thư mục gốc dự án.

### C.1. Môi trường

```powershell
py -3.11 eval/setup_env.py
```

`eval/requirements.txt` gồm cả Flask cho website.

### C.2. Chuẩn bị dữ liệu bổ sung

```powershell
.\eval\.venv\Scripts\python.exe data/fetch_africa_smishing.py
.\eval\.venv\Scripts\python.exe data/augment_smishing.py --apply
.\eval\.venv\Scripts\python.exe data/build_external_test.py
```

Dữ liệu thô từ nguồn [17], [18] đã có trong `data/raw/` (`africa_smish_raw.jsonl`, `mendeley_sms_phishing_5971.zip`), nên lệnh tải đầu tiên chỉ cần khi muốn lấy lại từ Hugging Face. `augment_smishing.py` chạy ở chế độ xem thử nếu không có `--apply`, sao lưu split trước khi ghi và từ chối chạy lần hai khi đã có manifest. Chạy lại `build_external_test.py` từ tệp zip trong repo cho ra `test_external.csv` giống hệt từng byte.

### C.3. Huấn luyện và đánh giá

```powershell
.\eval\.venv\Scripts\python.exe models/mlp/train.py --train-file data/splits/train_base.csv --output-dir eval/results/baseline/mlp
.\eval\.venv\Scripts\python.exe models/lstm/train.py --train-file data/splits/train_base.csv --output-dir eval/results/baseline/lstm
.\eval\.venv\Scripts\python.exe models/transformer/train.py --train-file data/splits/train_base.csv --output-dir eval/results/baseline/transformer
.\eval\.venv\Scripts\python.exe eval/evaluate.py --test data/splits/test.csv --checkpoint-root eval/results/baseline --output eval/results/baseline_test.json
.\eval\.venv\Scripts\python.exe eval/report.py --input eval/results/baseline_test.json --output eval/results/baseline_test.md
```

Lượt augmented dùng cùng lệnh train không có `--train-file` (mặc định `train.csv`, lưu vào `models/<kind>/checkpoint`) và `--checkpoint-root models` khi đánh giá. Các tập `test_smishing.csv`, `test_external.csv` được chấm bằng cách đổi `--test`.

### C.4. Unit test, website và kiểm tra báo cáo

```powershell
.\eval\.venv\Scripts\python.exe -m unittest discover -s tests -t .
.\eval\.venv\Scripts\python.exe app.py
.\eval\.venv\Scripts\python.exe docs/bao-cao/ban-thao-hoan-chinh/ho-so-bien-tap/kiem-thu-website.py
.\eval\.venv\Scripts\python.exe docs/bao-cao/ban-thao-hoan-chinh/ho-so-bien-tap/tong-hop-so-lieu.py
powershell -File docs/bao-cao/ban-thao-hoan-chinh/ho-so-bien-tap/ghep-va-ra-soat.ps1
```

`eval/evaluate.py` chỉ tìm checkpoint trong `--checkpoint-root` đã chỉ định; muốn chấm checkpoint Colab thì truyền `--checkpoint-root outputs`.

## Phụ lục D. Các đoạn mã ngắn giải thích quyết định quan trọng

### D.1. Chọn checkpoint chỉ khi validation F1 tăng

```python
if val_f1 > best_f1:
    best_f1, stale = float(val_f1), 0
    best_state = {key: value.detach().cpu().clone()
                  for key, value in model.state_dict().items()}
    config["best_epoch"] = epoch
```

*Nguồn: trích `train` trong `models/common.py`, chỉ ngắt dòng để trình bày. Toán tử `>` giữ epoch đầu khi bằng điểm; `clone()` lưu bản trọng số độc lập với các cập nhật sau.*

### D.2. Pooling loại vị trí đệm trong Transformer

```python
encoded = self.encoder(embedded + self.position(positions),
                       src_key_padding_mask=~mask)
pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / (
    mask.sum(dim=1).clamp(min=1).unsqueeze(-1))
```

*Nguồn: `SequenceClassifier.forward` trong `models/common.py`, thay đổi xuống dòng để dễ đọc. Mask trong encoder và mask trong pooling có hai vai trò riêng; `clamp(min=1)` tránh chia cho 0.*

### D.3. Website chuẩn hóa đầu vào và lấy ngưỡng từ config

```python
features = transform_texts(MODEL_KIND, encoder,
                           [normalize_text(message)], config["max_len"])
with torch.inference_mode():
    probability = torch.sigmoid(model(features)).item()
threshold = config.get("decision_threshold", 0.5)
label = "SPAM" if probability >= threshold else "HAM"
```

*Nguồn: `classify_message` trong `app.py`, lược bỏ chú thích và dòng trống. `normalize_text` là cùng hàm đã sinh cột `text` của dữ liệu huấn luyện; ngưỡng trong config checkpoint demo là 0,5.*

## Phụ lục E. Kiểm thử và kiểm tra đã thực hiện

**Bảng PL.E.1. Các nhóm unit test của dự án.**

| Nhóm | Tên test tiêu biểu | Nội dung được kiểm tra | Trạng thái |
|---|---|---|---|
| Chuẩn hóa | `test_preserves_punctuation_after_urls` | Giữ dấu câu cuối URL | Đạt |
| URL trần | `test_bare_domains_become_url_tokens` | `bit.ly/...`, `t.me/...`, `*.co.uk` thành `<URL>`; không bắt nhầm `ok.me`, email | Đạt |
| Mã tiền tệ | `test_iso_currency_codes_become_money_tokens` | `NGN25000`, `KES 1,000`, `500 GHS` thành `<MONEY>` | Đạt |
| Số tiền/điện thoại | `test_separates_uk_phone_from_attached_pence_rate` | Tách điện thoại gắn khoản phí | Đạt |
| Unicode | `test_unicode_and_normalization_are_stable` | NFKC và tính ổn định khi chuẩn hóa lại | Đạt |
| Đọc CSV | `test_reads_quoted_newline_in_kaggle_message` | SMS có xuống dòng trong trường được quote | Đạt |
| Chia nhóm | `test_grouped_split_preserves_all_rows_and_prevents_overlap` | Giữ bản ghi, không trùng tập trên dữ liệu fixture | Đạt |
| Bảo vệ split | `test_existing_splits_are_protected_without_force` | Từ chối ghi đè thiếu `--force` | Đạt |
| Vocabulary | `test_vocab_is_train_only_and_padding_is_stable` | PAD/UNK và mã hóa mẫu kiểm thử | Đạt |
| Đầu ra mạng | `test_sequence_models_return_one_logit_per_message` | Shape đầu ra BiLSTM/Transformer | Đạt |
| Metric/ngưỡng | `test_accuracy_and_fixed_threshold_boundary` | Điểm 0,5 được gán lớp dương | Đạt |
| Latency | `test_latency_is_amortized_per_message_including_partial_batch` | Chia thời gian cho số mẫu, kể cả batch cuối | Đạt |
| Report | `test_report_compares_three_models_without_legacy_routes` | Bảng đủ ba mô hình và metadata | Đạt |

*Nguồn: `tests/test_prepare_data.py`, `tests/test_benchmark.py`, `tests/test_augmentation.py`. Toàn bộ 42 test chạy đạt: “Ran 42 tests … OK”. `test_augmentation.py` kiểm tra chia theo template, đọc zip Mendeley và tham số `--train-file`.*

**Bảng PL.E.2. Các kiểm tra khác đã thực hiện.**

| Phép kiểm tra | Nguồn/phương pháp | Kết quả quan sát |
|---|---|---|
| Số SMS và nhãn | `tong-hop-so-lieu.py` đếm mọi CSV | Đúng Bảng 2.1 và 2.5 |
| Trùng lặp giữa các tập | `summarize_splits` | Đúng Bảng 2.4; 84 dòng test trùng train theo `text` |
| Không trùng của tập bổ sung | So khớp `text` của hai tập test bổ sung với train/val/test | 0 dòng trùng |
| Accuracy/Precision/Recall/F1 | Tính từ TN/FP/FN/TP trong JSON | Khớp cả ba mô hình trước làm tròn |
| Bảng metric Chương 4 | `ghep-va-ra-soat.ps1` so từng hàng với `baseline_test.json` | Khớp |
| Website | `kiem-thu-website.py` và server thật | Đúng Bảng 5.2 |

*Nguồn: các script trong hồ sơ biên tập và đầu ra `tong-hop-so-lieu.json`, `kiem-thu-website.json`, `ket-qua-ra-soat.txt`.*
