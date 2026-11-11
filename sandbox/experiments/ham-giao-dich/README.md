# Thí nghiệm: token `<CODE>` + tin HAM giao dịch tổng hợp (06/11/2026)

Không sửa `data/`, `models/`, `app.py`. Mọi checkpoint nằm trong `checkpoints/`.

- `normalize_v2.py`: chuẩn hóa cũ + dãy số sau từ khóa (code, order, card ending, tracking...) → `<CODE>`.
- `gen_ham.py`: 500 tin HAM giao dịch tổng hợp từ 54 mẫu (seed 7); bỏ tin trùng hoặc gần trùng (difflib ≥ 0,8) với val/test và bộ tin tự soạn.
- `prep_splits.py`: chuẩn hóa lại cột `text` của mọi split (thành viên, nhãn, `raw_text` giữ nguyên).
- `prep_mix.py`: `train_mix` = `train_base` + 500 HAM + tin africa-smishing đã có trong `train.csv`.
- `run_train.py base|ham|mix`, `run_eval.py` → `results/summary.{md,json}`.

Cấu hình: `orig` = checkpoint hiện tại; `base` = chỉ đổi chuẩn hóa; `ham` = thêm 500 HAM;
`mix` = thêm 500 HAM + smishing. Một seed (42), ngưỡng 0,5.

Kết luận: không cấu hình nào tốt hơn `orig` trên tập ngoài nguồn (dữ liệu thật). `ham` hết báo nhầm nhưng
bỏ sót nhiều spam. `mix` tốt nhất trên 76 tin tự soạn, nhưng các tin này cùng thể loại với mẫu sinh, nên con số này bị thổi cao.
Chưa đưa vào repo.
