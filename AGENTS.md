# sms-spam-classification — hướng dẫn chung cho mọi AI

File này là **nguồn quy tắc duy nhất** cho Claude, Codex, Cursor, Gemini, Copilot.
`CLAUDE.md`, `GEMINI.md` chỉ trỏ về đây. Sửa quy tắc thì sửa ở file này.

## Bối cảnh

- Mục tiêu: train 3 kiến trúc MLP / LSTM / Transformer trên cùng dữ liệu SMS spam tiếng Anh, rồi benchmark và so sánh công bằng.
- Không chạy đua hiệu suất tuyệt đối; ưu tiên setup so sánh được, tái lập được.
- Transformer train từ đầu (small encoder), không dùng checkpoint pretrained trong `legacy/bert-tiny-sms-spam/`.
- Tier-1 classical + cascade đã bỏ khỏi mã và số liệu benchmark.
- Ngôn ngữ: tài liệu tiếng Việt; code, comment, log, commit message tiếng Anh.

## Cấu trúc

```
AGENTS.md                 quy tắc chung (file này)
PLAN.md                   kế hoạch benchmark đầy đủ
README.md                 chỉ dẫn file
models/mlp/               code train MLP từ đầu
models/lstm/              code train LSTM từ đầu
models/transformer/       code train Transformer nhỏ từ đầu
data/raw/ data/processed/ data/splits/  dữ liệu
eval/                     benchmark chung 3 model, report so sánh
```

## Quy tắc

1. Train cả 3 model (MLP, LSTM, Transformer) từ đầu trên cùng `data/splits/train.csv`, tune trên `val.csv`.
2. Mọi model đánh giá trên cùng `data/splits/test.csv` đã khóa — không trộn train vào test, không ghi đè split khi chưa được yêu cầu.
3. Thống nhất ở bước chuẩn hóa văn bản: lowercase, chuẩn hóa unicode, thay URL -> `<URL>`, SĐT -> `<PHONE>`, số tiền -> `<MONEY>`, giữ dấu câu (!, $, %). Vector hóa khác theo kiến trúc: MLP dùng TF-IDF từ và ký tự; LSTM/Transformer dùng token ID từ vocab chỉ học trên train.
4. Metric benchmark chung cho cả 3 model: spam Precision / Recall / F1, Confusion Matrix, ROC-AUC, latency trung bình, số tham số / thời gian train.
5. Report so sánh đặt 3 model cạnh nhau trên cùng một bảng, ghi rõ kiểu đầu vào, vocab/feature, max_len, embedding, optimizer, epoch, seed và ngưỡng quyết định; trường không áp dụng ghi `n/a`.
6. Dùng chung ngưỡng quyết định 0.5 khi chọn checkpoint trên validation và đánh giá test. Đây là lựa chọn cố định cho phép so sánh này, không phải ngưỡng đã tối ưu F1 cho từng model.
7. Không dùng checkpoint pretrained, không fine-tune BERT-tiny, không dùng Tier-1/cascade trong số liệu benchmark.
8. Không tự commit/push khi chưa được yêu cầu rõ ràng.
9. Sửa xong chạy validation hẹp nhất đủ tin cậy, báo kết quả trung thực kèm output.
