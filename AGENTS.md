# sms-spam-classification — hướng dẫn chung cho mọi AI

File này là **nguồn quy tắc duy nhất** cho Claude, Codex, Cursor, Gemini, Copilot.
`CLAUDE.md`, `GEMINI.md` chỉ trỏ về đây. Sửa quy tắc thì sửa ở file này.

## Bối cảnh

- Phát hiện SMS spam tiếng Anh: Tier-1 classical lọc nhanh + Tier-2 Transformer.
- Model chính đã chốt: Transformer `mrm8488/bert-tiny-finetuned-sms-spam-detection` (đã clone local).
- MLP/LSTM chỉ là baseline tham khảo, không triển khai chính.
- Ngôn ngữ: tài liệu tiếng Việt; code, comment, log, commit message tiếng Anh.

## Cấu trúc

```
AGENTS.md                 quy tắc chung (file này)
PLAN.md                   kế hoạch đầy đủ
README.md                 chỉ dẫn file
tier1_rules.yaml          ngưỡng + rule ép xuống Tier-2
tier1/                    code load checkpoint classical
models/transformer/bert-tiny-sms-spam/  checkpoint Tier-2 đã clone
models/mlp/ models/lstm/  baseline tham khảo
data/raw/ data/processed/ data/splits/  dữ liệu
eval/                     evaluate.py, report.py
```

## Quy tắc

1. Tier-1 classical (`jngb-labs/sms-spam-classical`) chỉ là pre-filter, không thay thế Transformer.
2. Rule trong `tier1_rules.yaml` chỉ quyết định ca nào xuống Tier-2, không tự kết luận spam/ham.
3. Test set trong `data/splits/` đã khóa — mọi model dùng chung, không trộn train vào test.
4. Không fine-tune lại checkpoint BERT-tiny trừ khi plan yêu cầu số liệu riêng.
5. Không tự commit/push khi chưa được yêu cầu rõ ràng.
6. Sửa xong chạy validation hẹp nhất đủ tin cậy, báo kết quả trung thực kèm output.
