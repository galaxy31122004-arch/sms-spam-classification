---
type: academic
tone: clean
---
# Phát hiện tin nhắn SMS rác bằng MLP, LSTM và Transformer
## Ninh Thị Hạnh Chi – MSSV 223001698 · GVHD: Nguyễn Quốc Tuấn

## Bài toán và mục tiêu
- Phân loại SMS tiếng Anh thành HAM (0) hoặc SPAM (1)
- Xây dựng ba mô hình học từ đầu trên cùng dữ liệu: MLP, BiLSTM, Transformer nhỏ
- So sánh theo cùng quy trình: cùng split, cùng ngưỡng 0,5, cùng metric
- Minh họa bằng website demo phân loại tin nhắn

## Section: Dữ liệu và tiền xử lý

## Bộ dữ liệu SMS Spam Collection
- 5.572 SMS, chia train / validation / test = 3.900 / 836 / 836 (seed 42)
- Test gồm 724 HAM và 112 SPAM
- Hai tập đánh giá bổ sung:
  - 387 SMS thật ngoài nguồn (Mendeley SMS Phishing)
  - 689 SMS lừa đảo tài chính tổng hợp (africa-smishing)

## Chuẩn hóa văn bản dùng chung
- Chữ thường, chuẩn hóa Unicode
- URL → <URL>, số điện thoại → <PHONE>, số tiền → <MONEY>
- Giữ dấu câu ! $ %
- MLP dùng TF-IDF từ và ký tự; BiLSTM và Transformer dùng token ID từ vocab học trên train

## Section: Ba mô hình

## Ba kiến trúc được so sánh
- MLP: vector TF-IDF → các lớp kết nối đầy đủ → sigmoid
- BiLSTM: embedding → LSTM hai chiều → sigmoid
- Transformer nhỏ: embedding token và vị trí → 2 tầng encoder → masked mean pooling → sigmoid
- Chọn checkpoint theo F1 SPAM cao nhất trên validation

## Section: Kết quả

## Kết quả trên test UCI (836 SMS)
- MLP: Accuracy 99,16% · F1 SPAM 96,77% · 0 FP, 7 FN
- BiLSTM: Accuracy 98,68% · F1 SPAM 94,88% · 1 FP, 10 FN
- Transformer: Accuracy 98,33% · F1 SPAM 93,69% · 6 FP, 8 FN
- Transformer có ROC-AUC cao nhất: 99,38%

## Chi phí tính toán
- Số tham số: MLP 3,87 triệu · BiLSTM 1,16 triệu · Transformer 1,16 triệu
- Thời gian train: 8,9 s · 29,8 s · 104,9 s
- Latency batch 1: 2,56 · 1,65 · 2,26 ms/tin
- Latency batch 32: 0,44 · 0,47 · 0,81 ms/tin
- Đo trên CPU i7-12700H, một lần đo

## Dữ liệu thật ngoài nguồn: thứ hạng đảo ngược
- Transformer: F1 97,56% · bỏ sót 6/166 tin rác
- BiLSTM: F1 94,60% · bỏ sót 17/166
- MLP: F1 83,92% · bỏ sót 46/166
- F1 cao trên test cùng nguồn không bảo đảm khả năng tổng quát

## Bổ sung dữ liệu lừa đảo tổng hợp
- Thêm 3.906 SMS tổng hợp vào train
- Cả ba mô hình đạt 100% trên test tổng hợp
- F1 trên test UCI giảm ở cả ba mô hình
- Trên dữ liệu thật chỉ MLP cải thiện (83,92% → 89,04%)
- Dữ liệu theo template có giá trị hạn chế

## Kiểm tra với 66 SMS tự soạn
- Transformer baseline nhận đúng 59/66
- BiLSTM 54/66 · MLP 50/66
- Báo nhầm: tin giao dịch có số tiền ($<MONEY>), mã vận đơn bị nhận là <PHONE>
- Bỏ sót: lừa đảo giả làm người thân, không có liên kết

## Section: Website demo

## Website phân loại SMS
- Flask, dùng Transformer baseline (tốt nhất trên dữ liệu ngoài nguồn)
- Hiển thị các bước: chuẩn hóa → mã hóa token → logit → sigmoid → nhãn
- Đã kiểm thử các nhánh GET/POST và kiểm tra đầu vào

## Kết luận
- MLP tốt nhất trên test cùng nguồn; Transformer tổng quát tốt nhất trên SMS thật
- Thứ hạng mô hình phụ thuộc vào dữ liệu đánh giá
- Hạn chế: 84 tin test trùng với train, một seed, tập ngoài nguồn nhỏ
- Hướng phát triển: chia lại dữ liệu theo nhóm, thu thập SMS lừa đảo thật, sửa luật nhận số điện thoại

## Cảm ơn thầy và các bạn đã lắng nghe
