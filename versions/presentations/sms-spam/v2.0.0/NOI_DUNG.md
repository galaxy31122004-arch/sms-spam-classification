# Nội dung slide v2.0.0 – Phát hiện SMS rác (17 slide)

- Bản trước: v1.0.0 (17 slide, PPTX xanh–trắng dựng bằng python-pptx).
- Thay đổi: dựng lại trên claude.ai Slides, mẫu "Kem – xanh ngọc"; thêm slide tập đánh giá bổ sung, ví dụ chuẩn hóa, cấu hình từng mô hình, kiểm thử 66 SMS theo bộ, phân tích lỗi.
- Bản nháp 10 slide trung gian chỉ còn trong lịch sử artifact (phiên bản 18), không xuất PPTX.
- Nguồn deck: https://claude.ai/artifact/FfyySR3KGKL6P6x3Hno91t (phiên bản 21).
- Số tự tính: 86,6% / 13,4% (724/836, 112/836); trọng số lớp ≈ 6,46 (3.377/523).

Nội dung hiển thị từng slide (trích tự động từ mã slide):

## 1. Phát hiện tin nhắn SMS rác bằng MLP, LSTM và Transformer

- TRƯỜNG ĐẠI HỌC THỦ ĐÔ HÀ NỘI
- Khoa Toán – Công nghệ thông tin · Đồ án môn học
- So sánh ba mạng nơ-ron học từ đầu trên cùng dữ liệu, cùng quy trình đánh giá
- Sinh viên thực hiện
- Ninh Thị Hạnh Chi · 223001698
- Giảng viên hướng dẫn
- Nguyễn Quốc Tuấn
- Test UCI · F1 SPAM cao nhất
- MLP 96,77%
- SMS thật ngoài nguồn · F1 SPAM
- Transformer 97,56%
- Dữ liệu
- 5.572 SMS

## 2. Bài toán và mục tiêu

- Phân loại SMS tiếng Anh thành HAM = 0 (hợp lệ) hoặc SPAM = 1 (rác). Mục tiêu là so sánh công bằng ba kiến trúc, không chạy đua điểm tuyệt đối.
- 86,6%
- tin trong test là HAM (724/836)
- 13,4%
- tin là SPAM (112/836)
- Dữ liệu lệch lớp: đoán toàn HAM đã đạt Accuracy 86,6%. Vì vậy chỉ số chính là F1 của lớp SPAM .
- Mục tiêu đồ án
- 1. Xây ba mô hình học từ đầu: MLP, BiLSTM, Transformer nhỏ
- 2. Đánh giá theo cùng split, ngưỡng và bộ chỉ số
- 3. Kiểm tra khả năng tổng quát trên dữ liệu ngoài nguồn
- 4. Minh họa bằng website phân loại tin nhắn
- Phạm vi
- Không dùng mô hình pretrained, không fine-tune BERT
- Không dùng bộ lọc luật hay mô hình cổ điển trong benchmark

## 3. Bộ dữ liệu SMS Spam Collection

- Tập
- Số SMS
- HAM
- SPAM
- Tỉ lệ
- Train
- 3.900
- 3.377
- 523
- 70%
- Validation
- 836
- 724
- 112
- 15%
- Test
- 836
- 724
- 112
- 15%
- Tổng
- 5.572
- 4.825
- 747
- 100%
- Nguồn: SMS Spam Collection (UCI / Kaggle), tin tiếng Anh thu thập khoảng 2006–2012.
- Chia phân tầng, cố định
- Giữ tỉ lệ HAM/SPAM như nhau ở ba tập; seed 42; test khóa, không dùng để chọn mô hình.
- Vai trò từng tập
- Train học tham số và vocab; validation chọn epoch tốt nhất; test chỉ dùng để báo cáo.
- Lưu ý
- 84 tin test trùng nội dung với train, nên điểm trên test có thể lạc quan.

## 4. Hai tập đánh giá bổ sung

- SMS thật ngoài nguồn (Mendeley, 2022)
- Bước lọc
- Số tin bỏ
- Trùng với UCI
- 5.035
- Gần trùng UCI (difflib ≥ 0,8)
- 447
- Quá ngắn
- Trùng nội bộ
- Lỗi mã hóa
- 5
- Còn lại: 221 HAM · 166 SPAM
- 387
- 166 SPAM gồm 102 tin lừa đảo và 64 tin rác thường. Không tin nào trùng train/val/test.
- Lừa đảo tài chính tổng hợp
- 689 SMS
- Từ bộ africa-smishing (Hugging Face): 10.000 tin sinh theo mẫu câu, 11 loại lừa đảo như giả ngân hàng, đánh cắp PIN, trúng thưởng
- Vai trò khác nhau
- Tập Mendeley đo khả năng tổng quát trên tin thật. Tập tổng hợp chỉ đo dạng lừa đảo hiện đại, vì các tin rất giống nhau.

## 5. Chuẩn hóa văn bản dùng chung

- Các bước
- chữ thường, Unicode NFKC
- link → <URL>
- số tiền → <MONEY>
- SĐT 7–15 chữ số → <PHONE>
- giữ ! $ %
- Nhãn
- Tin gốc (trích từ train)
- Sau chuẩn hóa
- SPAM
- FREE entry into our £250 weekly competition just text the word WIN to 80086 NOW. 18 T&C www.txttowin.co.uk
- free entry into our £ <MONEY> weekly competition just text the word win to 80086 now. 18 t&c <URL>
- SPAM
- Loans for any purpose even if you have Bad Credit! Tenants Welcome. Call NoWorriesLoans.com on 08717111821
- loans for any purpose even if you have bad credit! tenants welcome. call <URL> on <PHONE>
- HAM
- Hey...Great deal...Farm tour 9am to 5pm $95/pax, $50 deposit by 16 May
- hey...great deal...farm tour 9am to 5pm $ <MONEY> /pax, $ <MONEY> deposit by 16 may
- MLP
- TF-IDF từ (1–2 gram, 10.000 đặc trưng) + ký tự (3–5 gram, 5.000)
- BiLSTM, Transformer
- Token ID từ vocab 6.856 từ, chỉ học trên train; cắt hoặc đệm về 100 token
- Vì sao gộp nhãn
- Mô hình học "có link, có số tiền" thay vì học thuộc từng link hay số cụ thể

## 6. Ba kiến trúc được so sánh

- MLP
- Túi từ, bỏ qua thứ tự
- Đầu vào: TF-IDF theo từ và ký tự
- Lõi: các lớp kết nối đầy đủ
- Tham số: 3,87 triệu
- BiLSTM
- Đọc chuỗi theo hai chiều
- Đầu vào: token ID → embedding
- Lõi: LSTM hai chiều
- Tham số: 1,16 triệu
- Transformer
- Mỗi token nhìn toàn câu
- Đầu vào: embedding token + vị trí
- Lõi: 2 tầng encoder, masked mean pooling
- Tham số: 1,16 triệu
- Cả ba kết thúc bằng sigmoid cho xác suất SPAM. Vocab và TF-IDF chỉ học trên train.

## 7. Cấu hình từng mô hình

- Thông số
- MLP
- BiLSTM
- Transformer
- Đầu vào
- TF-IDF
- token ID
- token ID
- Vocab / đặc trưng
- 15.000
- 6.856
- 6.856
- max_len
- n/a
- 100
- 100
- Embedding
- n/a
- 128
- 128
- Lớp ẩn
- 256 → 128
- 128 × 2 chiều
- FFN 256, 4 head
- Optimizer
- Adam
- Adam
- AdamW
- Epoch tốt nhất / chạy
- 8 / 10
- 8 / 10
- 4 / 7
- F1 SPAM (val)
- 95,41%
- 94,50%
- 91,07%
- Cân bằng lớp
- Loss BCE có trọng số SPAM = HAM / SPAM trong train ≈ 6,46
- Chống quá khớp
- Dropout 0,4 ở đầu ra; dừng sớm khi F1 val không tăng sau 3 epoch
- Riêng Transformer
- Warmup 10% số bước rồi giảm tuyến tính; 2 tầng encoder, embedding vị trí học được

## 8. Điều kiện so sánh chung

- Nội dung
- Điều kiện
- Dữ liệu
- Cùng train, validation, test
- Lớp dương
- SPAM = 1, HAM = 0
- Chọn checkpoint
- F1 SPAM cao nhất trên validation
- Ngưỡng
- 0,5 cho cả ba mô hình
- Seed
- Huấn luyện
- Batch 64, learning rate 0,001
- Epoch
- Tối đa 10, dừng sớm sau 3 epoch
- Máy đo
- CPU i7-12700H, PyTorch 2.15 (CPU)
- Bộ chỉ số
- Precision, Recall, F1 lớp SPAM
- Ma trận nhầm lẫn, ROC-AUC
- Số tham số, thời gian train, latency
- Giới hạn
- So sánh pipeline, chưa tách riêng tác dụng kiến trúc; một seed

## 9. Test UCI: MLP dẫn đầu

- 836 SMS
- MLP
- BiLSTM
- Transformer
- Accuracy
- 99,16%
- 98,68%
- 98,33%
- Precision
- 100,00%
- 99,03%
- 94,55%
- Recall
- 93,75%
- 91,07%
- 92,86%
- F1 SPAM
- 96,77%
- 94,88%
- 93,69%
- ROC-AUC
- 98,13%
- 98,43%
- 99,38%
- TN / FP
- 724 / 0
- 723 / 1
- 718 / 6
- FN / TP
- 7 / 105
- 10 / 102
- 8 / 104
- MLP không chặn nhầm tin nào
- Bắt 105/112 tin rác, bỏ sót 7
- Transformer: AUC cao, F1 thấp
- Xếp hạng tốt trên mọi ngưỡng, nhưng ngưỡng 0,5 chưa hợp với phân bố điểm
- Chênh lệch nhỏ
- 1 tin rác ≈ 0,89 điểm Recall; 84 tin test trùng train

## 10. Chi phí tính toán

- Chỉ số
- MLP
- BiLSTM
- Transformer
- Số tham số
- 3.873.281
- 1.158.273
- 1.155.457
- Thời gian train (giây)
- 8,9
- 29,8
- 104,9
- Latency batch 1 (ms/tin)
- 2,56
- 1,65
- 2,26
- Latency batch 32 (ms/tin)
- 0,44
- 0,47
- 0,81
- Nhiều tham số, train nhanh
- MLP lớn nhất nhưng chỉ nhân ma trận trên vector cố định
- Xử lý theo lô rẻ hơn
- Batch 32 giảm ms/tin 2,8–5,8 lần so với batch 1
- Phạm vi đo
- Một CPU, một lần đo; Transformer chạy 7 epoch, hai mô hình kia 10

## 11. SMS thật ngoài nguồn: thứ hạng đảo ngược

- Hạng 1 · Transformer
- 97,56%
- F1 SPAM (test UCI: hạng 3)
- Hạng 2 · BiLSTM
- 94,60%
- F1 SPAM (test UCI: hạng 2)
- Hạng 3 · MLP
- 83,92%
- F1 SPAM (test UCI: hạng 1)
- 387 SMS
- FP
- FN
- Recall
- AUC
- Transformer
- 2
- 6
- 96,39%
- 99,90%
- BiLSTM
- 0
- 89,76%
- 99,24%
- MLP
- 0
- 72,29%
- 98,99%
- Khác biệt nằm ở Recall : MLP bỏ sót 46/166 tin rác. Giải thích phù hợp: MLP dựa vào cụm từ cụ thể đã gặp khi train.
- Giới hạn: tập nhỏ, tin 2015–2020, nghiêng về người dùng Ấn Độ.

## 12. Bổ sung dữ liệu lừa đảo tổng hợp

- Thêm 3.906 SMS tổng hợp vào train (F1 trước → sau)
- F1 (%)
- UCI
- Tổng hợp
- Ngoài nguồn
- MLP
- 96,8 → 95,4
- 34,3 → 100
- 83,9 → 89,0
- BiLSTM
- 94,9 → 92,5
- 47,4 → 100
- 94,6 → 93,3
- Transformer
- 93,7 → 93,5
- 38,4 → 100
- 97,6 → 95,0
- Nguồn: africa-smishing, chỉ thêm vào train; validation và test giữ nguyên để so được với số cũ.
- Số tin bỏ sót tăng
- FN
- UCI
- Ngoài nguồn
- MLP
- 7 → 9
- 46 → 32
- BiLSTM
- 10 → 14
- 17 → 21
- Transformer
- 8 → 12
- 6 → 15
- 100% trên tập tổng hợp vì test gần như cùng mẫu câu với train. Chỉ MLP tốt lên trên SMS thật: dữ liệu theo khuôn mẫu có giá trị hạn chế .

## 13. Kiểm thử với 66 SMS tự soạn

- Bộ 1 · 8 + 8
- Khóa tài khoản, hoàn thuế; HAM có OTP, giao dịch thẻ
- Bộ 2 · 10 + 10
- Giả người thân, phí hải quan; HAM có link, tiền, mã đơn
- Bộ 3 · 15 SPAM
- Trúng thưởng, đầu tư, giả thương hiệu, tuyển dụng
- Bộ 4 · 15 HAM
- OTP, giao hàng, hóa đơn, chuyến bay, đặt bàn
- Mô hình
- Bộ 1
- Bộ 2
- Bộ 3
- Bộ 4
- Tổng
- Nhầm
- Sót
- Transformer
- 14/16
- 19/20
- 14/15
- 12/15
- 59/66
- 5
- 2
- Transformer aug.
- 15/16
- 16/20
- 12/15
- 13/15
- 56/66
- 5
- 5
- BiLSTM
- 13/16
- 17/20
- 12/15
- 12/15
- 54/66
- 4
- 8
- MLP
- 13/16
- 14/20
- 10/15
- 13/15
- 50/66
- 5
- Báo nhầm HAM gần như bằng nhau (4–5 tin); khác biệt nằm ở số SPAM bị bỏ sót. Thứ hạng khớp với tập ngoài nguồn. Bộ tin nhỏ, không chọn ngẫu nhiên: chỉ dùng để tìm dạng lỗi.

## 14. Phân tích lỗi của mô hình website

- Báo nhầm HAM: thử bỏ từng phần của tin
- Tin HAM
- Điểm
- Biến thể
- Điểm mới
- FedEx: package 7741 2093
- 1,00
- Bỏ mã vận đơn
- 0,02
- Payment of $45.00 received
- 0,94
- Bỏ số tiền
- 0,22
- Table for 4… Reply CANCEL
- 0,96
- Bỏ "Reply CANCEL"
- 0,30
- Wells Fargo: purchase $62.40
- 1,00
- Bỏ số tiền
- 0,99
- Trong UCI, ký hiệu tiền kèm số và cụm "reply X to Y" gần như chỉ có ở tin rác, nên mô hình coi chúng là dấu hiệu SPAM. Tin giao dịch ngân hàng hợp lệ hầu như không có trong train.
- Lỗi chuẩn hóa
- Mã vận đơn 8 chữ số bị đổi thành <PHONE> vì luật nhận mọi dãy 7–15 chữ số
- Bỏ sót SPAM (0,01–0,03)
- "Hi mum, I dropped my phone… send me $300": lừa đảo giả người thân, không link, văn phong như tin thường
- Làm đúng đáng chú ý
- Cảnh báo đăng nhập thật của Chase: 0,00; tin giả Chase yêu cầu bấm link: 1,00

## 15. Website demo phân loại SMS

- LUỒNG XỬ LÝ MỘT TIN NHẮN
- Tin nhắn nhập vào
- Chuẩn hóa
- Mã hóa token
- Logit → sigmoid
- HAM / SPAM
- Dùng chung hàm chuẩn hóa với lúc huấn luyện · ngưỡng 0,5 · điểm hiển thị chưa hiệu chỉnh xác suất
- Vì sao Transformer?
- Tốt nhất trên SMS ngoài nguồn – gần với tin người dùng tự nhập. Đánh đổi khoảng 3 điểm F1 trên test UCI.
- Hiển thị từng bước
- Văn bản sau chuẩn hóa, token, logit, xác suất SPAM và nhãn cuối cùng.
- Đã kiểm thử
- Các nhánh GET/POST và kiểm tra đầu vào. Demo cục bộ, chưa thử chịu tải.

## 16. Kết luận và hạn chế

- Kết quả chính
- Đánh giá
- Tốt nhất
- F1 SPAM
- Test UCI
- MLP
- 96,77%
- SMS thật ngoài nguồn
- Transformer
- 97,56%
- ROC-AUC test UCI
- Transformer
- 99,38% (AUC)
- 66 SMS tự soạn
- Transformer
- 59/66 đúng
- Thứ hạng phụ thuộc dữ liệu đánh giá. MLP học cụm từ cụ thể nên mạnh trên dữ liệu cùng nguồn; Transformer tổng quát tốt hơn trên tin mới.
- Hạn chế
- Dữ liệu
- 84 tin test trùng train; tập ngoài nguồn nhỏ (387) và cũ (2015–2020)
- Thực nghiệm
- Một seed, một lần đo latency; chưa đo độ dao động
- Mô hình
- Báo nhầm tin giao dịch có số tiền; luật SĐT bắt nhầm mã đơn; sót lừa đảo không link

## 17. Hướng phát triển

- 1. Chia lại dữ liệu
- Chia theo nhóm nội dung để bỏ 84 tin trùng giữa train và test, rồi train lại
- 2. Dữ liệu thật, hiện đại
- Thu thập SMS lừa đảo và tin giao dịch hợp lệ thật, không chỉ dữ liệu sinh theo mẫu
- 3. Sửa chuẩn hóa
- Tách mã đơn, mã OTP khỏi luật nhận số điện thoại
- 4. Thực nghiệm chắc hơn
- Chạy nhiều seed, báo độ dao động; phân tích lỗi từng mẫu trên tập ngoài nguồn
- Cảm ơn thầy và các bạn đã lắng nghe
- Em xin nhận câu hỏi
