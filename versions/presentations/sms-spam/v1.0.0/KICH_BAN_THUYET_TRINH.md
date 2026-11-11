# Kịch bản thuyết trình — sms-spam v1.0.0

**Slide 1 – Bìa.** Em xin trình bày đề tài phát hiện tin nhắn SMS rác bằng ba mạng nơ-ron: MLP, LSTM và Transformer.

**Slide 2 – Bài toán.** Đây là bài toán phân loại nhị phân: HAM là 0, SPAM là 1. Mục tiêu không phải đạt điểm cao nhất tuyệt đối, mà là so sánh công bằng ba kiến trúc: cả ba học từ đầu, cùng dữ liệu, cùng ngưỡng 0,5 và cùng bộ chỉ số.

**Slide 4 – Dữ liệu.** Bộ SMS Spam Collection có 5.572 tin, chia cố định với seed 42. Tập test có 724 HAM và chỉ 112 SPAM, nên dữ liệu lệch lớp; vì vậy em dùng F1 của lớp SPAM làm chỉ số chính chứ không chỉ Accuracy. Ngoài ra có hai tập đánh giá thêm: 387 SMS thật từ nguồn khác và 689 SMS lừa đảo tổng hợp.

**Slide 5 – Chuẩn hóa.** Cả ba mô hình dùng chung một bước chuẩn hóa. URL, số điện thoại và số tiền được thay bằng nhãn chung, để mô hình học "có một đường link" thay vì học thuộc từng đường link. Khác nhau ở vector hóa: MLP dùng TF-IDF, hai mô hình chuỗi dùng token ID.

**Slide 7 – Kiến trúc.** MLP nhìn tin nhắn như một túi từ, không có thứ tự. BiLSTM đọc chuỗi theo hai chiều. Transformer dùng self-attention để mỗi token nhìn toàn bộ câu, rồi lấy trung bình có mặt nạ để bỏ qua phần đệm. Checkpoint được chọn theo F1 SPAM trên validation.

**Slide 9 – Kết quả test UCI.** MLP tốt nhất: F1 96,77%, không chặn nhầm tin hợp lệ nào. Transformer có F1 thấp nhất nhưng ROC-AUC cao nhất. Điều này không mâu thuẫn: AUC đo khả năng xếp hạng trên mọi ngưỡng, còn F1 đo tại đúng ngưỡng 0,5.

**Slide 10 – Chi phí.** MLP nhiều tham số nhất nhưng train nhanh nhất, vì phép nhân ma trận trên vector cố định rẻ hơn xử lý tuần tự hay attention. Transformer train lâu nhất. Số đo chỉ trên một CPU, một lần đo.

**Slide 11 – Dữ liệu ngoài nguồn.** Đây là kết quả quan trọng nhất. Trên SMS thật từ nguồn khác, thứ hạng đảo ngược: Transformer 97,56%, MLP chỉ 83,92% vì bỏ sót 46 trên 166 tin rác. Cách giải thích phù hợp là MLP phụ thuộc vào các cụm từ cụ thể đã gặp khi train.

**Slide 12 – Bổ sung dữ liệu.** Thêm dữ liệu tổng hợp giúp đạt 100% trên tập tổng hợp, nhưng làm giảm kết quả trên test UCI, và trên dữ liệu thật chỉ MLP cải thiện. Kết luận: dữ liệu sinh theo khuôn mẫu có giá trị hạn chế.

**Slide 13 – 66 tin tự soạn.** Transformer baseline đúng 59/66, cùng thứ hạng với dữ liệu ngoài nguồn. Lỗi có hai loại: tin hợp lệ có số tiền bị coi là rác vì trong dữ liệu train, ký hiệu tiền hầu như chỉ xuất hiện ở tin rác; và lừa đảo giả làm người thân không có link nên bị bỏ sót. Bộ tin nhỏ nên chỉ dùng để kiểm tra định tính.

**Slide 15 – Website.** Website Flask dùng Transformer vì người dùng nhập tin mới, không thuộc phân bố UCI. Trang hiển thị từng bước từ chuẩn hóa đến xác suất và nhãn.

**Slide 16 – Kết luận.** Thứ hạng mô hình phụ thuộc vào dữ liệu đánh giá. Hạn chế: 84 tin test trùng nội dung với train, chỉ một seed, tập ngoài nguồn nhỏ. Hướng phát triển: chia lại dữ liệu theo nhóm, thu thập tin lừa đảo thật, sửa luật nhận số điện thoại.

## Câu hỏi có thể gặp

- **Vì sao không chọn mô hình có F1 cao nhất cho website?** Vì demo xử lý tin mới; trên dữ liệu ngoài nguồn Transformer tốt nhất. Đánh đổi khoảng 3 điểm F1 trên test UCI.
- **Ngưỡng 0,5 có tối ưu không?** Không; đó là quy ước chung để so sánh công bằng, chưa tối ưu riêng cho từng mô hình.
- **Kết quả có ổn định không?** Chỉ một seed, nên chưa đo được độ biến thiên; đây là hạn chế đã nêu.
