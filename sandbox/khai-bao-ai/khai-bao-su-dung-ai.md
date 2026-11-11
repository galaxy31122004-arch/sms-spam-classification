# BẢN KHAI BÁO SỬ DỤNG CÔNG CỤ TRÍ TUỆ NHÂN TẠO

**Đề tài:** Phát hiện tin nhắn SMS rác bằng MLP, LSTM và Transformer
**Sinh viên:** Ninh Thị Hạnh Chi – MSSV 223001698
**Giảng viên hướng dẫn:** Nguyễn Quốc Tuấn
**Học phần:** Đồ án môn học – Khoa Toán – Công nghệ thông tin, Trường Đại học Thủ đô Hà Nội
**Ngày khai báo:** …/…/2026

## 1. Công cụ AI đã sử dụng

| Công cụ | Nhà cung cấp | Hình thức sử dụng | Thời gian |
|---|---|---|---|
| Claude Code (mô hình Claude) | Anthropic | Trợ lý lập trình chạy trên máy cá nhân, đọc và sửa tệp trong thư mục đồ án | [điền: tháng … – 11/2026] |
| [điền nếu có: ChatGPT / Codex / Copilot / Gemini…] | | | |

## 2. Nội dung có sự hỗ trợ của AI

| Hạng mục | AI hỗ trợ việc gì | Phần sinh viên tự thực hiện |
|---|---|---|
| Mã nguồn huấn luyện và đánh giá | [điền: ví dụ gợi ý cấu trúc mã, sửa lỗi, viết script đánh giá chung] | [điền: chọn kiến trúc, đặt cấu hình, chạy huấn luyện, kiểm tra kết quả] |
| Dữ liệu | Viết script tải, lọc trùng tập ngoài nguồn và tập tổng hợp | Chọn nguồn dữ liệu, duyệt quy tắc lọc, kiểm tra số liệu |
| Kiểm thử thủ công | Soạn một phần tin nhắn thử (bộ 1, 2, 4), chạy và tổng hợp điểm | Cung cấp bộ tin 3 và bộ 10 tin HAM; đánh giá kết quả |
| Báo cáo Word | Định dạng tài liệu, dựng công thức dạng Word Equation, rà soát câu chữ | Viết và duyệt nội dung, trang bìa tự chỉnh sửa |
| Slide thuyết trình | Dựng slide theo dàn ý, đề xuất bố cục, viết kịch bản thuyết trình | Duyệt dàn ý, yêu cầu chỉnh sửa bố cục, chốt nội dung |
| Website demo | [điền: ví dụ hỗ trợ viết mã Flask] | [điền] |
| Thí nghiệm bổ sung (không đưa vào kết quả) | Thử token &lt;CODE&gt; và tin HAM giao dịch tổng hợp | Quyết định không áp dụng |

## 3. Cách kiểm soát kết quả do AI tạo ra

- Mọi số liệu trong báo cáo và slide lấy từ tệp kết quả do chính mã của đồ án sinh ra (`eval/results/`, các tệp cấu hình checkpoint), không lấy số do AI tự nêu.
- Các số tự tính (ví dụ tỉ lệ 86,6% / 13,4%, trọng số lớp ≈ 6,46) được ghi rõ cách tính.
- Mã do AI viết được chạy lại trên máy và kiểm tra đầu ra trước khi sử dụng.
- Không dùng AI để tạo dữ liệu đánh giá chính; tập test UCI được khóa, không chỉnh sửa.
- Mô hình không dùng checkpoint pretrained; cả ba mô hình được huấn luyện từ đầu.

## 4. Cam kết

Em xin cam kết đã khai báo trung thực việc sử dụng công cụ AI trong đồ án. Em hiểu và chịu trách nhiệm về toàn bộ
nội dung, mã nguồn và kết quả được nộp, và có thể giải thích các phần này khi được hỏi.

Hà Nội, ngày … tháng … năm 2026
**Sinh viên ký tên**

Ninh Thị Hạnh Chi
