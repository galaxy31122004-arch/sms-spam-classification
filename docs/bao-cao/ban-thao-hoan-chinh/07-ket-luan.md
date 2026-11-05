# KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## KL.1. Kết quả đạt được theo mục tiêu đề tài

Đề tài “Phát hiện tin nhắn SMS rác bằng mạng nơ-ron MLP, LSTM và Transformer” đã hình thành ba pipeline phân loại SMS tiếng Anh học từ đầu trên cùng bộ dữ liệu. MLP dùng TF-IDF từ/ký tự và các lớp kết nối đầy đủ; BiLSTM học embedding và mã hóa chuỗi hai chiều; Transformer nhỏ kết hợp embedding token/vị trí, hai tầng encoder và masked mean pooling. Phép so sánh nhờ đó thể hiện ba cách kết hợp biểu diễn văn bản với kiến trúc mạng, trong phạm vi cấu hình đã có.

Báo cáo nối dữ liệu, nguyên lý, triển khai và kết quả trong cùng quy ước HAM = 0, SPAM = 1. SMS Spam Collection có 5.572 SMS, chia thành 3.900 train, 836 validation và 836 test với seed 42; ba cấu hình cùng dùng ngưỡng 0,5 và chọn checkpoint theo F1 SPAM validation cao nhất. Ngoài test UCI, ba mô hình được đánh giá trên 387 SMS thật ngoài nguồn và trên 689 SMS tổng hợp kiểu lừa đảo tài chính.

**Bảng KL.1. Mức đáp ứng các mục tiêu nêu ở Mở đầu.**

| Mục tiêu | Kết quả có bằng chứng | Giới hạn còn lại |
|---|---|---|
| Tổ chức và mô tả dữ liệu | Có CSV gốc/split, số đếm nhãn, kiểm tra trùng lặp; cột `text` sinh lại bằng hàm chuẩn hóa hiện tại, có bản sao lưu | Thành viên các tập chưa được chia lại theo nhóm nội dung |
| Xây dựng ba phương án | Có mã, config, encoder và checkpoint; đã nạp và chạy lại trên máy thử nghiệm | Chưa chốt phiên bản thư viện trong requirements |
| So sánh theo quy trình chung | Có JSON trên ba tập test cùng ngưỡng, bảng metric và chi phí kèm thông tin máy đo | Một seed; một lần đo latency |
| Phân tích kết quả và giới hạn | Số học ma trận khớp JSON; có đánh giá ngoài nguồn và thí nghiệm bổ sung dữ liệu | Tập ngoài nguồn nhỏ; chưa phân tích lỗi theo từng mẫu |
| Minh họa qua ứng dụng | Website Flask dùng Transformer, chuẩn hóa đầu vào giống lúc train; sáu nhánh xử lý đã kiểm thử, có ảnh chụp giao diện thật | Chưa kiểm thử chịu tải |

*Nguồn: tổng hợp dữ liệu, mã và artifact đã phân tích ở Chương 2–5.*

Trên test UCI, MLP đạt Accuracy 99,16% và F1 SPAM 96,77%, cao nhất trong ba mô hình, với 0 FP và 7 FN. BiLSTM đạt F1 94,88% (1 FP, 10 FN); Transformer đạt F1 93,69% (6 FP, 8 FN) nhưng có ROC-AUC cao nhất, 99,38%. Trên SMS thật ngoài nguồn, thứ hạng đảo ngược: Transformer đạt F1 97,56%, BiLSTM 94,60% và MLP 83,92%, chủ yếu do MLP bỏ sót 46/166 tin rác. Kết quả cho thấy F1 cao trên test cùng nguồn không bảo đảm khả năng tổng quát, và thứ hạng phụ thuộc vào dữ liệu đánh giá.

Thí nghiệm bổ sung 3.906 SMS tổng hợp vào train giúp cả ba mô hình nhận diện đúng hoàn toàn tập test tổng hợp, nhưng làm F1 trên test UCI giảm và chỉ cải thiện MLP trên dữ liệu thật ngoài nguồn. Dữ liệu tổng hợp theo template vì vậy có giá trị hạn chế cho mô hình đã tổng quát tốt.

Website dùng Transformer baseline, mô hình tốt nhất trên dữ liệu ngoài nguồn, để minh họa quá trình chuẩn hóa, mã hóa token, tính logit, sigmoid và gán nhãn. Các nhánh GET/POST và kiểm tra đầu vào đã chạy đúng trong kiểm thử; sản phẩm vẫn là demo cục bộ, chưa đủ bằng chứng để triển khai thực tế.

## KL.2. Hạn chế của đồ án

Hạn chế đầu tiên là nội dung trùng giữa các tập: 84 dòng test trùng `text` với train, gồm 23 dòng SPAM. Sự trùng này có thể khiến kết quả trên test UCI lạc quan hơn so với SMS mới. Đánh giá ngoài nguồn bù một phần hạn chế này nhưng không thay thế việc chia lại dữ liệu.

Dữ liệu đánh giá ngoài nguồn còn nhỏ và cũ: 387 SMS, thu thập khoảng 2015–2020, nghiêng về người dùng Ấn Độ, tin HAM chủ yếu là trao đổi cá nhân ngắn. Bộ dữ liệu bổ sung huấn luyện là dữ liệu tổng hợp theo template. Kiểm tra định tính với SMS tự soạn cho thấy cả ba mô hình vẫn bỏ sót lừa đảo không có liên kết, như tin giả làm người thân xin chuyển tiền, và đôi khi gán nhầm tin giao dịch ngân hàng hợp lệ.

Hồ sơ chỉ có một seed và một lần đo latency trên một máy. Cùng số lượng vocab và cùng test không loại bỏ mọi khác biệt giữa pipeline; MLP dùng TF-IDF còn hai mạng chuỗi dùng embedding, nên chưa thể quy toàn bộ khác biệt cho riêng kiến trúc. Điểm hiển thị dưới nhãn “Xác suất SPAM” chưa được hiệu chỉnh.

## KL.3. Hướng phát triển

Ưu tiên đầu tiên là cải thiện thiết kế đánh giá: gom nhóm các văn bản trùng và phân bổ mỗi nhóm vào một tập, rồi huấn luyện và đánh giá lại cả ba mô hình. Mã chia nhóm đã có trong `data/prepare_data.py`; split hiện tại đã được sao lưu để bảo toàn nguồn gốc kết quả.

Tiếp theo là thu thập SMS lừa đảo thật, hiện đại và đa dạng hơn, gồm cả tin HAM có liên kết, số tiền hay mã OTP, để vừa bổ sung huấn luyện vừa đánh giá. Kết quả của đồ án gợi ý dữ liệu thật đa dạng có giá trị hơn dữ liệu tổng hợp theo template. Các lượt chạy nên lặp với nhiều seed và ghi điểm từng mẫu để báo cáo độ biến thiên, vẽ đường ROC và phân tích lỗi theo nội dung.

Đối với suy luận, nên ghi phiên bản hàm chuẩn hóa vào config để phát hiện sai lệch giữa checkpoint và mã, và khảo sát ngưỡng trên validation theo chi phí FP/FN khi có nhu cầu vận hành cụ thể; các thử nghiệm đó phải được báo cáo riêng, không thay quy ước benchmark 0,5. Đối với website, bước tiếp theo là kiểm tra xử lý lỗi khi thiếu artifact và đo thời gian đáp ứng.
