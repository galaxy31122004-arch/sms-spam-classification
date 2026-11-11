# Dàn ý 17 slide (mẫu Kem – xanh ngọc, mở rộng từ deck 10 slide)

Giữ thiết kế và nội dung hiện có; tách slide dày, thêm nội dung lấy từ repo (config checkpoint,
`docs/smishing-augmentation-log.md`, `docs/kiem-thu-tin-nhan-thuc-te.md`). Không thêm số liệu mới tự bịa.

| # | Tiêu đề | Nội dung | Nguồn | Mới / từ slide cũ |
|---|---|---|---|---|
| 1 | Bìa | Như hiện tại | – | cũ 1 |
| 2 | Bài toán và mục tiêu | Phân loại HAM/SPAM; vì sao F1 SPAM (lệch lớp 724/112); 4 mục tiêu | báo cáo | tách từ cũ 2 |
| 3 | Bộ dữ liệu SMS Spam Collection | Bảng train/val/test 3.900/836/836, số HAM/SPAM; chia theo nhóm văn bản, seed 42; 84 tin test trùng train | prepare_data, báo cáo | tách từ cũ 2 |
| 4 | Hai tập đánh giá bổ sung | Ngoài nguồn: Mendeley, loại 5.035 trùng + 447 gần trùng… còn 387 (221 HAM/166 SPAM); Tổng hợp africa-smishing 689 | augmentation-log | mới |
| 5 | Chuẩn hóa văn bản | Dải bước chuẩn hóa + 2–3 ví dụ tin thật trước → sau (chạy `normalize_text`) | prepare_data.py | mở rộng từ cũ 3 |
| 6 | Ba kiến trúc | 3 thẻ MLP / BiLSTM / Transformer | common.py | cũ 3 |
| 7 | Cấu hình từng mô hình | Bảng: đầu vào, vocab (15.000 / 6.856 / 6.856), max_len, embedding, hidden, optimizer, epoch tốt nhất / đã chạy, val F1 | config.json | mới |
| 8 | Điều kiện so sánh chung | Như hiện tại | – | cũ 4 |
| 9 | Test UCI: MLP dẫn đầu | Như hiện tại | – | cũ 5 |
| 10 | Chi phí tính toán | Như hiện tại | – | cũ 6 |
| 11 | SMS thật ngoài nguồn | Như hiện tại (nền tối) | – | cũ 7 |
| 12 | Bổ sung dữ liệu tổng hợp | Bảng F1 trước → sau + FP/FN trên SMS thật; nhận xét | augmentation-log | tách từ cũ 8 |
| 13 | Kiểm thử 66 SMS tự soạn | 4 bộ tin + bảng kết quả từng bộ, báo nhầm / bỏ sót | kiem-thu doc | tách từ cũ 8 |
| 14 | Phân tích lỗi | Bảng tin lỗi + biến thể thử (bỏ số tiền 0,94 → 0,22; bỏ mã vận đơn 1,00 → 0,02…) | kiem-thu doc | mới |
| 15 | Website demo | Như hiện tại | – | cũ 9 |
| 16 | Kết luận và hạn chế | Kết quả + hạn chế | – | tách từ cũ 10 |
| 17 | Hướng phát triển và cảm ơn | Hướng phát triển + dải cảm ơn | – | tách từ cũ 10 |

Ghi chú: thí nghiệm `<CODE>` / HAM tổng hợp hôm nay không đưa vào (bạn đã bảo không cần).
Bản này lưu là v3.0.0 khi chốt; v1.0.0 (17 slide xanh–trắng) và v2.0.0 (10 slide) giữ nguyên.
