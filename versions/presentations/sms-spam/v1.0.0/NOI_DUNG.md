# Nội dung slide — sms-spam v1.0.0

- Phiên bản trước: không có (bản đầu tiên).
- Ngày: 06/11/2026. 17 slide, 16:9, theme sáng (nền trắng, nhấn xanh `2563EB`), phông Calibri.
- Nguồn số liệu: Chương 4 và Kết luận của báo cáo (`docs/bao-cao/ban-thao-hoan-chinh/05-*.md`, `07-*.md`),
  gốc là `eval/results/*.json`.
- Dựng bằng `scripts/build_deck.py` (python-pptx); ảnh xem trước trong `images/`.
  Không dùng `render_pptx.py` của axon vì bộ đó tự cắt bớt ý (dạng metrics giữ 3 ý, cards giữ 4).

## Danh sách slide

| # | Tiêu đề | Dạng |
|---|---|---|
| 1 | Phát hiện tin nhắn SMS rác bằng MLP, LSTM và Transformer | Bìa |
| 2 | Bài toán và mục tiêu | Ý chính |
| 3 | Dữ liệu và tiền xử lý | Ngăn phần |
| 4 | Bộ dữ liệu SMS Spam Collection | Ý chính |
| 5 | Chuẩn hóa văn bản dùng chung | Ý chính |
| 6 | Ba mô hình | Ngăn phần |
| 7 | Ba kiến trúc được so sánh | Bảng |
| 8 | Kết quả | Ngăn phần |
| 9 | Kết quả trên test UCI (836 SMS) | Bảng 5 chỉ số + FP/FN |
| 10 | Chi phí tính toán | Bảng |
| 11 | Dữ liệu thật ngoài nguồn: thứ hạng đảo ngược | Bảng |
| 12 | Bổ sung dữ liệu lừa đảo tổng hợp | Bảng trước → sau |
| 13 | Kiểm tra với 66 SMS tự soạn | Bảng |
| 14 | Website demo | Ngăn phần |
| 15 | Website phân loại SMS | Ý chính |
| 16 | Kết luận | Ý chính |
| 17 | Cảm ơn thầy và các bạn đã lắng nghe | Kết |

## Chưa có trong bản này

- Ảnh chụp giao diện website, sơ đồ kiến trúc, ma trận nhầm lẫn dạng hình.
