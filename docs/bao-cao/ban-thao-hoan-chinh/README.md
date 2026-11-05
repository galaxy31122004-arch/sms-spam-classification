# Bản thảo báo cáo hoàn chỉnh

Đề tài: **Phát hiện tin nhắn SMS rác bằng mạng nơ-ron MLP, LSTM và Transformer**.

Bản thảo soạn ngày 28/10/2026, cập nhật ngày 05/11/2026. Các chương chưa được ghi nhận là đã được người dùng duyệt. Chưa tạo DOCX/PDF hoặc dàn trang theo mẫu trường.

## Thay đổi trong lần cập nhật 05/11/2026

- Số liệu chính chuyển từ lượt Colab (`outputs/test_results.json`) sang lượt huấn luyện lại cục bộ (`eval/results/baseline_test.json`), sau khi cột `text` của mọi split được sinh lại bằng hàm chuẩn hóa hiện tại. Lượt Colab được giữ để đối chiếu ở mục 4.6.
- Thêm mục 2.7 và 4.7: hai bộ dữ liệu bổ sung (africa-smishing tổng hợp, Mendeley SMS thật), thí nghiệm bổ sung dữ liệu và đánh giá ngoài nguồn. Thêm tài liệu tham khảo [17], [18].
- Chương 5: website dùng Transformer, đã chuẩn hóa đầu vào giống lúc train; có kết quả kiểm thử thật (Bảng 5.2).
- Python chạy được (`eval/.venv`), nên unit test, checkpoint và website đều đã được thực thi; các ghi chú "chưa chạy được" trước đây đã được thay bằng kết quả.

## Tệp nội dung

| Tệp | Nội dung |
|---|---|
| [01-mo-dau.md](01-mo-dau.md) | Năm mục MĐ.1–MĐ.5 |
| [02-chuong-1-co-so-ly-thuyet.md](02-chuong-1-co-so-ly-thuyet.md) | Biểu diễn văn bản, MLP, BiLSTM, Transformer, loss, đánh giá |
| [03-chuong-2-du-lieu-va-tien-xu-ly.md](03-chuong-2-du-lieu-va-tien-xu-ly.md) | CSV, phân bố nhãn, tiền xử lý, train-only fit, duplicate, dữ liệu bổ sung |
| [04-chuong-3-mo-hinh-va-huan-luyen.md](04-chuong-3-mo-hinh-va-huan-luyen.md) | Sơ đồ kiến trúc, bảng cấu hình, chọn checkpoint, môi trường |
| [05-chuong-4-ket-qua-va-so-sanh.md](05-chuong-4-ket-qua-va-so-sanh.md) | Metric, ma trận, latency, đối chiếu Colab, đánh giá ngoài nguồn |
| [06-chuong-5-website-demo.md](06-chuong-5-website-demo.md) | Flask/HTML/Jinja, Transformer, đồng bộ chuẩn hóa, kiểm thử |
| [07-ket-luan.md](07-ket-luan.md) | Đối chiếu mục tiêu, hạn chế và hướng phát triển |
| [08-tai-lieu-tham-khao.md](08-tai-lieu-tham-khao.md) | 18 mục tham khảo |
| [09-phu-luc.md](09-phu-luc.md) | Cây dự án, config đầy đủ, lệnh đã chạy, mã ngắn, trạng thái test |
| [BAO-CAO-TONG-HOP.md](BAO-CAO-TONG-HOP.md) | Bản ghép đúng thứ tự, có mục lục và danh mục bảng/hình |

Các sơ đồ dùng Mermaid; công thức dùng LaTeX trong Markdown. Khi chuyển sang Word cần kết xuất sơ đồ và kiểm tra lại bảng/công thức.

## Hồ sơ biên tập

- [tong-hop-so-lieu.py](ho-so-bien-tap/tong-hop-so-lieu.py) → [tong-hop-so-lieu.json](ho-so-bien-tap/tong-hop-so-lieu.json): mọi số liệu báo cáo dẫn, sinh trực tiếp từ CSV, config và JSON kết quả.
- [kiem-thu-website.py](ho-so-bien-tap/kiem-thu-website.py) → [kiem-thu-website.json](ho-so-bien-tap/kiem-thu-website.json): kiểm thử sáu nhánh của website.
- [ghep-va-ra-soat.ps1](ho-so-bien-tap/ghep-va-ra-soat.ps1) → [ket-qua-ra-soat.txt](ho-so-bien-tap/ket-qua-ra-soat.txt): ghép chương, kiểm tra citation/caption/Markdown và so bảng Chương 4 với bản sao kết quả baseline trong `tong-hop-so-lieu.json` (chạy được ngay sau khi clone). Tệp có BOM UTF-8 để Windows PowerShell 5.1 đọc đúng tiếng Việt.
- [chup-giao-dien.py](ho-so-bien-tap/chup-giao-dien.py) → `hinh/*.png`: ảnh giao diện website (cần Microsoft Edge).

Hồ sơ kiểm tra của bản 28/10/2026 đã được xóa ngày 05/11/2026 vì dựa trên số liệu Colab cũ; xem lịch sử git nếu cần.

## Giới hạn còn lại

Một số nguồn học thuật [4], [5], [7] chưa được xác minh độc lập; giấy phép của [17] không thống nhất giữa README (MIT) và metadata (`other`). Thành viên các split chưa được chia lại theo nhóm nội dung.

Ảnh giao diện website nằm trong `hinh/`, tạo lại bằng `ho-so-bien-tap/chup-giao-dien.py` (cần Microsoft Edge).
