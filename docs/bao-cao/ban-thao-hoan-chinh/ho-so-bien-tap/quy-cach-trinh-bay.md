# Quy cách trình bày bản Word

Nguồn hình thức: `D:\Randombullsheet\bcaohocmaynhom20.docx` (đồ án môn học nhóm 20, Trường Đại học
Thủ Đô Hà Nội), đo ngày 06/11/2026 bằng `python-docx`. Bảng dưới ghi rõ thông số nào lấy nguyên
từ mẫu, thông số nào là lựa chọn của báo cáo này và vì sao.

## 1. Thông số đã đo ở mẫu

| Nhóm | Giá trị trong mẫu |
|---|---|
| Khổ giấy | A4 21 × 29,7 cm |
| Lề | Section 1: trái 3,05 cm; phải, trên, dưới 2,54 cm. Section 2: 2,54 cm cả bốn phía |
| Phông | Times New Roman; bìa 13 pt, thân bài 14 pt |
| Giãn dòng | `docDefaults` đặt 1,15 (276/240); đoạn không có cách trước/sau |
| Heading | Style ghi 20/16/14 pt nhưng run đè 14 pt, nên thực tế cả ba cấp đều 14 pt, không đậm, căn trái, đánh số `1.`, `2.1.`, `4.1.1.` |
| Heading 3 | Style có màu xám `434343` |
| Bảng | Style `Table Grid`, hàng đầu là tiêu đề, nền trắng |
| Chân trang | Trường `PAGE \* MERGEFORMAT`, căn giữa |
| Bìa | Toàn chữ, không khung, không logo: tên trường, tên khoa, dòng gạch ngang, loại tài liệu, đề tài, giảng viên, sinh viên, địa danh và năm |
| Hình/bảng | Không có chú thích đánh số |
| Công thức | Không có biểu thức Word nào |

## 2. Thông số áp dụng cho báo cáo này

| Mục | Giá trị dùng | Nguồn |
|---|---|---|
| Khổ giấy | A4 | Theo mẫu |
| Lề | Trái 3 cm, phải/trên/dưới 2 cm | Chuẩn báo cáo tiếng Việt; mẫu dùng 3,05/2,54 là mặc định Word, chênh không đáng kể |
| Phông thân bài | Times New Roman 14 pt, màu đen | Theo mẫu |
| Phông bìa | Times New Roman 13 pt | Theo mẫu |
| Giãn dòng | 1,5; cách đoạn sau 6 pt | Lựa chọn: 1,15 của mẫu quá dày cho báo cáo 60+ trang |
| Căn lề đoạn | Căn đều hai bên, không thụt dòng đầu | Theo mẫu |
| Heading 1/2/3 | Times New Roman 14 pt, **đậm**, màu đen, căn trái, cách trước 18/12/10 pt, cách sau 6/6/4 pt, `keep with next` | Cỡ chữ và căn lề theo mẫu; thêm đậm và bỏ màu xám vì mẫu để heading không đậm và Heading 3 màu `434343`, khó phân biệt với thân bài |
| Ngắt trang chương | Không tự ngắt | Theo mẫu và theo quy tắc của skill |
| Bảng | `Table Grid`, chữ 12 pt, hàng tiêu đề đậm căn giữa và lặp lại khi bảng dài, không hàng nào vỡ qua trang, bảng ≤ 12 hàng giữ nguyên khối | Kiểu bảng theo mẫu; các ràng buộc phân trang là bổ sung |
| Chú thích | `Bảng x.y.` / `Hình x.y.` đậm 13 pt căn giữa phía trên đối tượng; dòng `Nguồn:` nghiêng 12 pt căn giữa phía dưới | Bổ sung: mẫu không có chú thích, nhưng báo cáo này có 25 bảng và 7 hình nên cần đánh số và dẫn nguồn |
| Danh sách | Ý chính dấu `-` thụt 1,13 cm, treo 0,63 cm; ý phụ dấu `+` thụt thêm 0,63 cm | Mẫu dùng dấu `-` thường; thông số thụt lấy theo chuẩn NCKH của skill |
| Mã nguồn | Consolas 10 pt, giãn dòng đơn, thụt trái 0,75 cm | Bổ sung: mẫu không có khối mã |
| Chân trang | Trường `PAGE \* MERGEFORMAT`, căn giữa, 11 pt | Theo mẫu |
| Mục lục | Trường `TOC \o "1-3"`, cập nhật bằng Word; tiêu đề “MỤC LỤC” là đoạn thường để không tự liệt kê chính nó | Bổ sung |
| Công thức | Biểu thức Word thật (OMML), dựng từ UnicodeMath bằng `OMath.BuildUp()` | Bổ sung: mẫu không có công thức; yêu cầu là phân số phải có gạch ngang, không viết `a/b` |

## 3. Bìa

Khung và logo lấy từ `cover-template.docx` của skill (nằm trong header trang đầu); chữ lấy bố cục
của mẫu nhóm 20. Dòng “Hà Nội, năm 2026” đặt trong first-page footer để không chạm khung.

```text
TRƯỜNG ĐẠI HỌC THỦ ĐÔ HÀ NỘI          (13 pt, căn giữa)
KHOA TOÁN – CÔNG NGHỆ THÔNG TIN       (13 pt, đậm, căn giữa)
-----    -----
ĐỒ ÁN MÔN HỌC                         (13 pt, đậm, căn giữa)
ĐỀ TÀI: <tên đề tài>                  (13 pt, đậm, căn giữa)
    Giảng viên: <họ tên>              (13 pt, căn trái, thụt 2,5 cm)
    Sinh viên thực hiện: <họ tên>
    MSSV: <mã số>
Hà Nội, năm <năm>                     (13 pt, đậm, căn giữa)
```

Khoảng cách giữa các khối đặt bằng `Space Before`, không dùng dòng trống. Mẫu liệt kê ba sinh viên
trong một bảng; báo cáo này chỉ có một sinh viên nên viết thành dòng, bỏ bảng.

## 3b. Công thức

Toàn bộ 152 công thức là biểu thức Word thật, dựng theo
[quy trình trong skill](file:///C:/Users/letha/.claude/skills/document-authoring/references/cong-thuc-word.md):
chuyển LaTeX sang UnicodeMath, đánh dấu bằng bookmark rồi gọi `OMaths.BuildUp()`. Công thức trong
dòng dùng dấu chia U+2215 để không kéo giãn dòng chữ, trừ khi đã có `√`, `∑` hoặc hệ phương trình
thì giữ dạng hai tầng. Trước khi dựng, quét bản thảo tìm lệnh LaTeX chưa hỗ trợ và báo lỗi.

## 4. Những chỗ cố ý không theo mẫu

- Mẫu để heading không đậm và Heading 3 màu xám. Báo cáo này in đậm cả ba cấp và dùng màu đen để
  phân cấp rõ khi in đen trắng.
- Mẫu không đánh số hình/bảng. Báo cáo này đánh số theo chương và có dòng nguồn, vì số lượng lớn
  và nhiều bảng dẫn từ artifact cụ thể.
- Mẫu dùng giãn dòng 1,15. Báo cáo này dùng 1,5 cho dễ đọc ở độ dài 60+ trang.
