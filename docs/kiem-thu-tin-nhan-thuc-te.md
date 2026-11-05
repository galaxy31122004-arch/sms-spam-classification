# Kiểm thử thủ công trên tin nhắn ngoài dataset

Ngày chạy: 06/11/2026. Mục đích: xem ba model phản ứng thế nào với tin nhắn kiểu hiện đại,
vốn không có trong SMS Spam Collection (dữ liệu gốc thu thập khoảng 2006–2012).

Đây **không phải** một phép đo metric. Bộ tin nhỏ, do người viết báo cáo và người dùng soạn,
không chọn ngẫu nhiên từ phân bố thật. Kết quả chỉ dùng để tìm dạng lỗi, không được báo cáo
thành Precision/Recall. Số liệu chính thức vẫn là `eval/results/*` trên `test.csv`,
`test_smishing.csv` và `test_external.csv`.

## 1. Cách chạy

```powershell
.\eval\.venv\Scripts\python.exe compare_demo_models.py all --json docs/ket-qua-kiem-thu-thu-cong.json
```

- Script: `compare_demo_models.py` (bộ tin nằm trong mã nguồn, không đọc từ split).
- Điểm từng tin lưu ở `docs/ket-qua-kiem-thu-thu-cong.json`, gồm cả văn bản sau chuẩn hóa.
- Mỗi tin đi qua đúng đường của website: `normalize_text` → `transform_texts` → sigmoid → ngưỡng 0,5.
- Không tin nào trùng với train/val/test; không tin nào được dùng để chọn checkpoint.

## 2. Các checkpoint được so

| Tên | Model | Nguồn | Ghi chú |
|---|---|---|---|
| TRF-demo | Transformer | `models/transformer/demo_checkpoint` | Chính là model website đang dùng |
| MLP | MLP | `eval/results/baseline/mlp` | Baseline, train trên `train_base.csv` |
| LSTM | BiLSTM | `eval/results/baseline/lstm` | Baseline |
| TRF-aug | Transformer | `models/transformer/checkpoint` | Train thêm africa-smishing tổng hợp |

## 3. Bốn bộ tin

| Bộ | Số tin | Nội dung | Người soạn |
|---|---:|---|---|
| 1 | 16 (8 spam / 8 ham) | Lừa đảo khóa tài khoản, hoàn thuế, trúng thưởng; ham có OTP, giao dịch thẻ | Người viết báo cáo |
| 2 | 20 (10 spam / 10 ham) | Thêm dạng giả mạo người thân, phí hải quan, đầu tư; ham có link, tiền, mã đơn | Người viết báo cáo |
| 3 | 15 (15 spam) | Trúng thưởng, đầu tư, giả mạo thương hiệu, thu hồi nợ, SEO, tuyển dụng | Người dùng cung cấp |
| 4 | 15 (15 ham) | Tin giao dịch thật: OTP, giao hàng, hóa đơn, chuyến bay, đặt bàn, link công việc | Người viết báo cáo |

Bộ 3 chỉ có spam nên chỉ đo được khả năng bắt; bộ 4 chỉ có ham nên chỉ đo được mức báo nhầm.
Hai bộ này được soạn thành cặp để nhìn cả hai loại lỗi.

## 4. Kết quả

Số tin nhận đúng trên tổng số tin của từng bộ:

| Model | Bộ 1 (16) | Bộ 2 (20) | Bộ 3 (15 spam) | Bộ 4 (15 ham) | Tổng (66) | Báo nhầm ham | Bỏ sót spam |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRF-demo | 14 | 19 | 14 | 12 | **59** | 5 | 2 |
| TRF-aug | 15 | 16 | 12 | 13 | 56 | 5 | 5 |
| LSTM | 13 | 17 | 12 | 12 | 54 | 4 | 8 |
| MLP | 13 | 14 | 10 | 13 | 50 | 5 | 11 |

Hai cột cuối đếm trên cả 66 tin, lấy trực tiếp từ `ket-qua-kiem-thu-thu-cong.json`. Bốn model báo nhầm
ham gần bằng nhau (4–5 tin); khác biệt chính nằm ở số spam bị bỏ sót.

Thứ hạng này khớp với kết quả trên `test_external.csv` (tin thật ngoài nguồn): Transformer
baseline tổng quát tốt nhất, MLP kém nhất, và bản thêm dữ liệu tổng hợp không giúp ích.

## 5. Phân tích lỗi của model website (TRF-demo)

### 5.1. Bỏ sót spam

| Tin | Điểm | Nguyên nhân đã kiểm chứng |
|---|---:|---|
| "Hi mum, I dropped my phone... send me $300" (bộ 2) | 0,03 | Không có link, không từ khóa quảng cáo; văn phong như tin người thân |
| "Hey, it's been a while! ... gossip-trends.xxx" (bộ 3) | 0,01 | Thử thay link thành `<URL>` chỉ lên 0,22; bỏ hẳn link vẫn 0,01 — lỗi do văn phong, không do link |

Cả hai đều là kiểu tấn công phi kỹ thuật dựa trên quan hệ cá nhân. Dữ liệu train không có dạng này.

Ghi chú phụ: đuôi `.xxx` không nằm trong danh sách tên miền của `normalize_text`, nên
`gossip-trends.xxx` giữ nguyên thay vì thành `<URL>`. Các tin khác có `http://` nên không bị ảnh hưởng.

### 5.2. Báo nhầm tin thường

Năm tin, đều là thông báo giao dịch hoặc xác nhận dịch vụ. Với mỗi tin, chạy thêm các biến thể để tách nguyên nhân:

| Tin | Điểm | Biến thể thử | Kết luận |
|---|---:|---|---|
| FedEx: "package 7741 2093 is out for delivery" (bộ 4) | 1,00 | Bỏ mã vận đơn → 0,02 | **Lỗi chuẩn hóa**: mã vận đơn 8 chữ số bị đổi thành `<PHONE>` vì luật nhận số điện thoại bắt mọi dãy 7–15 chữ số |
| Wells Fargo: "A purchase of $62.40 ... card ending 7781" (bộ 4) | 1,00 | Bỏ số tiền → 0,99; bỏ tên ngân hàng → 1,00 | **Lỗi model**: cấu trúc thông báo giao dịch tự nó bị coi là rác |
| "Your table for 4 ... Reply CANCEL to cancel" (bộ 4) | 0,96 | Bỏ câu "Reply CANCEL" → 0,30 | **Lỗi model**: cụm "reply X to Y" trong UCI gần như chỉ có ở tin quảng cáo |
| "Payment of $45.00 to City Water received" (bộ 1) | 0,94 | Bỏ số tiền → 0,22; viết số tiền bằng chữ → 0,01 | **Lỗi model**: token `$<MONEY>` tự nó là dấu hiệu rác mạnh |
| "Your Chase card ending 4821 was used for $23.50 at Starbucks" (bộ 1) | 1,00 | Cùng dạng với tin Wells Fargo | **Lỗi model**: thông báo giao dịch thẻ |

Nguyên nhân chung của bốn tin cuối: trong SMS Spam Collection, ký hiệu tiền tệ kèm số hầu như chỉ
xuất hiện ở tin trúng thưởng hoặc dịch vụ tính phí, nên `$<MONEY>` trở thành đặc trưng của lớp rác.
Thông báo giao dịch ngân hàng hợp lệ — dạng tin phổ biến ngày nay — gần như không có trong dữ liệu train.

### 5.3. Điểm model làm đúng đáng chú ý

Model phân biệt được hai tin gần giống nhau về từ ngữ: tin cảnh báo đăng nhập **thật** của Chase
("If this was you, no action is needed") cho 0,00, còn tin giả mạo Chase yêu cầu bấm link cho 1,00.

## 6. Kết luận

Trên 66 tin tự soạn, model website sai 7 tin (5 báo nhầm ham, 2 bỏ sót spam). Tỷ lệ này xấu hơn nhiều so với
F1 97,56% đo trên `test_external.csv`, vì bộ tin ở đây cố ý nhắm vào các dạng mà dữ liệu huấn luyện
không có. Hai nhóm lỗi còn lại:

1. **Thông báo giao dịch, giao hàng, đặt chỗ** bị báo nhầm là rác — cần dữ liệu ham hiện đại để sửa.
2. **Lừa đảo phi kỹ thuật không có link** bị bỏ sót — cần dữ liệu spam hiện đại để sửa.

Riêng lỗi mã vận đơn thành `<PHONE>` là lỗi của hàm chuẩn hóa, sửa được bằng mã mà không cần dữ liệu mới;
việc sửa kéo theo sinh lại split và huấn luyện lại cả ba model nên chưa thực hiện.

Các hướng sửa khác đã cân nhắc và loại trong phạm vi đồ án: nâng ngưỡng quyết định không cứu được vì
các tin báo nhầm đều ở mức 0,94–1,00; model pretrained bị `AGENTS.md` loại khỏi đề tài; thêm lớp luật
lọc (cascade) cũng đã bị loại khỏi phạm vi. Tìm dataset ham giao dịch tiếng Anh công khai không có bộ
nào đủ lớn, nên hướng bổ sung dữ liệu thật chưa khả thi với nguồn hiện có.
