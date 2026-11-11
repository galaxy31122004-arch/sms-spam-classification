# Kịch bản thuyết trình v2.0.0 – Phát hiện SMS rác (17 slide)

Tổng thời lượng dự kiến: **khoảng 15 phút** + hỏi đáp. Mỗi slide gồm: thời gian, lời nói (đọc gần nguyên văn),
ý cần nhấn, câu chuyển sang slide sau. Cuối file là phần chuẩn bị hỏi đáp.

Quy ước khi nói số: đọc "chín mươi sáu phẩy bảy bảy phần trăm"; với bảng lớn, chỉ đọc con số được tô hoặc in đậm,
không đọc hết bảng.

| Phần | Slide | Thời gian |
|---|---|---|
| Mở đầu, bài toán, dữ liệu | 1–4 | 3 phút |
| Phương pháp | 5–8 | 3 phút 30 |
| Kết quả | 9–11 | 3 phút |
| Độ bền và phân tích lỗi | 12–14 | 3 phút 30 |
| Demo, kết luận | 15–17 | 2 phút |

---

## Slide 1. Bìa (30 giây)

**Lời nói:**
"Em chào thầy và các bạn. Em là Ninh Thị Hạnh Chi, mã sinh viên 223001698. Hôm nay em xin trình bày đồ án
*Phát hiện tin nhắn SMS rác bằng MLP, LSTM và Transformer*, dưới sự hướng dẫn của thầy Nguyễn Quốc Tuấn.

Trước khi vào chi tiết, em tóm tắt kết quả bằng ba con số bên phải: MLP đạt F1 96,77% trên tập test cùng nguồn;
nhưng trên tin nhắn thật từ nguồn khác, Transformer lại tốt nhất với 97,56%. Toàn bộ thí nghiệm dùng 5.572 tin
nhắn. Bài trình bày sẽ giải thích vì sao thứ hạng hai mô hình đổi chỗ như vậy."

**Nhấn:** kết luận chính được nói ngay từ đầu — thứ hạng phụ thuộc dữ liệu đánh giá.

---

## Slide 2. Bài toán và mục tiêu (50 giây)

**Lời nói:**
"Bài toán là phân loại nhị phân: mỗi tin nhắn tiếng Anh được gán HAM bằng 0 nếu là tin hợp lệ, SPAM bằng 1 nếu là
tin rác.

Điểm cần lưu ý là dữ liệu lệch lớp: trong tập test, 86,6% là HAM, chỉ 13,4% là SPAM. Nghĩa là một mô hình
không học gì, cứ đoán toàn HAM, vẫn đạt Accuracy 86,6%. Vì vậy em dùng **F1 của lớp SPAM** làm chỉ số chính,
vì nó phạt cả việc bỏ sót tin rác lẫn chặn nhầm tin thường.

Đồ án có bốn mục tiêu: xây ba mô hình học từ đầu; đánh giá chúng trong cùng điều kiện; kiểm tra khả năng
tổng quát trên dữ liệu ngoài nguồn; và làm một website minh họa. Về phạm vi, em không dùng mô hình pretrained
như BERT, cũng không dùng luật lọc thủ công, để so sánh thuần ba kiến trúc mạng."

**Chuyển:** "Tiếp theo là bộ dữ liệu."

---

## Slide 3. Bộ dữ liệu SMS Spam Collection (50 giây)

**Lời nói:**
"Dữ liệu chính là SMS Spam Collection, 5.572 tin nhắn tiếng Anh thu thập khoảng 2006 đến 2012.

Em chia theo tỉ lệ 70 – 15 – 15: train 3.900 tin, validation và test mỗi tập 836 tin. Việc chia có phân tầng,
nên tỉ lệ HAM/SPAM ở ba tập như nhau, và cố định bằng seed 42 để tái lập được.

Mỗi tập có một vai trò riêng: train để học tham số và từ vựng; validation để chọn epoch tốt nhất; còn test được
khóa, chỉ dùng để báo cáo kết quả cuối cùng.

Một hạn chế em phát hiện là 84 tin trong test trùng nội dung với train. Do đó điểm trên test có thể cao hơn thực
tế, và đây là lý do em cần thêm hai tập đánh giá độc lập."

**Chuyển:** "Hai tập đó là..."

---

## Slide 4. Hai tập đánh giá bổ sung (50 giây)

**Lời nói:**
"Tập thứ nhất là SMS thật từ nguồn khác, bộ của Mishra và Soni năm 2022 trên Mendeley Data. Bộ này chứa phần lớn tin
của UCI, nên em phải lọc kỹ: bỏ 5.035 tin trùng, 447 tin gần trùng với độ giống từ 0,8 trở lên, cùng các tin
quá ngắn, trùng nội bộ hoặc lỗi mã hóa. Còn lại 387 tin, gồm 221 HAM và 166 SPAM, không tin nào trùng với dữ liệu
huấn luyện. Tập này dùng để đo khả năng tổng quát trên tin thật.

Tập thứ hai là 689 tin lừa đảo tài chính tổng hợp, lấy từ bộ africa-smishing trên Hugging Face. Các tin này sinh từ
mẫu câu, nên rất giống nhau; em chỉ dùng nó để xem mô hình có nhận ra dạng lừa đảo hiện đại hay không, không coi
là thước đo tổng quát."

**Nhấn:** hai tập có vai trò khác nhau — một tập thật, một tập tổng hợp.

**Chuyển:** "Trước khi đưa vào mô hình, mọi tin nhắn đều qua cùng một bước chuẩn hóa."

---

## Slide 5. Chuẩn hóa văn bản dùng chung (1 phút)

**Lời nói:**
"Bước chuẩn hóa gồm: chuyển chữ thường, chuẩn hóa Unicode, thay đường link bằng nhãn URL, số tiền bằng MONEY,
số điện thoại từ 7 đến 15 chữ số bằng PHONE, và giữ lại các dấu chấm than, đô la, phần trăm vì chúng hay xuất hiện
trong tin rác.

Ba ví dụ trong bảng lấy trực tiếp từ tập train và chạy qua đúng hàm của đồ án. Ví dụ tin đầu, '£250' thành ký hiệu
bảng Anh cộng MONEY, còn trang web thành URL. Mục đích là để mô hình học được *có một đường link, có một số tiền*,
thay vì học thuộc từng link hay từng con số cụ thể.

Sau bước chung này, mỗi mô hình vector hóa khác nhau: MLP dùng TF-IDF theo từ và theo cụm ký tự; BiLSTM và
Transformer dùng mã số token từ bộ từ vựng 6.856 từ, chỉ học trên train, mỗi tin cắt hoặc đệm về 100 token."

**Chuyển:** "Với đầu vào như vậy, ba kiến trúc được so sánh là..."

---

## Slide 6. Ba kiến trúc được so sánh (50 giây)

**Lời nói:**
"MLP coi tin nhắn như một túi từ, bỏ qua thứ tự; vector TF-IDF đi qua các lớp kết nối đầy đủ. Mô hình có 3,87 triệu
tham số, nhiều nhất trong ba.

BiLSTM đọc chuỗi theo cả hai chiều, từ trái sang phải và ngược lại, nên nắm được ngữ cảnh trước và sau mỗi từ.

Transformer nhỏ dùng cơ chế self-attention: mỗi token nhìn toàn bộ câu. Mô hình có hai tầng encoder, sau đó lấy
trung bình các token thật, bỏ phần đệm. Hai mô hình chuỗi có cùng khoảng 1,16 triệu tham số.

Cả ba đều kết thúc bằng hàm sigmoid, cho ra xác suất tin là SPAM."

**Chuyển:** "Thông số cụ thể của từng mô hình ở slide sau."

---

## Slide 7. Cấu hình từng mô hình (50 giây)

**Lời nói:**
"Bảng này lấy từ file cấu hình lưu cùng mỗi checkpoint, nên khớp với mô hình đã chạy. MLP dùng 15.000 đặc trưng;
hai mô hình chuỗi dùng embedding 128 chiều. Transformer dùng AdamW, có warmup 10% số bước đầu rồi giảm tuyến tính.

Về huấn luyện: vì SPAM ít, em đặt trọng số lớp SPAM trong hàm mất mát khoảng 6,46, bằng số HAM chia số SPAM trong
train, để mô hình không bỏ qua lớp ít. Em dùng dropout 0,4 và dừng sớm nếu F1 trên validation không tăng sau
3 epoch. Nhờ vậy Transformer dừng ở epoch 7, checkpoint tốt nhất ở epoch 4.

Hàng cuối là F1 trên validation: MLP cao nhất 95,41%."

---

## Slide 8. Điều kiện so sánh chung (40 giây)

**Lời nói:**
"Để so sánh công bằng, cả ba mô hình dùng chung: cùng dữ liệu, cùng seed 42, cùng batch 64 và learning rate 0,001,
tối đa 10 epoch, cùng cách chọn checkpoint theo F1 SPAM trên validation, và cùng ngưỡng quyết định 0,5.

Ngưỡng 0,5 là quy ước cố định để so sánh, không phải ngưỡng tối ưu riêng cho từng mô hình.

Bộ chỉ số gồm Precision, Recall, F1 lớp SPAM, ma trận nhầm lẫn, ROC-AUC, và chi phí: số tham số, thời gian train,
độ trễ.

Em cũng lưu ý giới hạn: vì MLP dùng TF-IDF còn hai mô hình kia dùng embedding, đây là so sánh cả *pipeline*,
chưa tách riêng tác dụng của kiến trúc; và mới chạy một seed."

**Chuyển:** "Bây giờ là kết quả."

---

## Slide 9. Test UCI: MLP dẫn đầu (1 phút 10)

**Lời nói:**
"Trên 836 tin của tập test, MLP tốt nhất với F1 96,77% và Accuracy 99,16%. Đáng chú ý, MLP không chặn nhầm tin
hợp lệ nào — precision 100% — và bắt được 105 trên 112 tin rác.

BiLSTM xếp thứ hai với F1 94,88%. Transformer thấp nhất về F1, 93,69%, nhưng lại có ROC-AUC cao nhất, 99,38%.
Hai con số này không mâu thuẫn: AUC đo khả năng xếp hạng tin rác cao hơn tin thường trên mọi ngưỡng, còn F1 đo tại
đúng ngưỡng 0,5. Nghĩa là Transformer phân biệt tốt, nhưng ngưỡng 0,5 chưa hợp với phân bố điểm của nó.

Cuối cùng, chênh lệch giữa các mô hình khá nhỏ: lớp SPAM chỉ có 112 tin, nên mỗi tin thay đổi Recall khoảng 0,89
điểm. Cộng thêm 84 tin trùng train, em không kết luận mạnh từ riêng bảng này."

---

## Slide 10. Chi phí tính toán (40 giây)

**Lời nói:**
"Về chi phí: MLP có nhiều tham số nhất nhưng train nhanh nhất, chỉ 8,9 giây, vì chỉ nhân ma trận trên vector cố
định. Transformer train lâu nhất, khoảng 105 giây.

Về độ trễ, khi xử lý từng tin một, BiLSTM nhanh nhất với 1,65 mili-giây mỗi tin. Khi xử lý theo lô 32 tin, chi phí
mỗi tin giảm 2,8 đến 5,8 lần, và MLP nhanh nhất. Cả ba đều dưới 3 mili-giây, đủ nhanh cho ứng dụng thực tế.

Số đo chỉ trên một CPU i7-12700H và một lần đo, nên chỉ dùng để so sánh tương đối."

**Chuyển:** "Kết quả quan trọng nhất của đồ án nằm ở slide tiếp theo."

---

## Slide 11. SMS thật ngoài nguồn: thứ hạng đảo ngược (1 phút 10)

**Lời nói:**
"Trên 387 tin nhắn thật từ nguồn khác, thứ hạng đảo ngược hoàn toàn. Transformer, vốn xếp cuối trên test UCI, giờ
đứng đầu với F1 97,56%. BiLSTM giữ hạng hai, 94,60%. Còn MLP, vốn đứng đầu, rơi xuống cuối với 83,92%.

Nhìn bảng dưới, khác biệt nằm ở Recall: MLP bỏ sót 46 trên 166 tin rác, trong khi Transformer chỉ bỏ sót 6. Giải
thích phù hợp là MLP dựa nhiều vào các cụm từ cụ thể đã gặp khi train; gặp tin rác với cách viết mới, nó không
nhận ra. Transformer học được ngữ cảnh rộng hơn nên tổng quát tốt hơn.

Bài học chính: **F1 cao trên test cùng nguồn không bảo đảm mô hình hoạt động tốt với dữ liệu mới.** Tuy vậy, tập
này nhỏ, gồm tin từ 2015 đến 2020 và nghiêng về người dùng Ấn Độ, nên cần thận trọng."

**Nhấn:** đây là slide nền tối — dừng lại một nhịp sau câu bài học chính.

---

## Slide 12. Bổ sung dữ liệu lừa đảo tổng hợp (1 phút)

**Lời nói:**
"Ở slide 4, em đã nhắc tập tổng hợp. Trước khi bổ sung, cả ba mô hình gần như vô dụng trên lừa đảo kiểu mới: F1
chỉ 34 đến 47%. Em thử thêm 3.906 tin tổng hợp vào train, giữ nguyên validation và test.

Kết quả: F1 trên tập tổng hợp lên 100% ở cả ba mô hình. Nhưng đây là bài quá dễ, vì tin test gần như cùng mẫu câu
với tin train. Ngược lại, trên test UCI cả ba đều giảm, chủ yếu do bỏ sót spam nhiều hơn — bảng bên phải: số tin
bỏ sót của Transformer tăng từ 8 lên 12. Trên SMS thật, chỉ MLP cải thiện, từ 83,9 lên 89,0; BiLSTM và
Transformer đều giảm.

Kết luận: dữ liệu sinh theo khuôn mẫu có giá trị hạn chế."

---

## Slide 13. Kiểm thử với 66 SMS tự soạn (1 phút)

**Lời nói:**
"Để xem mô hình phản ứng với tin nhắn hiện đại, em soạn 66 tin, chia bốn bộ: hai bộ trộn cả spam và ham; bộ 3
chỉ có spam để đo khả năng bắt; bộ 4 chỉ có ham, gồm OTP, giao hàng, hóa đơn, để đo mức báo nhầm.

Transformer bản gốc — cũng là mô hình chạy trên website — đúng 59 trên 66 tin. Bản Transformer có thêm dữ liệu
tổng hợp được 56, BiLSTM 54, MLP 50.

Điều thú vị là số tin HAM bị báo nhầm gần như bằng nhau, 4 đến 5 tin; khác biệt chính nằm ở số spam bị bỏ sót, MLP
sót tới 11. Thứ hạng này khớp với kết quả trên tập ngoài nguồn ở slide 11.

Em nhấn mạnh: bộ tin nhỏ và không chọn ngẫu nhiên, nên đây không phải phép đo metric, chỉ dùng để tìm dạng lỗi."

**Chuyển:** "Vậy mô hình sai ở đâu, và vì sao?"

---

## Slide 14. Phân tích lỗi của mô hình website (1 phút 20)

**Lời nói:**
"Với mỗi tin báo nhầm, em chạy thêm các biến thể, bỏ từng phần của tin để tách nguyên nhân.

Tin FedEx có mã vận đơn được chấm 1,00, tức chắc chắn là rác. Bỏ mã vận đơn đi, điểm giảm còn 0,02. Nguyên nhân là
**lỗi của hàm chuẩn hóa**: luật nhận số điện thoại bắt mọi dãy 7 đến 15 chữ số, nên mã vận đơn bị đổi thành PHONE.

Tin thanh toán tiền nước 45 đô được 0,94; bỏ số tiền thì còn 0,22. Tin đặt bàn có câu 'Reply CANCEL' được 0,96;
bỏ câu đó còn 0,30. Đây là **lỗi của mô hình**: trong dữ liệu UCI, ký hiệu tiền kèm số và cụm 'reply X to Y' gần như
chỉ có ở tin rác, nên mô hình coi chúng là dấu hiệu SPAM. Tin giao dịch ngân hàng hợp lệ hầu như không có trong
train. Riêng tin Wells Fargo, bỏ số tiền vẫn 0,99, tức cả cấu trúc thông báo giao dịch bị coi là rác.

Về bỏ sót: tin lừa đảo giả làm người thân, 'Hi mum, I dropped my phone', chỉ được 0,03 vì không có link, văn phong
như tin thường.

Ngược lại, có một điểm mô hình làm tốt: cảnh báo đăng nhập thật của ngân hàng Chase được 0,00, còn tin giả mạo Chase
yêu cầu bấm link được 1,00 — hai tin gần giống nhau về từ ngữ nhưng mô hình phân biệt được."

**Nhấn:** nâng ngưỡng không cứu được, vì các tin báo nhầm đều có điểm 0,94–1,00.

---

## Slide 15. Website demo (50 giây — có thể demo trực tiếp)

**Lời nói:**
"Em xây một website bằng Flask để minh họa. Luồng xử lý: tin nhắn nhập vào được chuẩn hóa — dùng chung hàm với lúc
huấn luyện để kết quả nhất quán — sau đó mã hóa thành token, đưa qua mô hình ra logit, qua sigmoid thành xác suất,
và so với ngưỡng 0,5 để ra nhãn.

Em chọn Transformer cho website vì nó tốt nhất trên tin thật ngoài nguồn, gần với tin người dùng tự nhập; đánh đổi
là thấp hơn khoảng 3 điểm F1 trên test UCI. Website hiển thị từng bước trung gian để người xem hiểu mô hình làm gì.
Em đã kiểm thử các nhánh GET, POST và kiểm tra đầu vào; đây là demo chạy cục bộ, chưa thử chịu tải."

**Nếu demo trực tiếp (gợi ý 2 tin):**
1. Tin rác có link, ví dụ thông báo trúng thưởng → SPAM, điểm gần 1.
2. Tin hỏi thăm bình thường → HAM, điểm gần 0.

Tránh tự demo bằng tin OTP hoặc tin có số tiền: đây là dạng mô hình đã biết hay báo nhầm (slide 14). Nếu hội đồng
hỏi thử dạng này, nói trước rằng đây là lỗi đã phân tích.

---

## Slide 16. Kết luận và hạn chế (50 giây)

**Lời nói:**
"Tóm lại, không có mô hình tốt nhất cho mọi tập. MLP thắng trên test cùng nguồn với F1 96,77%. Transformer thắng
trên tin thật ngoài nguồn với 97,56%, có AUC cao nhất và đúng nhiều nhất trên tin tự soạn. Lý do là MLP học cụm từ
cụ thể nên mạnh trên dữ liệu quen, còn Transformer tổng quát tốt hơn trên tin mới.

Đồ án còn ba nhóm hạn chế. Về dữ liệu: 84 tin test trùng train, tập ngoài nguồn nhỏ và cũ. Về thực nghiệm: mới chạy
một seed và một lần đo độ trễ, chưa biết kết quả dao động bao nhiêu. Về mô hình: còn báo nhầm tin giao dịch có số
tiền, luật số điện thoại bắt nhầm mã đơn, và bỏ sót lừa đảo không có link."

---

## Slide 17. Hướng phát triển và cảm ơn (40 giây)

**Lời nói:**
"Từ các hạn chế đó, em đề xuất bốn hướng: chia lại dữ liệu theo nhóm nội dung để bỏ tin trùng giữa train và test;
thu thập tin lừa đảo và tin giao dịch hợp lệ thật, hiện đại, thay vì dữ liệu sinh theo mẫu; sửa bước chuẩn hóa để
tách mã đơn, mã OTP khỏi số điện thoại; và chạy nhiều seed để báo độ dao động của kết quả.

Em xin cảm ơn thầy và các bạn đã lắng nghe. Em xin nhận câu hỏi."

---

## Chuẩn bị hỏi đáp

**1. Vì sao website không dùng MLP, mô hình có F1 cao nhất?**
Vì website xử lý tin mới do người dùng nhập. Trên tin thật ngoài nguồn, MLP bỏ sót 46/166 tin rác, Transformer chỉ
sót 6. F1 cao của MLP đo trên dữ liệu cùng nguồn, có 84 tin trùng train.

**2. Ngưỡng 0,5 có tối ưu không?**
Không. Đây là quy ước cố định để ba mô hình so sánh trong cùng điều kiện. Transformer có AUC cao nhất nhưng F1 thấp
nhất trên UCI, cho thấy nó có thể hưởng lợi từ ngưỡng khác; chọn ngưỡng riêng trên validation là hướng làm tiếp.
Tuy vậy, với các tin báo nhầm ở slide 14 (điểm 0,94–1,00), đổi ngưỡng không sửa được.

**3. Kết quả có ổn định không?**
Chưa đo được, vì mới chạy một seed. Chênh lệch trên tập ngoài nguồn giữa các mô hình chỉ vài tin đến vài chục tin,
nên cần chạy nhiều seed trước khi khẳng định.

**4. So sánh có công bằng giữa các kiến trúc không?**
Công bằng về dữ liệu, ngưỡng, cách chọn checkpoint và siêu tham số chung. Nhưng MLP dùng TF-IDF còn hai mô hình kia
dùng embedding, nên đây là so sánh pipeline, chưa tách riêng tác dụng của kiến trúc.

**5. Vì sao không dùng BERT hay mô hình pretrained?**
Mục tiêu đồ án là so sánh ba kiến trúc học từ đầu trên cùng dữ liệu. Mô hình pretrained đã học từ kho văn bản lớn
khác, sẽ làm so sánh không còn cùng điều kiện.

**6. Tại sao thêm dữ liệu lại làm kết quả tệ hơn?**
Dữ liệu tổng hợp sinh từ mẫu câu, rất giống nhau; mô hình học thuộc khuôn mẫu đó (đạt 100% trên tập cùng loại) nhưng
dịch chuyển ranh giới quyết định, nên bỏ sót nhiều spam thật hơn. Chỉ MLP cải thiện trên SMS thật.

**7. Có thể sửa lỗi báo nhầm tin giao dịch không?**
Lỗi mã vận đơn sửa được bằng cách đổi luật chuẩn hóa, nhưng phải train lại cả ba mô hình. Lỗi số tiền cần thêm tin
giao dịch hợp lệ thật vào train. Em đã thử thêm tin giao dịch tự sinh theo mẫu, nhưng mô hình chuyển sang bỏ sót
nhiều spam hơn, nên đây vẫn là hướng phát triển.

**8. Tại sao Transformer ít tham số hơn MLP mà train lâu hơn?**
Số tham số MLP chủ yếu nằm ở lớp đầu nối 15.000 đặc trưng, nhưng mỗi bước chỉ là một phép nhân ma trận. Transformer
phải tính attention giữa mọi cặp trong 100 token của mỗi tin, nên mỗi bước tốn nhiều phép tính hơn.

**9. 84 tin trùng có làm sai lệch kết luận không?**
Có thể làm điểm trên test UCI lạc quan, nhất là với MLP vốn dựa vào cụm từ cụ thể. Đó là lý do kết luận chính dựa
trên tập ngoài nguồn đã lọc trùng.
