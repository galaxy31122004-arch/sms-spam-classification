# CHƯƠNG 4 — KẾT QUẢ THỰC NGHIỆM VÀ SO SÁNH

## 4.1. Nguồn kết quả và điều kiện so sánh

Nguồn chính của chương là `eval/results/baseline_test.json`, tạo bằng `eval/evaluate.py` từ ba checkpoint huấn luyện lại cục bộ trên `data/splits/train_base.csv` (3.900 SMS UCI) với hàm chuẩn hóa hiện tại. Cấu hình được đối chiếu với ba `eval/results/baseline/<kind>/config.json`. Tập test `data/splits/test.csv` có 836 SMS, gồm 724 HAM và 112 SPAM. Toàn bộ số liệu trong chương được tổng hợp lại vào hồ sơ biên tập `tong-hop-so-lieu.json` để đối chiếu.

Lượt chạy trước đó trên Google Colab (`outputs/test_results.json`) dùng cột `text` sinh bởi phiên bản chuẩn hóa cũ, không còn tái lập được bằng mã hiện tại. Kết quả đó được giữ ở mục 4.6 như lượt chạy lịch sử, không dùng làm số liệu chính.

**Bảng 4.1. Điều kiện chung và giới hạn kiểm soát của phép so sánh.**

| Nội dung | Điều kiện được dùng | Giới hạn diễn giải |
|---|---|---|
| Dữ liệu | Cùng `train_base.csv`, `val.csv`, `test.csv` | Còn nội dung trùng giữa các tập |
| Lớp dương | SPAM = 1; HAM = 0 | Chỉ bài toán nhị phân trên SMS tiếng Anh |
| Chọn checkpoint | F1 SPAM validation cao nhất; giữ epoch đầu khi hòa | Log từng epoch chỉ có ở đầu ra dòng lệnh, chưa lưu thành artifact |
| Ngưỡng | 0,5 cho cả ba mô hình | Chưa tối ưu ngưỡng riêng |
| Seed | 42 trong ba config | Chưa khảo sát nhiều seed |
| Batch huấn luyện, learning rate khởi tạo | 64; 0,001 | Transformer có lịch learning rate riêng theo mã |
| Ngân sách epoch | Yêu cầu 10; patience 3 | Số epoch thực chạy khác nhau |
| Biểu diễn | TF-IDF cho MLP; token ID cho BiLSTM/Transformer | So sánh pipeline, không cô lập tác dụng riêng của kiến trúc |
| Môi trường | CPU Intel Core i7-12700H, RAM 31,7 GB, Windows 11; Python 3.11.9, PyTorch 2.15.0+cpu, scikit-learn 1.7.2, NumPy 2.4.6 | Một máy, một lần đo; không đại diện cho mọi thiết bị |

*Nguồn: ba config baseline, `eval/results/baseline_test.json`, mã train/evaluate, kiểm tra CSV và thông tin môi trường đọc trên máy chạy thực nghiệm. Bảng cấu hình kiến trúc đầy đủ đặt tại Bảng 3.1–3.2.*

Điều kiện chung giúp phép so sánh có quy tắc nhất quán, nhưng chưa phải khảo sát mọi tổ hợp siêu tham số. Không có bằng chứng cả ba mô hình đã được tối ưu ngang nhau về công sức tuning hoặc được cấp cùng tổng số phép tính. Kết luận phù hợp là so sánh các cấu hình được lưu trên cùng dữ liệu, không phải xếp hạng mọi MLP, LSTM và Transformer có thể xây dựng.

## 4.2. Kết quả phân loại

**Bảng 4.2. Các chỉ số test của ba mô hình trên 836 SMS.**

| Chỉ số | MLP | BiLSTM | Transformer |
|---|---:|---:|---:|
| Accuracy | 99,16% | 98,68% | 98,33% |
| Precision SPAM | 100,00% | 99,03% | 94,55% |
| Recall SPAM | 93,75% | 91,07% | 92,86% |
| F1 SPAM | 96,77% | 94,88% | 93,69% |
| ROC-AUC | 98,13% | 98,43% | 99,38% |

*Nguồn: `eval/results/baseline_test.json`, làm tròn hai chữ số phần trăm. Accuracy/Precision/Recall/F1 dùng ngưỡng 0,5; ROC-AUC dùng điểm liên tục. Tính lại bốn chỉ số đầu từ ma trận khớp giá trị JSON trước làm tròn.*

MLP có Accuracy và F1 SPAM cao nhất. So với BiLSTM, Accuracy cao hơn khoảng 0,48 điểm phần trăm và F1 cao hơn 1,89 điểm phần trăm. So với Transformer, các chênh lệch tương ứng khoảng 0,84 và 3,08 điểm phần trăm. Những độ chênh này được tính từ giá trị JSON đầy đủ rồi làm tròn.

Transformer có ROC-AUC cao nhất, hơn BiLSTM khoảng 0,96 điểm phần trăm và hơn MLP khoảng 1,25 điểm phần trăm. Kết quả này không mâu thuẫn với F1 thấp hơn tại ngưỡng 0,5. AUC thể hiện thứ tự điểm trên nhiều ngưỡng; F1 thể hiện nhãn tại một ngưỡng cụ thể. Một mô hình có thể xếp hạng phần lớn cặp SPAM–HAM tốt nhưng có phân bố điểm chưa phù hợp với ngưỡng đang cố định.

Thứ hạng F1 trên test UCI (MLP > BiLSTM > Transformer) giống lượt chạy Colab trước đó, nhưng mục 4.7 cho thấy thứ hạng này đảo ngược trên dữ liệu ngoài nguồn. Vì vậy, kết quả của mục này chỉ nên đọc là kết quả trong cùng phân bố với dữ liệu huấn luyện.

## 4.3. Ma trận nhầm lẫn và hai loại sai sót

Ba ma trận đều có dạng `[[TN, FP], [FN, TP]]`; hàng là nhãn thật, cột là dự đoán, thứ tự HAM rồi SPAM.

**Bảng 4.3. Ma trận nhầm lẫn MLP tại ngưỡng 0,5.**

| Nhãn thật \ Dự đoán | HAM | SPAM | Tổng hàng |
|---|---:|---:|---:|
| HAM | 724 (TN) | 0 (FP) | 724 |
| SPAM | 7 (FN) | 105 (TP) | 112 |
| Tổng cột | 731 | 105 | 836 |

*Nguồn: `eval/results/baseline_test.json`, mô hình `mlp`.*

MLP phát hiện 105/112 SMS rác, bỏ sót 7 và không gán nhầm SMS HAM nào thành SPAM trong tập test. Precision 100,00% nghĩa là cả 105 tin bị đánh dấu SPAM đều là SPAM thật trong tập này; con số đó không có nghĩa mô hình phát hiện mọi tin rác. Tỷ lệ bỏ sót lớp SPAM là $7/112=6{,}25\%$. Tính lại F1 cho thấy $2\times105/(2\times105+0+7)=0{,}967741\ldots$, phù hợp 96,77% sau làm tròn.

**Bảng 4.4. Ma trận nhầm lẫn BiLSTM tại ngưỡng 0,5.**

| Nhãn thật \ Dự đoán | HAM | SPAM | Tổng hàng |
|---|---:|---:|---:|
| HAM | 723 (TN) | 1 (FP) | 724 |
| SPAM | 10 (FN) | 102 (TP) | 112 |
| Tổng cột | 733 | 103 | 836 |

*Nguồn: `eval/results/baseline_test.json`, mô hình `lstm`; lớp được triển khai là LSTM hai chiều.*

BiLSTM có 11 lỗi: một tin hợp lệ bị chặn nhầm và mười tin rác bị bỏ sót. So với MLP, tổng lỗi tăng bốn. Từ ma trận tổng hợp không biết hai mô hình có sai trên cùng những SMS nào.

**Bảng 4.5. Ma trận nhầm lẫn Transformer tại ngưỡng 0,5.**

| Nhãn thật \ Dự đoán | HAM | SPAM | Tổng hàng |
|---|---:|---:|---:|
| HAM | 718 (TN) | 6 (FP) | 724 |
| SPAM | 8 (FN) | 104 (TP) | 112 |
| Tổng cột | 726 | 110 | 836 |

*Nguồn: `eval/results/baseline_test.json`, mô hình `transformer`.*

Transformer có 14 lỗi, gồm sáu HAM bị chặn nhầm và tám SPAM bị bỏ sót. Recall cao hơn BiLSTM (104 so với 102 TP), nhưng FP nhiều hơn năm mẫu làm Precision và F1 thấp hơn. Tỷ lệ chặn nhầm trên 724 HAM là khoảng 0,14% với BiLSTM và 0,83% với Transformer; MLP là 0% trong tập này.

Trong sử dụng SMS, FP có thể làm người dùng bỏ qua một tin hợp lệ, còn FN làm bộ lọc giữ lại một tin rác. Chưa có mô hình chi phí hoặc khảo sát người dùng để quy đổi hai lỗi thành một lợi ích cụ thể. Ngưỡng 0,5 là quy ước so sánh, không phải ngưỡng phù hợp nhất cho mọi tình huống sử dụng.

## 4.4. Chi phí tính toán và cách đo latency

**Bảng 4.6. Số tham số, thời gian huấn luyện và latency.**

| Chỉ số | MLP | BiLSTM | Transformer |
|---|---:|---:|---:|
| Số tham số mạng | 3.873.281 | 1.158.273 | 1.155.457 |
| Thời gian huấn luyện (giây) | 8,909 | 29,752 | 104,944 |
| Latency batch 1 (ms/tin) | 2,5641 | 1,6477 | 2,2641 |
| Latency batch 32 (ms/tin) | 0,4355 | 0,4732 | 0,8080 |

*Nguồn: `eval/results/baseline_test.json`; số tham số và thời gian khớp ba config baseline. Đo trên CPU i7-12700H, tối đa bốn luồng PyTorch. Latency batch 32 có đơn vị ms/tin, không phải ms/batch.*

### 4.4.1. Phạm vi phép đo trong mã hiện tại

`load_checkpoint` nạp mô hình lên CPU; `evaluate` giới hạn số luồng PyTorch tối đa bốn. Trước khi đo, mã chạy một lượt warmup với một SMS ở batch 1 và một lượt với tối đa 32 SMS ở batch 32. Warmup không được tính vào thời gian hoặc metric chính.

`benchmark_prediction` bắt đầu đồng hồ `time.perf_counter`, chia toàn bộ danh sách SMS thành batch, gọi `transform_texts` rồi `predict_probabilities`, cuối cùng chia tổng thời gian cho tổng số SMS:

$$
Latency_{ms/tin}=\frac{1000\,(t_{end}-t_{start})}{N_{SMS}}. \tag{4.1}
$$

Hai mốc $t_{start}$ và $t_{end}$ là thời gian bắt đầu/kết thúc vòng xử lý, đo bằng giây; hệ số 1000 đổi sang mili giây và $N_{SMS}>0$ là tổng số tin được xử lý. Thời gian bao gồm TF-IDF/chuyển mảng đặc hoặc mã hóa token, forward mạng, sigmoid và thao tác gom điểm trong vòng lặp. Nó không bao gồm đọc CSV, nạp checkpoint, tính metric sau vòng lặp, tải trang hay truyền HTTP.

Với 836 SMS và batch 32, có 26 batch đầy cùng một batch cuối bốn SMS; mẫu cuối vẫn được tính trong mẫu số 836. Điểm từ lượt batch 1 được dùng để tính metric; lượt batch 32 phục vụ đo latency. Mã chỉ đo một lần cho mỗi cấu hình, không lưu độ lệch chuẩn hoặc phân vị p95.

### 4.4.2. Diễn giải các đánh đổi

MLP có nhiều tham số nhất nhưng thời gian train thấp nhất. Điều này phù hợp với đặc điểm các phép Dense trên vector cố định so với xử lý hồi tiếp hoặc attention; tuy nhiên, số liệu tổng hợp chưa đủ cô lập ảnh hưởng của kiến trúc, vector hóa hay số epoch. Transformer thực chạy bảy epoch, còn hai mô hình kia chạy mười; tổng thời gian không phải thời gian mỗi epoch.

BiLSTM có latency batch 1 thấp nhất, MLP có latency batch 32 thấp nhất. Transformer chậm nhất ở batch 32 dù có ít tham số nhất. Giảm ms/tin khi dùng batch cho thấy hiệu quả bình quân của xử lý theo lô. Không thể diễn giải các con số này thành thời gian đáp ứng mỗi yêu cầu web, vì chi phí ứng dụng chưa được đo, và chúng không xác nhận mô hình nhanh nhất trên GPU hoặc điện thoại.

## 4.5. Giới hạn độ tin cậy của thực nghiệm trên test UCI

Thứ nhất, nội dung trùng giữa các tập vẫn còn: 84 dòng test có `text` xuất hiện trong train, gồm 23 dòng SPAM; 81 dòng validation có `text` xuất hiện trong train (Bảng 2.4). Thành viên các tập được giữ nguyên để kết quả so sánh được với lượt chạy trước, nên hạn chế này chưa được khắc phục. Không được trừ 84 mẫu rồi suy ra metric mới.

Thứ hai, lớp SPAM trong test chỉ có 112 SMS. Một SMS được phát hiện thêm tương ứng thay đổi Recall khoảng $100/112\approx0{,}89$ điểm phần trăm. Chênh lệch Recall giữa MLP và Transformer chỉ là một TP.

Thứ ba, hồ sơ chỉ có một seed huấn luyện. Chưa phân biệt được bao nhiêu chênh lệch do phương án mô hình và bao nhiêu do ngẫu nhiên.

Thứ tư, test UCI cùng nguồn với train. Mục 4.7 bổ sung đánh giá trên dữ liệu ngoài nguồn, nhưng tập đó nhỏ và cũng có giới hạn riêng.

## 4.6. Đối chiếu với lượt chạy Colab trước đó

**Bảng 4.7. F1 SPAM trên `test.csv` của lượt Colab và lượt huấn luyện lại.**

| Mô hình | Lượt Colab (chuẩn hóa cũ) | Lượt hiện tại (chuẩn hóa mới) |
|---|---:|---:|
| MLP | 96,30% | 96,77% |
| BiLSTM | 95,37% | 94,88% |
| Transformer | 94,06% | 93,69% |

*Nguồn: `outputs/test_results.json` và `eval/results/baseline_test.json`. Hai lượt khác nhau về phiên bản chuẩn hóa của cột `text`, máy chạy và thư viện; thành viên ba tập không đổi.*

Hai lượt khác nhau trong phạm vi một đến hai SMS cho mỗi mô hình và giữ cùng thứ hạng. Phiên bản chuẩn hóa mới thay đổi cột `text` của 169 dòng train, 32 dòng validation và 39 dòng test, chủ yếu là các khoản cước dạng pence và tên miền không có `http`/`www` (mục 2.4). Chênh lệch nhỏ cho thấy kết quả không nhạy với thay đổi này, nhưng một lần so sánh không đủ để kết luận về độ ổn định.

## 4.7. Đánh giá ngoài nguồn và thí nghiệm bổ sung dữ liệu

### 4.7.1. Hai tập đánh giá bổ sung

Để kiểm tra khả năng tổng quát, đồ án dùng thêm hai tập không tham gia chọn checkpoint:

- `test_smishing.csv`: 689 SMS (562 HAM, 127 SPAM) từ bộ dữ liệu tổng hợp africa-smishing [17], chia theo nhóm template để không trùng với phần đưa vào train.
- `test_external.csv`: 387 SMS thật (221 HAM, 166 SPAM, trong đó 102 tin gốc mang nhãn smishing) từ bộ SMS Phishing Dataset trên Mendeley [18]. Phần lớn bộ này sao chép SMS Spam Collection, nên đã loại 5.035 dòng trùng chính xác và 447 dòng gần trùng (độ giống chuỗi chữ cái từ 0,8) với UCI, cùng các dòng quá ngắn, lỗi mã hóa hoặc lặp lại.

Cả hai tập không có dòng nào trùng `text` với train, validation hoặc test. Quy trình tạo tập nằm trong `data/augment_smishing.py` và `data/build_external_test.py`.

### 4.7.2. Thí nghiệm bổ sung dữ liệu smishing

Bộ africa-smishing có 10.000 dòng tổng hợp (5.000 HAM, 5.000 SPAM) theo các mẫu tin ngân hàng, ví điện tử và lừa đảo nhận thưởng. Sau khi chuẩn hóa và gom theo template (thay chữ số bằng 0), còn 1.109 nhóm; 85% số nhóm theo từng nhãn được thêm vào train, 15% làm `test_smishing.csv`. Tập `train.csv` sau bổ sung có 7.806 SMS (6.564 HAM, 1.242 SPAM). Ba mô hình được huấn luyện lại với cùng cấu hình; validation và test UCI giữ nguyên.

**Bảng 4.8. F1 SPAM trước (baseline) và sau (augmented) khi bổ sung dữ liệu, trên ba tập test.**

| Mô hình | `test.csv` (836) | `test_smishing.csv` (689) | `test_external.csv` (387) |
|---|---:|---:|---:|
| MLP baseline | 96,77% | 34,28% | 83,92% |
| MLP augmented | 95,37% | 100,00% | 89,04% |
| BiLSTM baseline | 94,88% | 47,40% | 94,60% |
| BiLSTM augmented | 92,45% | 100,00% | 93,25% |
| Transformer baseline | 93,69% | 38,43% | 97,56% |
| Transformer augmented | 93,46% | 100,00% | 94,97% |

*Nguồn: `eval/results/{baseline,augmented}_{test,test_smishing,test_external}.json`, ngưỡng 0,5. Các ma trận nhầm lẫn đầy đủ nằm trong cùng tệp và trong hồ sơ biên tập `tong-hop-so-lieu.json`.*

Trên `test_smishing.csv`, baseline gần như không dùng được: Precision chỉ 21,14–33,23% vì 211–429 trên 562 tin HAM tổng hợp bị gán SPAM. Sau bổ sung, cả ba đạt 100,00%. Kết quả tuyệt đối này không nên hiểu là khả năng tổng quát: các template chỉ khác nhau ở tên tổ chức và đơn vị tiền tệ, nên tập test vẫn rất gần train dù đã chia theo nhóm.

Trên test UCI, bổ sung dữ liệu làm F1 giảm ở cả ba mô hình, chủ yếu do Recall giảm: số SPAM bị bỏ sót tăng từ 7 lên 9 (MLP), 10 lên 14 (BiLSTM) và 8 lên 12 (Transformer).

### 4.7.3. Kết quả trên dữ liệu thật ngoài nguồn

**Bảng 4.9. Ma trận nhầm lẫn của các mô hình baseline trên `test_external.csv`.**

| Mô hình | TN | FP | FN | TP | Recall SPAM | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 221 | 0 | 46 | 120 | 72,29% | 98,99% |
| BiLSTM | 221 | 0 | 17 | 149 | 89,76% | 99,24% |
| Transformer | 219 | 2 | 6 | 160 | 96,39% | 99,90% |

*Nguồn: `eval/results/baseline_test_external.json`, ngưỡng 0,5.*

Trên SMS thật ngoài nguồn, thứ hạng đảo ngược so với test UCI: Transformer đạt F1 97,56%, BiLSTM 94,60% và MLP 83,92%. Khác biệt chủ yếu nằm ở Recall: MLP bỏ sót 46/166 tin rác, Transformer bỏ sót 6. Một cách giải thích phù hợp với số liệu là MLP dựa vào n-gram cụ thể đã gặp trong train UCI, còn hai mô hình chuỗi tổng quát tốt hơn khi từ ngữ thay đổi; tuy nhiên, chưa có phân tích lỗi theo từng mẫu để khẳng định nguyên nhân.

Bổ sung dữ liệu tổng hợp chỉ cải thiện MLP trên tập này (83,92% lên 89,04%), còn BiLSTM và Transformer giảm (Transformer bỏ sót tăng từ 6 lên 15 tin). Dữ liệu tổng hợp theo template giúp mô hình đơn giản học thêm dấu hiệu bề mặt, nhưng có thể làm mô hình mạnh hơn học các mẫu quá hẹp.

Giới hạn của tập này cần được giữ khi diễn giải. Tập chỉ có 387 SMS, nên mỗi chênh lệch tương ứng vài đến vài chục tin. Tin HAM chủ yếu là trao đổi cá nhân ngắn, dễ phân biệt, nên tỷ lệ chặn nhầm gần 0 có thể lạc quan hơn thực tế. Tin chủ yếu thu thập khoảng 2015–2020 và nghiêng về người dùng Ấn Độ, chưa đại diện cho SMS lừa đảo hiện nay.

### 4.7.4. Kiểm tra định tính với SMS tự soạn

Ngoài các tập có nhãn, 66 SMS tiếng Anh tự soạn (33 HAM, 33 SPAM) được dùng để kiểm tra định tính qua `compare_demo_models.py`. Bộ tin cố ý nhắm vào các dạng mà dữ liệu huấn luyện không có: lừa đảo khóa tài khoản, phí hải quan, đầu tư tiền mã hóa, giả làm người thân; phía HAM có thông báo giao dịch thẻ, giao hàng, OTP, hóa đơn và liên kết công việc.

**Bảng 4.10. Kết quả trên 66 SMS tự soạn.**

| Mô hình | Nhận đúng | Báo nhầm HAM | Bỏ sót SPAM |
|---|---:|---:|---:|
| Transformer baseline | 59/66 | 5 | 2 |
| Transformer augmented | 56/66 | 5 | 5 |
| BiLSTM baseline | 54/66 | 4 | 8 |
| MLP baseline | 50/66 | 5 | 11 |

*Nguồn: `compare_demo_models.py` và `docs/ket-qua-kiem-thu-thu-cong.json`; ngưỡng 0,5. Bộ tin do người viết báo cáo và người dùng soạn, không lấy từ bất kỳ split nào.*

Thứ hạng trùng với kết quả trên `test_external.csv`. Bốn mô hình báo nhầm HAM gần bằng nhau; khác biệt nằm ở số SPAM bỏ sót.

Chạy thêm các biến thể của từng tin sai cho phép tách hai loại nguyên nhân. Thứ nhất là hạn chế của bước chuẩn hóa: mã vận đơn “7741 2093” bị thay bằng `<PHONE>` do luật nhận số điện thoại bắt mọi dãy 7–15 chữ số, và chỉ riêng điều này đưa một thông báo giao hàng hợp lệ từ 0,02 lên 1,00. Thứ hai là hạn chế của dữ liệu: bốn tin HAM còn lại bị báo nhầm đều chứa `$<MONEY>` hoặc cụm “reply X to Y”; bỏ số tiền khỏi tin thanh toán tiền nước làm điểm giảm từ 0,94 xuống 0,22, viết số tiền bằng chữ làm điểm còn 0,01. Trong SMS Spam Collection, ký hiệu tiền tệ kèm số hầu như chỉ xuất hiện ở tin trúng thưởng hoặc dịch vụ tính phí, nên mô hình học `$<MONEY>` thành đặc trưng của lớp SPAM; thông báo giao dịch ngân hàng hợp lệ gần như không có trong dữ liệu.

Hai tin SPAM bị bỏ sót đều là lừa đảo phi kỹ thuật không chứa liên kết hoặc từ khóa quảng cáo (“Hi mum, I dropped my phone… send me $300” và một tin mạo danh bạn bè). Với tin thứ hai, thay liên kết thành `<URL>` chỉ nâng điểm từ 0,01 lên 0,22, nên nguyên nhân là văn phong chứ không phải cách xử lý liên kết.

Bộ tin này quá nhỏ và không chọn ngẫu nhiên nên không được báo cáo như một metric; quy trình đầy đủ và phân tích từng lỗi đặt tại `docs/kiem-thu-tin-nhan-thuc-te.md`.

## 4.8. Tổng hợp và cơ sở chọn mô hình cho demo

**Bảng 4.11. Kết luận được số liệu hỗ trợ cho từng mô hình.**

| Mô hình | Điểm thuận lợi quan sát được | Giới hạn cần giữ khi diễn giải |
|---|---|---|
| MLP | F1/Accuracy cao nhất trên test UCI; 0 FP; train nhanh nhất | Recall thấp nhất trên dữ liệu ngoài nguồn (72,29%); số tham số lớn nhất |
| BiLSTM | Latency batch 1 thấp nhất; 0 FP trên test ngoài | 10 FN trên test UCI; F1 thấp hơn Transformer trên test ngoài |
| Transformer | ROC-AUC cao nhất trên cả hai tập; F1 cao nhất trên test ngoài (97,56%) | F1 thấp nhất trên test UCI; thời gian train và latency batch 32 cao nhất |

*Nguồn: tổng hợp Bảng 4.2–4.10; nhận xét chỉ áp dụng cho cấu hình, dữ liệu và số liệu được lưu.*

Website dùng Transformer baseline vì đây là mô hình tổng quát tốt nhất trên dữ liệu thật ngoài nguồn, và mục tiêu của demo là phân loại SMS người dùng tự nhập, vốn không thuộc phân bố UCI. Lựa chọn này đánh đổi khoảng ba điểm phần trăm F1 trên test UCI. Số liệu hỗ trợ sự phù hợp của Transformer cho demo học thuật; chưa chứng minh mô hình đủ điều kiện triển khai thực tế.
