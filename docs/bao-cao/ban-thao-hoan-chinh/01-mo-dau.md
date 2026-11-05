# MỞ ĐẦU

## MĐ.1. Bối cảnh và lý do chọn đề tài

Phân loại tin nhắn SMS là bài toán xác định một nội dung văn bản thuộc nhóm tin nhắn bình thường hay tin nhắn rác. Trong bài toán này, hệ thống cần nhận diện các dấu hiệu liên quan đến nội dung không mong muốn, đồng thời hạn chế việc đánh dấu nhầm những tin nhắn hợp lệ. Hai loại sai sót đều cần được xem xét: bỏ sót tin nhắn rác làm giảm khả năng lọc của hệ thống, còn phân loại nhầm tin nhắn bình thường thành tin nhắn rác có thể khiến người dùng bỏ qua thông tin cần thiết. Vì vậy, đánh giá một bộ phân loại SMS cần quan tâm đến cách mô hình xử lý từng lớp, bên cạnh tỷ lệ dự đoán đúng trên toàn bộ dữ liệu.

Đối với đồ án học máy và học sâu, bài toán này cho phép khảo sát đầy đủ quá trình xây dựng một hệ thống phân loại văn bản: xác định nhãn, tổ chức dữ liệu, biểu diễn nội dung bằng các giá trị số, huấn luyện mô hình, đánh giá kết quả và tích hợp mô hình vào ứng dụng. Bộ dữ liệu SMS Spam Collection đang có trong dự án gồm các tin nhắn tiếng Anh được gán nhãn HAM hoặc SPAM, tạo cơ sở cho việc thực hiện và đối chiếu các bước trên trong một phạm vi cụ thể.

Đề tài lựa chọn ba kiến trúc MLP (mạng perceptron nhiều lớp), LSTM (mạng bộ nhớ dài–ngắn hạn) và Transformer để khảo sát những cách xử lý đầu vào khác nhau. Trong triển khai của đồ án, MLP tiếp nhận đặc trưng TF-IDF kết hợp từ các cụm từ và chuỗi ký tự, gọi chung là n-gram. LSTM hai chiều tiếp nhận chuỗi mã token, tức các đơn vị văn bản sau khi tách, và sử dụng lớp embedding để ánh xạ các mã này thành vector học từ đầu. Transformer encoder cũng xử lý chuỗi token, kết hợp thông tin vị trí và cơ chế attention để tính toán quan hệ giữa các token. So sánh ba cách tiếp cận này giúp làm rõ mối liên hệ giữa biểu diễn đầu vào, kiến trúc mô hình và kết quả phân loại trong điều kiện thực nghiệm của dự án.

Một yêu cầu của phép so sánh là các mô hình sử dụng cùng bộ chia dữ liệu và cùng quy tắc đánh giá. Tuy nhiên, đầu vào TF-IDF của MLP khác với đầu vào chuỗi token của hai mô hình còn lại. Do đó, kết quả phản ánh sự kết hợp giữa cách biểu diễn văn bản và kiến trúc, không cho phép quy toàn bộ chênh lệch chỉ cho một thành phần riêng lẻ. Cách đặt vấn đề này phù hợp với mục tiêu của đồ án: giải thích phương pháp, đối chiếu kết quả có căn cứ và nhận diện giới hạn của thực nghiệm.

Từ những lý do trên, đề tài **“Phát hiện tin nhắn SMS rác bằng mạng nơ-ron MLP, LSTM và Transformer”** được thực hiện với hai nội dung gắn kết: xây dựng và so sánh các mô hình phân loại trong thực nghiệm, đồng thời minh họa cách sử dụng mô hình đã huấn luyện qua một website. Website hiện sử dụng Transformer, mô hình tổng quát tốt nhất trên dữ liệu ngoài nguồn; MLP và LSTM phục vụ phần so sánh thực nghiệm. Việc tách rõ hai phạm vi này giúp xác định đúng chức năng của sản phẩm và ý nghĩa của các kết quả được trình bày.

## MĐ.2. Phát biểu bài toán

Đầu vào của hệ thống là nội dung một tin nhắn SMS tiếng Anh. Đầu ra là một trong hai nhãn: **HAM**, biểu thị tin nhắn bình thường, hoặc **SPAM**, biểu thị tin nhắn rác. Trong quá trình tính các chỉ số đánh giá, SPAM được quy ước là lớp dương, tương ứng nhãn số 1; HAM là lớp âm, tương ứng nhãn số 0.

Sau bước biểu diễn văn bản phù hợp với từng kiến trúc, mô hình tạo ra một điểm số thô gọi là logit. Hàm sigmoid chuyển giá trị này thành điểm dự đoán SPAM trong khoảng từ 0 đến 1. Hệ thống quyết định nhãn SPAM khi điểm dự đoán lớn hơn hoặc bằng 0,5; các trường hợp còn lại được gán nhãn HAM. Điểm dự đoán thể hiện đầu ra của mô hình, không tự nó bảo đảm mức độ chính xác của dự đoán đối với một tin nhắn mới.

Ngưỡng 0,5 được sử dụng thống nhất cho cả ba mô hình khi chọn bản lưu trọng số (checkpoint) trên tập validation và khi đánh giá trên tập test. Đây là ngưỡng cố định của phép so sánh, không phải ngưỡng được tối ưu F1 riêng cho từng mô hình. Bài toán của đồ án bao gồm việc đánh giá khả năng phân loại tại ngưỡng này và khả năng xếp hạng mẫu thông qua ROC-AUC, đồng thời xem xét chi phí tính toán đã được ghi nhận.

## MĐ.3. Mục tiêu của đề tài

Mục tiêu tổng quát là xây dựng hệ thống phân loại SMS tiếng Anh thành HAM hoặc SPAM, khảo sát ba kiến trúc mạng nơ-ron được huấn luyện từ đầu và trình bày kết quả so sánh trong phạm vi dữ liệu thực nghiệm hiện có. Transformer trong đề tài là một encoder nhỏ được tự xây dựng, không sử dụng checkpoint BERT hoặc trọng số pretrained.

Các mục tiêu cụ thể gồm:

1. **Mô tả và tổ chức dữ liệu cho bài toán phân loại.** Trình bày đặc điểm bộ dữ liệu, phân bố nhãn, các tập train/validation/test và vai trò của từng tập; làm rõ biểu diễn đầu vào của các mô hình.
2. **Xây dựng ba phương án phân loại.** Triển khai MLP trên đặc trưng TF-IDF, LSTM hai chiều trên chuỗi token và Transformer encoder học từ đầu; giải thích nguyên lý hoạt động và các thành phần chính của từng phương án.
3. **Thực hiện phép so sánh theo quy trình chung.** Sử dụng cùng các tập dữ liệu đã lưu, chọn checkpoint theo F1 của lớp SPAM trên validation và áp dụng ngưỡng quyết định 0,5 thống nhất.
4. **Phân tích kết quả và giới hạn của thực nghiệm.** Đối chiếu Accuracy, Precision, Recall, F1-score, ROC-AUC, ma trận nhầm lẫn, độ trễ, số tham số và thời gian huấn luyện; kiểm tra thêm trên dữ liệu ngoài nguồn và diễn giải những yếu tố ảnh hưởng đến độ tin cậy của kết luận.
5. **Minh họa việc sử dụng mô hình trong ứng dụng.** Mô tả website Python Flask cho phép nhập tin nhắn và xem nhãn phân loại cùng điểm SPAM do Transformer đã huấn luyện cung cấp.

## MĐ.4. Đối tượng, phạm vi, phương pháp và sản phẩm của đồ án

### MĐ.4.1. Đối tượng và phạm vi dữ liệu

Đối tượng nghiên cứu là phương pháp biểu diễn và phân loại nội dung SMS bằng MLP, LSTM và Transformer. Dữ liệu của đồ án là SMS Spam Collection, gồm **5.572 tin nhắn**, trong đó có **4.825 tin HAM** và **747 tin SPAM**. Số lượng hai lớp không cân bằng, vì vậy việc nhận xét kết quả cần xem riêng khả năng phát hiện SPAM và mức độ phân loại nhầm HAM.

Dữ liệu hiện có được chia theo tỷ lệ mục tiêu 70% cho huấn luyện, 15% cho validation và 15% cho test, với giá trị khởi tạo ngẫu nhiên (random seed) 42 theo hồ sơ dự án. Số lượng thực tế tương ứng là **3.900, 836 và 836 tin nhắn**. Cả ba mô hình sử dụng cùng các tập này. Tập train được dùng để học tham số và xây dựng bộ biểu diễn đầu vào; validation phục vụ theo dõi và chọn checkpoint; test được dành cho việc đánh giá sau khi chọn checkpoint.

Việc kiểm tra dữ liệu ghi nhận **84 dòng trong tập test có nội dung trùng với tập train**, khi so khớp chính xác văn bản đang được dùng làm đầu vào mô hình. Đây là một hạn chế có thể ảnh hưởng đến độ tin cậy của kết quả đánh giá: mức độ độc lập giữa train và test bị giảm, nên kết quả có thể lạc quan hơn so với khi đánh giá trên các tin nhắn hoàn toàn mới. Để bù một phần, đồ án đánh giá thêm trên 387 SMS thật từ một nguồn khác, đã loại mọi tin trùng hoặc gần trùng với dữ liệu UCI.

Phạm vi nội dung giới hạn ở phân loại nhị phân SMS tiếng Anh. Ngoài SMS Spam Collection, đồ án dùng hai bộ dữ liệu bổ sung: một bộ SMS lừa đảo tổng hợp để thử nghiệm bổ sung dữ liệu huấn luyện, và một bộ SMS thật để đánh giá ngoài nguồn. Kết quả chưa đủ để khẳng định khả năng nhận diện SMS tiếng Việt hoặc tin nhắn lừa đảo hiện nay. Các mô hình được so sánh là ba kiến trúc học từ đầu; các mô hình pretrained và phương án cascade không thuộc bảng kết quả của đồ án.

### MĐ.4.2. Phương pháp thực hiện và đánh giá

Đồ án sử dụng phương pháp thực nghiệm so sánh, kết hợp giải thích mô hình từ mã nguồn với phân tích cấu hình và kết quả. Ba mô hình lần đầu được huấn luyện trên Google Colab; kết quả chính trong báo cáo đến từ lượt huấn luyện lại cục bộ trên CPU bằng mã hiện tại, có ghi lại phần cứng và phiên bản thư viện. Lượt Colab được giữ để đối chiếu.

Về biểu diễn văn bản, MLP sử dụng đặc trưng TF-IDF từ và ký tự, còn LSTM và Transformer sử dụng mã token từ bộ từ vựng xây dựng trên train. Theo triển khai hiện có, việc học bộ TF-IDF và xây dựng từ vựng chỉ sử dụng tập train; validation và test được biến đổi bằng bộ biểu diễn tương ứng. Cần phân biệt bước chuẩn hóa văn bản với bước chuyển văn bản thành đặc trưng, bởi ba mô hình có cách biểu diễn đầu vào khác nhau.

Cột văn bản của mọi tập đã được sinh lại bằng hàm chuẩn hóa hiện tại, nên dữ liệu huấn luyện, đánh giá và website đi qua cùng một quy trình. Thành viên các tập được giữ nguyên; phương án chia dữ liệu theo nhóm văn bản đã có trong mã nhưng chưa được áp dụng.

Việc đánh giá sử dụng nhiều chỉ số để phản ánh các khía cạnh khác nhau của bộ phân loại. **Accuracy** là tỷ lệ dự đoán đúng trên toàn bộ mẫu. Với SPAM là lớp dương, **Precision** cho biết trong các tin bị mô hình gán nhãn SPAM, bao nhiêu tin thực sự là SPAM; **Recall** cho biết mô hình phát hiện được bao nhiêu trong tổng số tin SPAM thực tế. **F1-score** là trung bình điều hòa của Precision và Recall, giúp xem xét đồng thời hai đại lượng này. Các chỉ số trên được tính tại ngưỡng quyết định 0,5.

**ROC-AUC** phản ánh khả năng xếp hạng mẫu SPAM cao hơn mẫu HAM thông qua diện tích dưới đường ROC khi thay đổi ngưỡng. Chỉ số này không đồng nghĩa với Accuracy tại ngưỡng 0,5. **Ma trận nhầm lẫn** làm rõ số tin HAM được nhận diện đúng, HAM bị gán nhầm SPAM, SPAM bị bỏ sót và SPAM được nhận diện đúng. Trong báo cáo, hàng của ma trận là nhãn thật, cột là nhãn dự đoán và thứ tự nhãn là HAM rồi SPAM. Cách kết hợp các chỉ số giúp tránh đánh giá hệ thống chỉ từ một con số tổng hợp.

Bên cạnh chất lượng phân loại, báo cáo đối chiếu số tham số, thời gian huấn luyện và độ trễ xử lý. Phép đo độ trễ bao gồm bước vector hóa và suy luận trên CPU, được biểu diễn theo thời gian trung bình cho mỗi tin nhắn ở các kích thước lô xử lý (batch) tương ứng. Các số đo thực hiện trên một máy cụ thể, một lần đo, nên chưa đủ để suy rộng sang môi trường triển khai khác.

### MĐ.4.3. Sản phẩm và giới hạn triển khai

Các sản phẩm của đồ án gồm mã nguồn cho ba phương án phân loại, bộ biến đổi đầu vào và checkpoint đã lưu, kết quả đánh giá trên tập test hiện có và website minh họa. Báo cáo trình bày những sản phẩm này trong cùng một quy trình để người đọc theo dõi từ dữ liệu đến kết quả phân loại.

Website được xây dựng bằng Python Flask, tiếp nhận nội dung do người dùng nhập và sử dụng **mô hình Transformer** để trả về nhãn HAM hoặc SPAM cùng điểm dự đoán SPAM. MLP và LSTM được dùng trong phần so sánh thực nghiệm, chưa được tích hợp thành lựa chọn mô hình trên website. Phạm vi sản phẩm là một ứng dụng minh họa chạy cục bộ; các nhánh xử lý đã được kiểm thử trên một máy, chưa kiểm thử chịu tải.

Các hướng như chia lại dữ liệu để loại trùng lặp giữa các tập, thu thập thêm SMS lừa đảo thật và hiện đại, và khảo sát nhiều seed là các đề xuất hoàn thiện tiếp theo, chưa được xem là thí nghiệm đã hoàn thành.

## MĐ.5. Bố cục báo cáo

Ngoài phần Mở đầu và Kết luận, báo cáo được tổ chức thành năm chương:

- **Chương 1 — Cơ sở lý thuyết:** Trình bày phân loại văn bản, biểu diễn dữ liệu, nguyên lý của MLP/LSTM/Transformer, phương pháp huấn luyện và các chỉ số đánh giá.
- **Chương 2 — Dữ liệu và tiền xử lý:** Giới thiệu SMS Spam Collection và hai bộ dữ liệu bổ sung, phân bố lớp, bộ chia dữ liệu, biểu diễn văn bản và hạn chế do trùng nội dung giữa các tập.
- **Chương 3 — Thiết kế mô hình và quy trình huấn luyện:** Mô tả kiến trúc, cấu hình thực nghiệm, hàm mất mát, tối ưu hóa, lựa chọn checkpoint và quy trình đánh giá chung.
- **Chương 4 — Kết quả thực nghiệm và so sánh:** Trình bày các chỉ số, ma trận nhầm lẫn, chi phí tính toán, kết quả trên dữ liệu ngoài nguồn và thí nghiệm bổ sung dữ liệu.
- **Chương 5 — Website demo phân loại SMS:** Mô tả chức năng, luồng xử lý, cách tích hợp Transformer vào ứng dụng Flask và kết quả kiểm thử.

Phần **Kết luận** tổng hợp mức độ đáp ứng mục tiêu, các hạn chế của đồ án và hướng phát triển dựa trên những nội dung đã trình bày.
