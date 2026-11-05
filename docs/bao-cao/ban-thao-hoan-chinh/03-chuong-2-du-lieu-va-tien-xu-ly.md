# CHƯƠNG 2 — DỮ LIỆU VÀ TIỀN XỬ LÝ

## 2.1. Nguồn và đặc điểm dữ liệu

Đồ án sử dụng SMS Spam Collection, bộ dữ liệu phân loại SMS tiếng Anh được công bố trên UCI Machine Learning Repository [1]. Trong phạm vi dự án, bản dữ liệu là `data/raw/spam.csv`, có cột `v1` chứa nhãn và `v2` chứa nội dung. Không có manifest tải dữ liệu hoặc checksum từ nhà cung cấp để chứng minh phiên bản CSV cục bộ trùng hoàn toàn từng byte với bản công bố. Vì vậy, các số lượng trong chương được xác định trực tiếp từ CSV của dự án; thông tin UCI được dùng để nhận diện nguồn dataset, không thay cho kiểm tra bản cục bộ.

HAM biểu thị tin nhắn thông thường, SPAM biểu thị tin nhắn rác. Đây là các nhãn có sẵn trong dataset, không phải kết quả người thực hiện đồ án gán lại từng SMS. Báo cáo không mở rộng nhãn SPAM thành nhãn chuyên biệt cho phishing hoặc suy ra mọi SMS rác đều có ý định lừa đảo. Phạm vi thực nghiệm là học và đánh giá theo hai nhãn hiện có.

Việc kiểm tra bằng bộ đọc CSV cho thấy toàn bộ dữ liệu có 5.572 bản ghi, gồm 4.825 HAM và 747 SPAM. Cần đếm bản ghi đã phân tích theo quy tắc CSV, không đếm dòng vật lý trong tệp: một trường được đặt trong dấu ngoặc kép có thể chứa ký tự xuống dòng. Nguyên tắc này cũng được phản ánh trong ca kiểm thử đọc SMS có xuống dòng tại `tests/test_prepare_data.py` và trong tài liệu mô-đun `csv` của Python [12].

## 2.2. Phân bố nhãn và vai trò các tập

**Bảng 2.1. Quy mô dữ liệu và phân bố hai lớp trong các CSV hiện lưu.**

| Phạm vi | Tổng SMS | HAM | SPAM | Tỷ lệ SPAM |
|---|---:|---:|---:|---:|
| Toàn bộ dữ liệu | 5.572 | 4.825 | 747 | 13,41% |
| Train | 3.900 | 3.377 | 523 | 13,41% |
| Validation | 836 | 724 | 112 | 13,40% |
| Test | 836 | 724 | 112 | 13,40% |

*Nguồn: đếm `data/raw/spam.csv` và `data/splits/{train_base,val,test}.csv`, tổng hợp trong `tong-hop-so-lieu.json`; tỷ lệ bằng số SPAM chia tổng mẫu, làm tròn hai chữ số thập phân. Hàng Train là `train_base.csv`, tức phần UCI dùng cho kết quả chính.*

Tỷ lệ mục tiêu 70/15/15 và seed 42 được ghi trong `PLAN.md`, `README.md` và hồ sơ thực nghiệm. Số mẫu thực tế gần tỷ lệ đó, có điều chỉnh do số lượng phải là số nguyên. Chỉ từ các CSV, không thể chứng minh ngược thuật toán ngẫu nhiên hoặc seed tạo split; thông tin seed được trình bày theo hồ sơ, không được coi là kết quả suy luận từ số dòng.

Train cung cấp nội dung để học TF-IDF, xây từ vựng và cập nhật tham số mạng. Validation phục vụ theo dõi F1 và lựa chọn epoch. Test có 836 SMS được dùng chung cho ba mô hình sau khi chọn checkpoint. Dùng cùng test giúp đối chiếu số lỗi trên cùng tập đánh giá; tính độc lập giữa các tập vẫn phải được kiểm tra riêng vì các SMS có thể trùng nội dung.

Phân bố lớp có hệ quả trực tiếp đối với đánh giá. Trên test này, một quy tắc luôn trả HAM sẽ đạt $724/836\approx86{,}60\%$ Accuracy nhưng Recall SPAM bằng 0. Đây là phép tính tham chiếu từ phân bố nhãn, không phải một mô hình baseline đã được huấn luyện hoặc chạy trong dự án. Nó cho thấy vì sao đồ án ưu tiên xem thêm Precision, Recall và F1 của SPAM.

## 2.3. Cấu trúc tệp và quy trình đọc dữ liệu

Chuỗi các bước từ đọc văn bản, tạo biểu diễn đến huấn luyện hoặc suy luận được gọi là quy trình xử lý (*pipeline*). Mỗi trường dữ liệu có vai trò riêng trong quy trình này; giữ văn bản trước và sau xử lý giúp truy nguyên đầu vào mà mô hình thực sự nhận.

**Bảng 2.2. Các trường dữ liệu được sử dụng trong pipeline.**

| Tệp | Trường | Vai trò |
|---|---|---|
| CSV gốc | `v1` | Nhãn HAM/SPAM dưới dạng văn bản |
| CSV gốc | `v2` | Nội dung SMS trong bản CSV đầu vào |
| CSV split | `label` | Nhãn đã tổ chức cho pipeline |
| CSV split | `raw_text` | Văn bản được lưu trước bước chuẩn hóa của quy trình tạo split |
| CSV split | `text` | Văn bản mà train/evaluate hiện đọc để tạo đặc trưng |

*Nguồn: header CSV hiện có; `load_rows` trong `data/prepare_data.py` và `read_split` trong `models/common.py`.*

Trong mã chuẩn bị dữ liệu hiện tại, `load_rows` thử giải mã UTF-8 có xử lý BOM, rồi cp1252 nếu giải mã thất bại; nhận diện dấu phân cách và tên cột theo một số dạng thông dụng. Hàm kiểm tra nhãn, từ chối SMS rỗng và lưu cả `raw_text` lẫn `text`. Các biện pháp này nhằm báo lỗi dữ liệu đầu vào rõ ràng, không phải cơ chế sửa mọi lỗi mã hóa có thể tồn tại.

Trong mã huấn luyện và đánh giá, `read_split` mở CSV bằng `utf-8-sig`, yêu cầu có `label` và `text`, chuẩn hóa cách viết nhãn rồi mã hóa HAM thành 0, SPAM thành 1. Hàm từ chối nhãn ngoài hai lớp, văn bản rỗng và tập không có mẫu. Nó đọc trực tiếp cột `text`; không gọi lại `normalize_text`. Vì vậy, nội dung đã lưu trong cột này mới là đầu vào thực tế theo mã hiện tại, không phải kết quả của việc áp dụng tự động phiên bản chuẩn hóa mới mỗi lần chạy.

## 2.4. Chuẩn hóa văn bản và giới hạn lịch sử

### 2.4.1. Quy trình đang có trong mã

`normalize_text` sử dụng Unicode NFKC, chuyển chữ thường và xử lý chuỗi theo thứ tự xác định. NFKC là chuẩn hóa tương thích: một số cách biểu diễn Unicode khác nhau có thể được chuyển về dạng thống nhất; nó không phải thao tác dịch ngôn ngữ hoặc sửa mọi từ viết sai [13].

**Bảng 2.3. Các bước của hàm chuẩn hóa hiện tại và ý nghĩa.**

| Bước trong mã | Tác dụng chính | Điểm cần lưu ý |
|---|---|---|
| NFKC rồi lowercase | Giảm biến thể Unicode và hoa/thường | Có thể làm mất phân biệt chữ hoa |
| Sửa chuỗi lỗi `å£` | Xử lý một mẫu lỗi mã hóa ký hiệu bảng Anh | Không phải bộ sửa mã hóa tổng quát |
| Thay URL bằng `<URL>` | Gom địa chỉ web về cùng vai trò, kể cả tên miền không có `http`/`www` như `bit.ly/...`, `t.me/...` | Tên miền trần chỉ được nhận khi có đường dẫn hoặc đuôi `.com`, `.net`, `.org`, `.info`, `.biz`, `.co.uk` |
| Tách điện thoại gắn cước dạng pence | Phân biệt số điện thoại và khoản phí trong một số dạng Anh | Có mẫu riêng, không bao phủ mọi cách viết |
| Thay số tiền bằng `<MONEY>` | Giảm biến thể giá trị tiền, kể cả mã ISO đứng trước số như `ngn25000`, `usd 500` | Có thể giữ ký hiệu tiền tệ như `$` bên cạnh placeholder; chỉ nhận 15 mã ISO liệt kê trong mã |
| Thay điện thoại bằng `<PHONE>` | Giảm biến thể số liên hệ | Kiểm tra số chữ số và tránh một số dạng ngày tháng |
| Chuẩn hóa placeholder và khoảng trắng | Giữ `<URL>`, `<PHONE>`, `<MONEY>` viết hoa, gom khoảng trắng | Kết quả có thể chứa dấu câu có ích |

*Nguồn: `normalize_text`, `replace_url`, `replace_phone` và các biểu thức chính quy trong `data/prepare_data.py`; hành vi được kiểm tra bằng các ca trong `tests/test_prepare_data.py`. Đây là phiên bản đã sinh cột `text` của mọi split hiện tại.*

Giữ các dấu như `!`, `$`, `%` giúp không loại bỏ trước các đặc trưng mà mô hình có thể học. Thứ tự xử lý có ý nghĩa: thay URL trước giúp không nhận nhầm các chữ số hoặc ký hiệu tiền nằm trong URL thành thành phần độc lập của SMS. Ngược lại, placeholder làm mất giá trị cụ thể của URL, số điện thoại hoặc khoản tiền, nên đây là đánh đổi giữa giảm độ thưa và giữ chi tiết.

### 2.4.2. Sinh lại cột `text` cho các split

Các CSV ban đầu được tạo bằng một phiên bản chuẩn hóa cũ không còn mã nguồn. Để dữ liệu huấn luyện, đánh giá và website dùng cùng một quy trình, `data/augment_smishing.py` đã tính lại cột `text` từ `raw_text` bằng hàm hiện tại cho cả train, validation và test. Thành viên, nhãn và thứ tự các dòng giữ nguyên; bản CSV trước khi sinh lại được lưu tại `data/splits/backup_20261105_173206/`.

So với phiên bản cũ, 169 dòng train, 32 dòng validation và 39 dòng test có `text` thay đổi. Phần lớn là cước dạng `150p` hay `10p/min` trước đây chưa được thay bằng `<MONEY>`, và tên miền như `lucozade.co.uk/wrc` trước đây chưa được thay bằng `<URL>`. Một số URL bị tách dấu cách trong dữ liệu gốc, như `http://tms. widelive.com/index. wml`, được thay thành hai placeholder liền nhau; đây là hạn chế của quy tắc dựa trên biểu thức chính quy.

Kết quả chính ở Chương 4 dùng các CSV sau khi sinh lại, nên có thể tái lập từ `raw_text` bằng mã hiện tại. Lượt chạy Colab trước đó dùng cột `text` cũ và chỉ được giữ để đối chiếu (mục 4.6).

## 2.5. Biểu diễn riêng cho từng kiến trúc

### 2.5.1. MLP: nối TF-IDF từ và ký tự

`feature_encoder` tạo một `FeatureUnion` gồm nhánh từ 1–2 gram, tối đa 10.000 đặc trưng, và nhánh ký tự 3–5 gram, tối đa 5.000 đặc trưng. Cấu hình MLP đã lưu có 15.000 đặc trưng thực tế. Nhánh từ dùng biểu thức tách có hỗ trợ placeholder và `!`, `$`, `%`, đặt `lowercase=False`; nhánh ký tự dùng `analyzer="char"` và giữ thiết lập lowercase mặc định của thư viện. Điều này cho thấy bộ biểu diễn còn có lựa chọn nội bộ riêng, không thay thế cho một hàm chuẩn hóa chung đã được xác định rõ [3].

Ma trận TF-IDF ban đầu là dạng thưa (sparse), nhưng `transform_texts` chuyển thành mảng đặc `float32` trước khi đưa vào MLP. Vì vậy, thời gian chuyển đổi và bộ nhớ cho đặc trưng đặc cũng thuộc triển khai thực tế. Không nên đồng nhất MLP này với một bộ phân loại tuyến tính chạy hoàn toàn trên ma trận thưa.

### 2.5.2. BiLSTM và Transformer: token ID từ train

Hai mô hình chuỗi dùng cùng cách xây từ vựng: đếm token trên train, sắp theo tần suất giảm dần rồi theo tên token để xử lý trường hợp bằng tần suất, dành hai mã đầu cho PAD/UNK và giới hạn tối đa 10.000 mục kể cả hai mã đặc biệt. Hai config baseline đều ghi kích thước thực tế 6.856 và `max_len=100`; lượt Colab với cột `text` cũ ghi 6.886.

Biểu thức tách token nhận placeholder viết hoa, chuỗi ký tự thuộc `\w+` và các dấu `!`, `$`, `%`; nó không giữ mọi dấu câu thành token độc lập. Điều này khác với bước chuẩn hóa vốn không xóa toàn bộ dấu câu, và khác với nhánh character TF-IDF có thể giữ dấu câu trong n-gram. Nếu văn bản không sinh token, bộ mã hóa dùng `<UNK>` để tránh chuỗi hoàn toàn rỗng.

Hai lượt huấn luyện xây từ vựng bằng cùng hàm trên cùng train nên được thiết kế để có ánh xạ nhất quán. Việc config cùng ghi 6.856 mục xác nhận kích thước; báo cáo chưa so sánh trực tiếp từng ánh xạ ID trong hai `encoder.joblib`.

### 2.5.3. Vì sao chỉ fit trên train

TF-IDF học cả tập đặc trưng và độ phổ biến của mỗi đặc trưng; từ vựng chuỗi học token nào được coi là đã biết. Nếu đưa test vào bước này, hệ thống đã dùng thông tin phân bố test trước đánh giá, dù không trực tiếp dùng nhãn. Mã hiện tại chỉ gọi `fit(train_texts)` hoặc `build_vocab(train_texts)`, rồi dùng cùng encoder để biến đổi validation/test. Đây là biện pháp chống rò rỉ ở bước học biểu diễn; nó không tự khắc phục rò rỉ do nội dung trùng giữa các tập.

## 2.6. Kiểm tra trùng nội dung và ảnh hưởng

Phép kiểm tra dùng hàm `summarize_splits` trong `data/prepare_data.py`. Với từng cột `raw_text` và `text`, hàm tạo tập chuỗi của tập bên trái rồi xét từng dòng bên phải xem có nằm trong tập đó hay không. Phép so khớp chính xác và phân biệt hoa/thường; không dùng độ giống ngữ nghĩa.

**Bảng 2.4. Nội dung trùng giữa các tập theo từng cột hiện lưu.**

| Cột | Cặp trái → phải | Số nội dung khác nhau cùng xuất hiện | Số dòng ở tập phải trùng tập trái | Trong đó dòng SPAM |
|---|---|---:|---:|---:|
| `raw_text` | Train → Validation | 69 | 77 | 20 |
| `raw_text` | Train → Test | 73 | 81 | 20 |
| `raw_text` | Validation → Test | 19 | 25 | 6 |
| `text` | Train → Validation | 72 | 81 | 24 |
| `text` | Train → Test | 76 | 84 | 23 |
| `text` | Validation → Test | 19 | 25 | 6 |

*Nguồn: `summarize_splits` trên `train_base.csv`, `val.csv`, `test.csv`; kết quả trong hồ sơ biên tập `tong-hop-so-lieu.json`. “Số dòng” tính cả nhiều bản ghi có cùng nội dung; các cặp không được cộng trực tiếp để suy ra số bản ghi trùng duy nhất.*

Trên đầu vào `text`, 84/836 dòng test, tương đương khoảng 10,05%, có nội dung đã xuất hiện trong train. Trong đó có 23/112 dòng SPAM test, tương đương khoảng 20,54% của lớp dương. Chuẩn hóa mới chỉ thêm một dòng trùng so với cột `text` cũ (83 dòng). Đây là tỷ lệ hiện diện nội dung trùng, không phải tỷ lệ dự đoán đúng nhờ ghi nhớ. Không có xác suất từng mẫu và thí nghiệm đối chứng để xác định số lỗi chịu ảnh hưởng hoặc mô hình nào được hưởng lợi nhiều hơn.

Hiện tượng này được gọi là rò rỉ do nội dung trùng (*duplicate leakage*). Nó làm giảm mức độc lập của đánh giá vì một phần nội dung test đã xuất hiện khi học. Trùng giữa train và validation còn có thể ảnh hưởng việc chọn checkpoint; trùng giữa validation và test làm hai bước lựa chọn và đánh giá gần nhau hơn về nội dung. Cùng dùng một test bảo đảm cùng tập so sánh, nhưng không làm biến mất hạn chế này.

Mã chuẩn bị dữ liệu đã có phương án gom nhóm theo `text` sau chuẩn hóa, giữ nguyên các bản ghi trong cùng nhóm và phân bổ toàn nhóm vào một tập. Phương án này chưa được áp dụng: thành viên các tập được giữ nguyên để kết quả so sánh được với lượt chạy trước. Chia lại theo nhóm, huấn luyện và đánh giá lại thuộc công việc tiếp theo.

## 2.7. Dữ liệu bổ sung ngoài SMS Spam Collection

Hai nguồn dữ liệu được bổ sung để kiểm tra khả năng tổng quát; chi tiết thí nghiệm ở mục 4.7.

**Bảng 2.5. Các tập dữ liệu bổ sung.**

| Tập | Nguồn | Tổng SMS | HAM | SPAM | Vai trò |
|---|---|---:|---:|---:|---|
| Phần bổ sung trong `train.csv` | africa-smishing, tổng hợp [17] | 3.906 | 3.187 | 719 | Train của thí nghiệm augmented |
| `test_smishing.csv` | africa-smishing, tổng hợp [17] | 689 | 562 | 127 | Test cùng nguồn tổng hợp |
| `test_external.csv` | Mendeley SMS Phishing, tin thật [18] | 387 | 221 | 166 | Test ngoài nguồn |

*Nguồn: `data/splits/augmentation_manifest.json`, đếm CSV trong `tong-hop-so-lieu.json`. SPAM của `test_external.csv` gộp 102 tin nhãn smishing và 64 tin nhãn spam gốc.*

Bộ africa-smishing có 10.000 dòng, nhưng nhiều dòng chỉ khác nhau ở mã giao dịch, số dư hoặc ngày tháng. Sau chuẩn hóa còn 4.595 văn bản khác nhau; khi thay mọi chữ số bằng 0 còn 1.109 template. Việc chia train/test theo template, không theo dòng, ngăn hai tin gần giống nhau rơi vào hai phía. Bộ Mendeley có 5.971 dòng nhưng phần lớn sao chép SMS Spam Collection; chỉ 387 dòng không trùng hoặc gần trùng UCI được giữ lại. Không dòng nào của hai tập test bổ sung trùng `text` với train, validation hoặc test UCI.
