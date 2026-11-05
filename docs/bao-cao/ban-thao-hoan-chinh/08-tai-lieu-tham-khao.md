# TÀI LIỆU THAM KHẢO

Các trích dẫn dùng số `[1]`–`[18]` thống nhất trong toàn báo cáo. Số liệu thực nghiệm được dẫn trực tiếp bằng đường dẫn tệp dự án dưới bảng/hình, tách khỏi nguồn lý thuyết bên ngoài. Tài liệu thư viện đã được đối chiếu thêm với mã và docstring có sẵn trong môi trường cục bộ; các liên kết công bố bên ngoài chưa thể truy cập trực tuyến trong lần biên soạn do công cụ web và trình duyệt không khả dụng. Vì vậy, danh mục không ghi ngày truy cập thành công hoặc tuyên bố đã kiểm tra nội dung trực tuyến của mọi tài liệu. Các phiên bản trang tài liệu hiện hành không được dùng để suy ra phiên bản môi trường Colab.

[1] UCI Machine Learning Repository. **SMS Spam Collection**. Trang dữ liệu chính thức: [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). Sử dụng để nhận diện nguồn dataset; số mẫu và nhãn của bản cục bộ được kiểm tra riêng từ CSV. Hồ sơ chưa có manifest lần tải để đối chiếu phiên bản cục bộ với bản công bố.

[2] C. D. Manning, P. Raghavan và H. Schütze. **Introduction to Information Retrieval**. Cambridge University Press, 2008. [Bản trực tuyến do nhóm tác giả cung cấp](https://nlp.stanford.edu/IR-book/). Thông tin tác giả, năm và tên sách có trong phần References của `sklearn/feature_extraction/text.py` cục bộ. Dùng cho biểu diễn không gian vector, trọng số thuật ngữ và phân loại văn bản.

[3] scikit-learn developers. **TfidfVectorizer; TfidfTransformer**. Tài liệu API chính thức: [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html). Công thức IDF làm trơn, chuẩn hóa L2 và các mặc định được đối chiếu với docstring trong `sklearn/feature_extraction/text.py` cục bộ.

[4] D. E. Rumelhart, G. E. Hinton và R. J. Williams. **Learning representations by back-propagating errors**. *Nature*, 323, 1986. [Trang bài báo tại nhà xuất bản](https://www.nature.com/articles/323533a0). Tài liệu nền tảng về học biểu diễn bằng lan truyền ngược. Các trường tác giả, tiêu đề, năm, tập và URL đang ghi chưa được xác minh độc lập bằng bản bài báo hoặc trang nhà xuất bản trong hồ sơ kiểm định này; không xem việc có mục dẫn là bằng chứng đã đọc nguồn gốc.

[5] S. Hochreiter và J. Schmidhuber. **Long Short-Term Memory**. *Neural Computation*, 9(8), 1997. [DOI: 10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735). Tài liệu gốc LSTM; các phương trình biến thể có cổng quên trong báo cáo được đối chiếu trực tiếp với API tại [6]. Các trường thư mục và DOI này chưa được xác minh độc lập bằng bản bài báo hoặc bản ghi DOI trong hồ sơ kiểm định hiện tại.

[6] PyTorch contributors. **LSTM**. Tài liệu API `torch.nn.LSTM`: [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html). Các cổng, trạng thái ẩn/ô nhớ, hai chiều và chiều tensor được đối chiếu thêm với `torch/nn/modules/rnn.py` cục bộ.

[7] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser và I. Polosukhin. **Attention Is All You Need**. *Advances in Neural Information Processing Systems*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Tiêu đề và liên kết bài báo được đối chiếu từ docstring `torch/nn/modules/transformer.py` cục bộ; danh sách tác giả, năm và nơi công bố chưa được xác minh độc lập từ bản bài báo hoặc trang công bố trong lần kiểm định này.

[8] PyTorch contributors. **TransformerEncoderLayer; TransformerEncoder**. Tài liệu API chính thức: [TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html), [TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html). Thành phần encoder, mặc định ReLU, dropout và `norm_first` được đối chiếu thêm với `torch/nn/modules/transformer.py` cục bộ.

[9] PyTorch contributors. **BCEWithLogitsLoss**. [Tài liệu API](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html). Công thức loss có trọng số dương và ý nghĩa `pos_weight` được đối chiếu với `torch/nn/modules/loss.py` cục bộ.

[10] scikit-learn developers. **Model evaluation: quantifying the quality of predictions**. [Hướng dẫn đánh giá mô hình](https://scikit-learn.org/stable/modules/model_evaluation.html). Dùng cho Accuracy, Precision, Recall, F1, confusion matrix và ROC-AUC; đối chiếu với `sklearn/metrics/_classification.py`, `_ranking.py` cục bộ và lời gọi trong `eval/evaluate.py`.

[11] PyTorch contributors. **Adam; AdamW**. Tài liệu API chính thức: [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html), [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html). Đối chiếu thêm `torch/optim/adam.py`, `adamw.py` cục bộ; không dùng mặc định của bản tài liệu hiện hành để tự điền metadata lượt chạy cũ.

[12] Python Software Foundation. **csv — CSV File Reading and Writing**. [Tài liệu thư viện chuẩn Python](https://docs.python.org/3/library/csv.html). Dùng cho nguyên tắc đọc CSV có trường đặt trong dấu ngoặc kép và xuống dòng; hành vi triển khai dự án được đọc trực tiếp trong `data/prepare_data.py`.

[13] Python Software Foundation. **unicodedata — Unicode Database**. [Tài liệu `unicodedata.normalize`](https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize). Dùng để nhận diện NFKC; lựa chọn `"NFKC"` của dự án có trong mã `normalize_text` và được kiểm tra qua unit test `test_unicode_and_normalization_are_stable`.

[14] Pallets. **Flask Quickstart**. [Tài liệu Flask chính thức](https://flask.palletsprojects.com/en/stable/quickstart/). Dùng cho route, phương thức HTTP và nhận trường biểu mẫu; đối chiếu thêm package `flask` cục bộ và `app.py` của dự án.

[15] Pallets. **Flask Templating**. [Tài liệu template chính thức](https://flask.palletsprojects.com/en/stable/templating/). Dùng cho quan hệ Flask–Jinja và render HTML; đối chiếu với `flask/templating.py`, `jinja2` cục bộ và `templates/index.html`.

[16] PyTorch contributors. **Saving and Loading Models**. [Hướng dẫn lưu và nạp mô hình](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). Dùng cho `state_dict` và quy trình dựng mạng trước khi nạp trọng số; đối chiếu với `torch/serialization.py` cục bộ và `load_checkpoint` của dự án. Trong lần cập nhật ngày 05/11/2026, `load_checkpoint` đã nạp và chạy thành công sáu checkpoint baseline/augmented và checkpoint MLP của lượt Colab bằng PyTorch 2.15.0.

[17] electricsheepafrica. **africa-smishing-sms-phishing**. Hugging Face Datasets: [electricsheepafrica/africa-smishing-sms-phishing](https://huggingface.co/datasets/electricsheepafrica/africa-smishing-sms-phishing). Bộ 10.000 SMS tổng hợp (cột `is_synthetic` bằng 1 ở mọi dòng), tải ngày 05/11/2026 qua API `datasets-server`. Cột `language` không khớp ngôn ngữ thực của văn bản; mọi dòng được kiểm tra là câu tiếng Anh theo template. Về giấy phép, README của dataset ghi “Released under mit”, nhưng metadata Hugging Face ghi `license: other` và lưu ý quyền nguồn gốc thuộc về nhà cung cấp dữ liệu ban đầu, không nêu tên; hai thông tin này không thống nhất (đối chiếu ngày 05/11/2026). Đồ án chỉ dùng dữ liệu cho mục đích học tập, không phân phối lại.

[18] S. Mishra và D. Soni. **SMS Phishing Dataset for Machine Learning and Pattern Recognition**. Mendeley Data, V1, 2022. [DOI: 10.17632/f45bkkt8pr.1](https://data.mendeley.com/datasets/f45bkkt8pr/1). Giấy phép CC BY 4.0. Tệp `Dataset_5971.zip` tải ngày 05/11/2026, SHA-256 khớp giá trị do Mendeley công bố (`9bbf3188…33cc3`). Phần lớn bản ghi sao chép SMS Spam Collection [1]; chỉ phần không trùng được dùng làm tập test ngoài.
