# CHƯƠNG 1 — CƠ SỞ LÝ THUYẾT

## 1.1. Phân loại văn bản và đặc điểm bài toán SMS

Phân loại văn bản là ánh xạ nội dung dạng chuỗi sang một hoặc nhiều nhãn định trước. Trong phân loại nhị phân của đồ án, mỗi SMS có nhãn thật $y\in\{0,1\}$, với 0 là HAM và 1 là SPAM. Gọi $s$ là văn bản, $\phi(s)$ là phép biểu diễn số và $f_\theta$ là mạng có tập tham số $\theta$. Mô hình tính logit $z=f_\theta(\phi(s))$, sau đó chuyển sang điểm SPAM bằng sigmoid. Như vậy, hệ thống được xác định bởi cả cách biểu diễn và mạng phân loại; cùng một kiến trúc nhưng khác bộ đặc trưng có thể cho hành vi khác nhau.

SMS thường chứa cách viết tắt, dấu câu và các chuỗi như số điện thoại hoặc địa chỉ web. Bộ phân loại cần học sự liên hệ giữa những dấu hiệu đó và nhãn trong dữ liệu, thay vì áp dụng quy tắc rằng mọi tin chứa URL đều là SPAM. Biểu diễn từ/ký tự và biểu diễn chuỗi token giữ những loại thông tin khác nhau, tạo cơ sở cho ba phương án được so sánh. Các nguyên lý phân loại văn bản và mô hình không gian vector được trình bày trong tài liệu của Manning và cộng sự [2].

Mất cân bằng lớp (class imbalance) xảy ra khi hai lớp có số mẫu khác nhau đáng kể. Nếu chỉ dự đoán lớp phổ biến, mô hình có thể có Accuracy tương đối cao mà không phát hiện được lớp ít mẫu. Với SMS, bỏ sót SPAM và chặn nhầm HAM cũng tạo ra hai hậu quả khác nhau. Vì vậy, đồ án dùng SPAM làm lớp dương và xem xét đồng thời Precision, Recall, F1 cùng ma trận nhầm lẫn, thay vì chỉ dùng Accuracy.

## 1.2. Chuẩn hóa, token, n-gram và TF-IDF

### 1.2.1. Chuẩn hóa và đơn vị văn bản

Chuẩn hóa văn bản đưa các biến thể hình thức về dạng thống nhất trước khi học. Các thao tác có thể gồm chuẩn hóa Unicode, chuyển chữ thường, gom khoảng trắng và thay chuỗi có cùng vai trò bằng ký hiệu đại diện. Chẳng hạn, placeholder `<URL>` giúp mô hình nhận biết sự xuất hiện của địa chỉ web mà không cần học riêng từng địa chỉ. Đây là phép biến đổi theo quy tắc, khác với học trọng số của mô hình. Quy trình hiện tại và giới hạn lịch sử của dữ liệu dự án được phân biệt ở Chương 2.

Token là đơn vị được bộ tách văn bản nhận diện; token có thể là từ, placeholder hoặc dấu câu. Từ vựng (vocabulary) là tập token có mã định danh. Token không đồng nghĩa hoàn toàn với từ trong ngôn ngữ: mã hiện tại có thể coi `!` là một token riêng. N-gram là một dãy liên tiếp gồm $n$ đơn vị. Word unigram giữ từng token, word bigram giữ cặp token liền nhau; character n-gram giữ các đoạn ký tự liên tiếp, có thể chứa khoảng trắng và dấu câu.

Đặc trưng từ thuận lợi cho việc học các dấu hiệu từ vựng và cụm ngắn. Đặc trưng ký tự cung cấp thông tin về cấu tạo chuỗi, hỗ trợ biểu diễn những biến thể không trùng hẳn một từ đã gặp. Tuy nhiên, hai loại này không giữ toàn bộ thứ tự của SMS: vector cho biết sự xuất hiện và trọng số của các n-gram đã chọn, không biểu diễn đầy đủ quan hệ xa giữa mọi token. Đây là đánh đổi của phương pháp không gian vector [2], [3].

### 1.2.2. Trọng số TF-IDF

TF-IDF viết tắt của *term frequency–inverse document frequency*, tức tần suất thuật ngữ kết hợp nghịch đảo tần suất tài liệu. Với thuật ngữ hoặc n-gram $t$, văn bản $d$, số văn bản train $N$ và số văn bản train chứa $t$ là $df(t)$, dạng có làm trơn mà `TfidfVectorizer` sử dụng theo mặc định là:

$$
idf(t)=\log\frac{1+N}{1+df(t)}+1,\qquad
u_{t,d}=tf(t,d)\,idf(t). \tag{1.1}
$$

Ở đây $tf(t,d)$ là số lần $t$ xuất hiện trong $d$ khi không bật biến đổi tần suất logarit; $u_{t,d}$ là trọng số trước chuẩn hóa vector. IDF làm giảm ảnh hưởng tương đối của những đặc trưng xuất hiện phổ biến trên nhiều tài liệu. Cộng 1 trong tử và mẫu tránh phép chia không xác định và là một lựa chọn triển khai cụ thể, không phải công thức duy nhất của TF-IDF [3].

Với chuẩn hóa L2, vector đầu ra của một nhánh là $x_d=u_d/\sqrt{\sum_t u_{t,d}^{2}}$ nếu mẫu có đặc trưng khác 0. Nhánh từ và nhánh ký tự trong dự án được tính riêng rồi nối bằng `FeatureUnion`. Do đó, không nên diễn giải vector ghép như thể chỉ có một phép chuẩn hóa L2 chung sau khi nối. Từ vựng đặc trưng, lựa chọn đặc trưng và IDF đều phải được học trên train; test chỉ được biến đổi bằng bộ đã học để tránh sử dụng trước thông tin của tập đánh giá.

## 1.3. Chuỗi token ID và embedding

Trong biểu diễn chuỗi, mỗi token được thay bằng số nguyên trong từ vựng. ID chỉ là chỉ số tra cứu, không có ý nghĩa số học như “token 8 lớn gấp đôi token 4”. Lớp embedding có ma trận $E\in\mathbb{R}^{V\times d_e}$, trong đó $V$ là kích thước từ vựng và $d_e$ là số chiều embedding. Token ID $k$ được ánh xạ thành hàng $E_k$. Những vector này được cập nhật bằng gradient trong huấn luyện, không lấy từ một mô hình ngôn ngữ pretrained.

Các SMS có số token khác nhau nên cần thống nhất kích thước tensor. Padding bổ sung mã `<PAD>` vào cuối chuỗi ngắn; truncation cắt chuỗi dài tới giới hạn $L$. Token chưa có trong từ vựng được ánh xạ sang `<UNK>`. Dự án dùng `<PAD>=0`, `<UNK>=1` và $L=100$. Nếu SMS bị cắt, mô hình chỉ nhận phần đầu 100 token; báo cáo không có thống kê tỷ lệ bị cắt vì hồ sơ thực nghiệm chưa lưu đại lượng này.

Padding là phần kỹ thuật, không phải nội dung SMS. Vì thế, BiLSTM cần chiều dài thật khi đóng gói chuỗi; Transformer cần mặt nạ padding và loại vị trí đệm khỏi pooling. Nếu bỏ qua điều này, phần bổ sung nhằm đồng nhất kích thước có thể làm thay đổi biểu diễn câu. Chi tiết hai cơ chế được trình bày ở Chương 3.

## 1.4. Mạng nơ-ron, MLP và lan truyền ngược

### 1.4.1. Logit, sigmoid và quyết định nhãn

Một lớp kết nối đầy đủ (fully connected, Dense) tính $a=Wx+b$, với vector đầu vào $x$, ma trận trọng số $W$ và vector độ lệch $b$. Ghép các lớp cùng hàm phi tuyến tạo thành mạng perceptron nhiều lớp (*multilayer perceptron*, MLP). Nếu chỉ ghép các phép biến đổi tuyến tính mà không có phi tuyến, toàn mạng vẫn tương đương một phép biến đổi affine, làm giảm khả năng mô tả tương tác phức tạp giữa đặc trưng.

Đồ án dùng hàm ReLU $\operatorname{ReLU}(a)=\max(0,a)$ theo từng thành phần và một đầu ra vô hướng $z$, gọi là logit. Hàm sigmoid và quy tắc phân loại là:

$$
p=\sigma(z)=\frac{1}{1+e^{-z}},\qquad
\widehat y=\begin{cases}1,&p\geq\tau,\\0,&p<\tau,\end{cases}\quad\tau=0{,}5. \tag{1.2}
$$

$p$ là điểm SPAM; $\widehat y$ là nhãn dự đoán. Với logit hữu hạn $z\in\mathbb R$, sigmoid toán học cho $0<p<1$ và ngưỡng $\tau=0{,}5$ tương ứng quy tắc dự đoán SPAM khi $z\geq0$. Trong tính toán số thực dấu phẩy động, điểm có thể làm tròn thành 0 hoặc 1. Điểm nằm trong $[0,1]$ không tự chứng minh rằng nó đã được hiệu chỉnh xác suất (*probability calibration*). Đặc biệt, loss có trọng số thay đổi cách mô hình trả giá cho lỗi lớp dương; cần đánh giá hiệu chỉnh riêng trước khi diễn giải điểm như tần suất đúng thực tế.

### 1.4.2. Dropout và cập nhật tham số

Dropout ngẫu nhiên loại một phần hoạt hóa trong lúc huấn luyện để hạn chế việc mạng phụ thuộc quá mức vào một số đường truyền. Theo cơ chế dropout chuẩn của PyTorch, các phần tử còn lại được điều chỉnh tỷ lệ trong train; khi gọi `eval()`, thao tác ngẫu nhiên này được tắt. Vì vậy, suy luận phải đặt mô hình vào chế độ đánh giá, bên cạnh việc tắt tính gradient. Trong MLP của dự án, tỷ lệ dropout là $0{,}4$ ở mỗi lớp ẩn; đây là xác suất loại hoạt hóa, không phải tỷ lệ loại mẫu dữ liệu.

Lan truyền ngược (*backpropagation*) dùng quy tắc đạo hàm hàm hợp để tính $\nabla_\theta\mathcal L$ từ loss đầu ra về các lớp trước. Với cập nhật gradient cơ bản, $\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L$, trong đó $\eta$ là learning rate. Adam và AdamW trong dự án thay quy tắc này bằng cập nhật thích nghi dựa trên thống kê gradient; nguyên lý tính gradient vẫn là lan truyền ngược. Bài báo của Rumelhart, Hinton và Williams là tài liệu nền tảng về học biểu diễn bằng cơ chế này [4].

## 1.5. LSTM và LSTM hai chiều

LSTM (*long short-term memory*, mạng bộ nhớ dài–ngắn hạn) là một dạng mạng hồi tiếp có trạng thái ô nhớ để kiểm soát thông tin được giữ và cập nhật qua chuỗi. Công trình của Hochreiter và Schmidhuber giới thiệu nền tảng LSTM [5]. Các phương trình dưới đây mô tả biến thể có cổng quên theo API `nn.LSTM` đang được dự án sử dụng [6]; không đồng nhất từng chi tiết với mọi biến thể trong bài báo ban đầu.

Với vector đầu vào tại bước $t$ là $x_t$, trạng thái ẩn trước đó $h_{t-1}$ và ô nhớ trước đó $c_{t-1}$:

$$
\begin{aligned}
i_t&=\sigma(W_i x_t+U_i h_{t-1}+b_i),\\
f_t&=\sigma(W_f x_t+U_f h_{t-1}+b_f),\\
g_t&=\tanh(W_g x_t+U_g h_{t-1}+b_g),\\
o_t&=\sigma(W_o x_t+U_o h_{t-1}+b_o),\\
c_t&=f_t\odot c_{t-1}+i_t\odot g_t,\\
h_t&=o_t\odot\tanh(c_t).
\end{aligned} \tag{1.3}
$$

Các $W$ biến đổi đầu vào, $U$ biến đổi trạng thái ẩn, $b$ là bias gộp để viết gọn, $\odot$ là phép nhân từng phần tử và $\tanh$ là tang hyperbolic. Cổng đầu vào $i_t$ điều tiết nội dung mới; cổng quên $f_t$ điều tiết phần ô nhớ cũ được giữ; $g_t$ là nội dung ứng viên; cổng đầu ra $o_t$ điều tiết phần ô nhớ được biểu hiện trong trạng thái ẩn. Đường cập nhật cộng của $c_t$ hỗ trợ truyền thông tin và gradient qua nhiều bước, nhưng không bảo đảm loại bỏ mọi khó khăn khi học chuỗi.

LSTM một chiều đọc SMS từ trái sang phải. LSTM hai chiều (*bidirectional LSTM*, BiLSTM) có thêm một nhánh đọc theo chiều ngược. Hai nhánh có tham số riêng, cho phép biểu diễn toàn câu kết hợp ngữ cảnh từ hai hướng. Vì SMS được nhập đầy đủ trước khi phân loại, việc đọc hai chiều phù hợp với luồng xử lý của đồ án.

Mã dự án không lấy trung bình toàn bộ trạng thái LSTM. Mạng nối trạng thái ẩn cuối của hai hướng thành vector 256 chiều, sau đó đưa qua đầu phân loại. Với hidden size 128 mỗi hướng, con số 256 là chiều vector ghép, không phải hidden size của một hướng. `pack_padded_sequence` dùng chiều dài thật để loại phần đệm khỏi quá trình hồi tiếp, tránh nhầm trạng thái tại token cuối với trạng thái sau một chuỗi PAD.

## 1.6. Transformer encoder nhỏ

### 1.6.1. Embedding token và vị trí

Transformer dùng attention để trao đổi thông tin giữa các vị trí trong chuỗi. Cơ sở kiến trúc được trình bày trong *Attention Is All You Need* [7]. Dự án chỉ dùng phần encoder, không có decoder sinh văn bản. Token embedding và positional embedding đều được học từ đầu:

$$
X_t=E_{k_t}+P_t,\quad t=0,\ldots,L-1. \tag{1.4}
$$

$k_t$ là token ID tại vị trí $t$, $E_{k_t}\in\mathbb R^{128}$ và $P_t\in\mathbb R^{128}$ là vector vị trí. Attention thuần không tự biết thứ tự; cộng $P_t$ cung cấp thông tin này. Mã hiện tại dùng bảng vị trí học được, không dùng công thức vị trí sin/cos và không nhân token embedding với $\sqrt{d_{\mathrm{model}}}$ trước phép cộng.

### 1.6.2. Scaled dot-product attention và nhiều đầu

Từ đầu vào $X$, mỗi đầu attention tạo truy vấn (*query*) $Q=XW_Q$, khóa (*key*) $K=XW_K$ và giá trị (*value*) $V=XW_V$. Các ma trận $W_Q$, $W_K$, $W_V$ là trọng số chiếu học được; $V$ trong công thức attention là ma trận value, khác ký hiệu kích thước từ vựng ở Mục 1.3. Cơ chế attention tích vô hướng có điều chỉnh tỷ lệ là:

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V. \tag{1.5}
$$

$X\in\mathbb R^{L\times d_{\mathrm{model}}}$ gồm $L$ vector token của một SMS; $Q,K\in\mathbb R^{L\times d_k}$, $V\in\mathbb R^{L\times d_v}$ và $M\in\mathbb R^{L\times L}$. Ở đây $d_k$ là số chiều key/query và $d_v$ là số chiều value của một đầu. Softmax áp dụng theo các vị trí key cho từng query; $M$ là mặt nạ cộng: 0 ở vị trí được phép chú ý, giá trị âm vô cùng ở key PAD. Mỗi query phải còn ít nhất một key không bị che để softmax được xác định, tức SMS phải có ít nhất một token khác PAD. Bộ mã hóa của dự án dùng `<UNK>` khi không tách được token; đây là cách xử lý trường hợp tách token rỗng. Công thức chiếu $Q,K,V$ viết gọn, không hiển thị các bias mà lớp thư viện có thể dùng.

Chia cho $\sqrt{d_k}$ kiểm soát độ lớn của tích vô hướng trước softmax. Sau khi nhân với $V$, đầu ra một đầu có dạng $L\times d_v$; mỗi hàng trọng số attention quyết định mức đóng góp của các vector value vào biểu diễn mới của một token.

Attention nhiều đầu (*multi-head attention*) thực hiện nhiều phép biến đổi $Q,K,V$, nối kết quả và chiếu lại bằng ma trận trọng số học được $W_O$. Trong dự án, $d_{\mathrm{model}}=128$, số đầu là 4 nên $d_k=d_v=32$ cho mỗi đầu. Các đầu cho phép học các quan hệ khác nhau, nhưng chưa có phân tích trọng số attention để gán một chức năng ngữ nghĩa cụ thể cho từng đầu. Encoder không dùng causal mask; token nội dung có thể chú ý tới các token nội dung ở cả hai phía.

### 1.6.3. Feed-forward, residual, LayerNorm và pooling

Mạng truyền thẳng theo vị trí (*position-wise feed-forward network*, FFN) biến đổi từng vector token bằng cùng bộ trọng số:

$$
\operatorname{FFN}(H)=\operatorname{ReLU}(HW_1+b_1)W_2+b_2. \tag{1.6}
$$

Trong Công thức (1.6), $H$ là các vector đầu vào, $W_1,W_2$ là ma trận trọng số, $b_1,b_2$ là bias của hai phép chiếu. Trong dự án, FFN có kích thước $128\rightarrow256\rightarrow128$. Mỗi encoder kết hợp attention và FFN với kết nối dư (*residual connection*), tức cộng đầu vào của một nhánh với đầu ra nhánh đó. Chuẩn hóa lớp (*layer normalization*, LayerNorm) chuẩn hóa các thành phần của vector tại từng vị trí rồi học hệ số co giãn và dịch chuyển. Với `norm_first=False`, thứ tự khái quát là:

$$
H=\operatorname{LN}(X+\operatorname{Dropout}(\operatorname{MHA}(X))),\quad
Z=\operatorname{LN}(H+\operatorname{Dropout}(\operatorname{FFN}(H))). \tag{1.7}
$$

LN là LayerNorm, MHA là attention nhiều đầu. Công thức khái quát bỏ qua vị trí dropout trung gian để dễ đọc; `TransformerEncoderLayer` còn đặt dropout trong attention và FFN theo thư viện [8]. Mô hình có hai encoder và dropout nội bộ $0{,}1$, dùng ReLU mặc định. Đây là các thành phần của lớp thư viện được gọi trong mã, không phải các lớp do tác giả đồ án viết lại thủ công.

Sau encoder, dự án dùng trung bình có mặt nạ (*masked mean pooling*):

$$
v=\frac{\sum_{t=0}^{L-1}m_t Z_t}{\max\!\left(1,\sum_{t=0}^{L-1}m_t\right)},\qquad
m_t=\begin{cases}1,&k_t\ne0,\\0,&k_t=0.\end{cases} \tag{1.8}
$$

$Z_t$ là vector đầu ra encoder, $m_t$ xác định token nội dung và $v$ là vector SMS 128 chiều. Chỉ số $t=0,\ldots,L-1$ thống nhất với Công thức (1.4); $\max(1,\ldots)$ tương ứng `clamp(min=1)` trong mã. Mặt nạ pooling ngăn phần đệm đóng góp vào trung bình; mặt nạ attention ngăn PAD đóng vai trò key. Hai thao tác có vai trò khác nhau và cùng cần thiết trong triển khai này. Đầu phân loại dùng dropout $0{,}4$ và Dense ra một logit, không có token `[CLS]` hoặc mô hình BERT.

## 1.7. Loss có trọng số, tối ưu hóa và chọn checkpoint

Với $B$ mẫu trong một batch, loss nhị phân có trọng số dương được viết:

$$
\mathcal L=-\frac{1}{B}\sum_{i=1}^{B}\left[w_+y_i\log p_i+(1-y_i)\log(1-p_i)\right],\qquad
w_+=\frac{N_{HAM}^{train}}{N_{SPAM}^{train}}. \tag{1.9}
$$

$y_i$ là nhãn thật, $p_i$ là sigmoid của logit, $N_{HAM}^{train}$ và $N_{SPAM}^{train}$ là số mẫu hai lớp trên train. Công thức logarit được viết với $B>0$, $0<p_i<1$ và train có cả hai lớp. Dự án dùng `BCEWithLogitsLoss`, gộp sigmoid và cross-entropy trong phép tính ổn định số học từ logit, thay vì tính trực tiếp logarit của điểm có thể đã làm tròn thành 0 hoặc 1; không gọi sigmoid trước khi đưa logit vào loss [9]. Với dữ liệu hiện có, $w_+=3377/523\approx6{,}457$. Hệ số này tăng chi phí thành phần lỗi của mẫu SPAM. Nó không thay đổi số mẫu và không bảo đảm Recall cao hơn trong mọi lượt chạy.

Adam dùng ước lượng trượt của gradient và bình phương gradient để điều chỉnh cập nhật từng tham số. AdamW tách tác dụng weight decay khỏi cập nhật thích nghi [11]. Dự án dùng Adam cho MLP/BiLSTM, AdamW cho Transformer. Không có một bộ giá trị learning rate đủ để mô tả toàn bộ lịch huấn luyện: mã Transformer còn tăng dần learning rate lúc đầu (*warmup*) rồi giảm tuyến tính theo số bước. Phân biệt giá trị khởi tạo trong config với learning rate tại mỗi bước là cần thiết khi tái lập.

Train dùng để học tham số và bộ biểu diễn. Validation dùng để so sánh các checkpoint trong quá trình huấn luyện. Test được giữ cho đánh giá sau chọn mô hình. Nếu dùng test để chọn epoch hoặc chỉnh ngưỡng, kết quả cuối sẽ không còn là đánh giá tách biệt với quyết định lựa chọn.

Một epoch là một lần đi qua các batch của train. Early stopping là dừng khi chỉ số validation không cải thiện sau số epoch liên tiếp quy định bởi *patience*. Mã dự án chọn F1 SPAM validation cao nhất tại ngưỡng $0{,}5$, chỉ cập nhật khi F1 mới lớn hơn nghiêm ngặt F1 tốt nhất. Vì vậy, nếu bằng điểm thì giữ checkpoint ở epoch đầu tiên; epoch cuối chạy có thể khác epoch được lưu trọng số. Validation F1 được dùng để chọn mô hình nên không thay thế cho kết quả test.

## 1.8. Chỉ số đánh giá và chi phí tính toán

**Bảng 1.1. Ý nghĩa các ô ma trận nhầm lẫn trong phân loại SMS.**

| Nhãn thật | Dự đoán HAM | Dự đoán SPAM |
|---|---|---|
| HAM | TN: tin hợp lệ được giữ đúng | FP: tin hợp lệ bị gán nhầm là rác |
| SPAM | FN: tin rác bị bỏ sót | TP: tin rác được phát hiện đúng |

*Nguồn: quy ước lớp dương SPAM trong `eval/evaluate.py`; định nghĩa chỉ số theo [10]. Hàng là nhãn thật, cột là dự đoán.*

Với $N=TN+FP+FN+TP$, các chỉ số tại ngưỡng đã chọn là:

$$
Accuracy=\frac{TP+TN}{N},\quad Precision=\frac{TP}{TP+FP},\quad
Recall=\frac{TP}{TP+FN},\quad F1=\frac{2TP}{2TP+FP+FN}. \tag{1.10}
$$

Precision phản ánh độ chính xác của nhóm bị đánh dấu SPAM; Recall phản ánh tỷ lệ phát hiện trong SPAM thật. F1 là trung bình điều hòa của Precision và Recall, không sử dụng trực tiếp TN. Accuracy phản ánh tất cả mẫu nên chịu ảnh hưởng lớn hơn từ lớp HAM đông hơn. Công thức Accuracy yêu cầu $N>0$; ba tỷ số còn lại áp dụng trực tiếp khi mẫu số tương ứng khác 0. Nếu mẫu số Precision, Recall hoặc F1 bằng 0, lời gọi scikit-learn của dự án dùng `zero_division=0`; bộ tính metric đồng thời từ chối tập nhãn rỗng. Trong các ma trận test hiện tại, các mẫu số đều khác 0.

Đường ROC biểu diễn $TPR=TP/(TP+FN)$ theo $FPR=FP/(FP+TN)$ khi thay đổi ngưỡng; TPR là tỷ lệ phát hiện SPAM và FPR là tỷ lệ HAM bị gán nhầm SPAM. ROC-AUC là diện tích dưới đường này. Với đánh giá nhị phân không trọng số và có cả hai lớp, cách viết tương đương theo thứ hạng điểm là:

$$
\operatorname{AUC}=\frac{1}{n_+n_-}\sum_{i=1}^{n_+}\sum_{j=1}^{n_-}g(s_i^+-s_j^-),\qquad
g(a)=\begin{cases}1,&a>0,\\\frac{1}{2},&a=0,\\0,&a<0.\end{cases} \tag{1.11}
$$

$n_+>0$ và $n_->0$ lần lượt là số SPAM và HAM; $s_i^+$ và $s_j^-$ là điểm SPAM của mẫu dương thứ $i$ và mẫu âm thứ $j$; $a$ là hiệu hai điểm. Hàm $g$ cho một điểm nếu mẫu SPAM được xếp cao hơn, nửa điểm nếu hòa và 0 nếu thấp hơn. Khi chỉ có một lớp, AUC nhị phân không xác định; `calculate_metrics` của dự án trả `None`, không gán AUC bằng 0. Trong test hiện lưu có cả hai lớp. Công thức (1.11) nêu định nghĩa lý thuyết, không phải phép tính lại AUC đã lưu vì hồ sơ không có điểm từng mẫu.

AUC đánh giá thứ tự điểm, còn F1/Accuracy trong đồ án đánh giá nhãn tại ngưỡng $0{,}5$. Hai loại chỉ số có thể tạo thứ hạng mô hình khác nhau [10]. Một AUC tổng hợp không đủ khôi phục đường ROC hay phân bố điểm theo từng mẫu.

Độ trễ (*latency*) và số tham số mô tả hai mặt khác của mô hình. Số tham số là tổng số phần tử tensor tham số, không bao gồm chi phí lưu bộ TF-IDF hoặc mọi bộ nhớ trung gian. Latency phụ thuộc số học của kiến trúc, bước tạo đặc trưng, kích thước batch, thư viện và máy đo; ít tham số hơn không đồng nghĩa luôn nhanh hơn. Dự án báo cáo thời gian trung bình trên mỗi SMS ở batch 1 và batch 32. Batch 32 là xử lý nhiều tin trong một lượt, khác với độ trễ tương tác của một yêu cầu web đơn lẻ.

Các khái niệm trên tạo quy ước chung cho báo cáo: phân biệt chất lượng tại một ngưỡng với khả năng xếp hạng; phân biệt chi phí mô hình với thời gian xử lý cả bước biểu diễn; và phân biệt cấu trúc thực hiện trong mã với những điều đã có bằng chứng từ lượt chạy được lưu.
