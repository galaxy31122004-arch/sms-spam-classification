from pathlib import Path

import torch
from flask import Flask, render_template, request

from data.prepare_data import normalize_text
from models.common import load_checkpoint, transform_texts

# Lấy đường dẫn thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent

# Giới hạn số luồng CPU để phù hợp với máy tính
torch.set_num_threads(4)

# Nạp Transformer train trên dữ liệu gốc (tốt nhất trên tập test ngoài, F1 0.976),
# không huấn luyện lại. Xem docs/smishing-augmentation-log.md.
MODEL_KIND = "transformer"
MODEL_DIR = BASE_DIR / "models" / "transformer" / "demo_checkpoint"
model, encoder, config = load_checkpoint(MODEL_KIND, MODEL_DIR)

# Tạo ứng dụng web Flask
app = Flask(__name__)


def classify_message(message):
    """Phân loại một tin nhắn thành SPAM hoặc HAM."""

    # Chuẩn hóa giống dữ liệu train (lowercase, <URL>, <PHONE>, <MONEY>),
    # rồi chuyển thành token ID theo vocab lúc train
    features = transform_texts(
        MODEL_KIND,
        encoder,
        [normalize_text(message)],
        config["max_len"]
    )

    # Dự đoán xác suất bằng mô hình đã nạp
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(features)).item()

    # So sánh xác suất với ngưỡng phân loại đã lưu
    threshold = config.get("decision_threshold", 0.5)
    label = "SPAM" if probability >= threshold else "HAM"

    return label, probability


@app.route("/", methods=["GET", "POST"])
def home():
    """Hiển thị trang chủ và xử lý nội dung người dùng nhập."""

    message = ""
    result = None
    probability = None
    error = None

    if request.method == "POST":
        # Lấy nội dung tin nhắn từ biểu mẫu
        message = request.form.get("message", "").strip()

        if not message:
            error = "Vui lòng nhập nội dung tin nhắn."
        elif len(message) > 5000:
            error = "Tin nhắn quá dài. Vui lòng nhập tối đa 5.000 ký tự."
        else:
            # Chạy mô hình để phân loại tin nhắn
            result, probability = classify_message(message)

    return render_template(
        "index.html",
        message=message,
        result=result,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    # Chỉ chạy trên máy tính của bạn, không bật chế độ debug
    app.run(host="127.0.0.1", port=5000, debug=False)