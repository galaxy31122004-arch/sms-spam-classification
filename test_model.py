import torch
from pathlib import Path

# Nhập các hàm dùng chung để nạp mô hình và chuyển tin nhắn thành dữ liệu số
from data.prepare_data import normalize_text
from models.common import load_checkpoint, transform_texts

# Đường dẫn đến thư mục chứa mô hình MLP đã huấn luyện trên Google Colab
MODEL_KIND = "transformer"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "transformer" / "demo_checkpoint"

# Giới hạn số luồng CPU để phù hợp với máy tính RAM 8 GB
torch.set_num_threads(4)

# Nạp trọng số mô hình, bộ chuyển đổi TF-IDF và cấu hình đã lưu
model, encoder, config = load_checkpoint(MODEL_KIND, MODEL_DIR)

# Hai tin nhắn mẫu để kiểm tra mô hình
messages = [
    "Congratulations! You have won a free prize. Call now!",
    "Hey, are we still meeting for lunch today?"
]

# Chuẩn hóa giống dữ liệu train, rồi chuyển thành vector TF-IDF
features = transform_texts(MODEL_KIND, encoder, [normalize_text(m) for m in messages], config["max_len"])

# Đưa mô hình về chế độ dự đoán và tắt tính toán gradient
model.eval()
with torch.inference_mode():
    # Mô hình trả về logits; sigmoid chuyển chúng thành xác suất spam
    probabilities = torch.sigmoid(model(features)).tolist()

# In kết quả phân loại của từng tin nhắn
threshold = config.get("decision_threshold", 0.5)

for message, probability in zip(messages, probabilities):
    label = "SPAM" if probability >= threshold else "HAM"

    print("-" * 60)
    print("Tin nhắn:", message)
    print(f"Xác suất SPAM: {probability:.2%}")
    print("Kết luận:", label)

print("-" * 60)
print(f"Đã nạp mô hình {MODEL_KIND} thành công.")