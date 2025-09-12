# pothole_segmentation_alert.py
import cv2
import glob
import os
import numpy as np
import time
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound

# ==============================
# 1. Tạo file âm thanh cảnh báo nếu chưa có
# ==============================
if not os.path.exists("canhbao.mp3"):
    tts = gTTS("Cảnh báo! Phía trước có ổ gà, hãy giảm tốc độ!", lang="vi")
    tts.save("canhbao.mp3")
    print("✅ Đã tạo file canhbao.mp3")

# ==============================
# 2. Tìm file YOLO SEG best.pt mới nhất
# ==============================
weight_paths = glob.glob("runs/segment/**/weights/best.pt", recursive=True)
if not weight_paths:
    raise FileNotFoundError("⚠️ Không tìm thấy file best.pt trong runs/segment/")

latest_weight = max(weight_paths, key=os.path.getmtime)
print(f"✅ Đang dùng mô hình Segmentation: {latest_weight}")

# Load YOLO segmentation
model = YOLO(latest_weight)

# ==============================
# 3. Mở video
# ==============================
video_path = "test2.mp4"   # đổi đường dẫn nếu cần
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"⚠️ Không thể mở video {video_path}")

last_alert_time = 0

# Màu cho từng class (BGR)
CLASS_COLORS = {
    "pothole": (0, 255, 0),   # Ổ gà -> Xanh lá
    "water": (255, 0, 0)      # Nước -> Xanh dương
}

# ==============================
# 4. Chạy detect + segmentation + cảnh báo
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detected = False

    for result in results:
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for mask, cls_id in zip(masks, cls_ids):
                detected = True
                mask = mask.astype(np.uint8) * 255
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Lấy tên class
                class_name = model.names[cls_id]

                # Chọn màu (mặc định đỏ nếu class chưa có trong dict)
                color = CLASS_COLORS.get(class_name, (0, 0, 255))

                # Tạo mask màu
                color_mask = np.zeros_like(frame)
                for i in range(3):
                    color_mask[:, :, i] = mask_resized * (color[i] / 255.0)

                # Kết hợp mask với ảnh gốc
                frame = cv2.addWeighted(frame, 1, color_mask.astype(np.uint8), 0.5, 0)

    # Nếu phát hiện ổ gà thì cảnh báo
    if detected and (time.time() - last_alert_time > 5):
        print("⚠️ Cảnh báo: Phía trước có ổ gà!")
        cv2.putText(frame, "CANH BAO: O GA PHIA TRUOC!",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        playsound("canhbao.mp3", block=False)  # phát giọng nói song song
        last_alert_time = time.time()

    # Hiển thị kết quả
    cv2.imshow("Pothole Segmentation + Alert", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
