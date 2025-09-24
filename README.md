🚗 PHÁT HIỆN Ổ GÀ TRÊN MẶT ĐƯỜNG VÀ CẢNH BÁO ÂM THANH SỬ DỤNG YOLOv8

1. Giới thiệu
Tai nạn giao thông do các yếu tố hạ tầng, đặc biệt là **ổ gà trên mặt đường**, đang là vấn đề nhức nhối hiện nay. Các phương tiện giao thông, đặc biệt là xe máy và ô tô, có thể gặp nguy hiểm khi đi qua những đoạn đường có ổ gà mà người lái không kịp xử lý.  

Vì vậy, nghiên cứu này tập trung xây dựng một hệ thống **tự động phát hiện ổ gà bằng camera** và đưa ra **tín hiệu cảnh báo bằng âm thanh** nhằm hỗ trợ người tham gia giao thông phòng tránh rủi ro.

2. Mục tiêu nghiên cứu
- Xây dựng hệ thống có khả năng **nhận diện ổ gà trên mặt đường** theo thời gian thực.  
- Phát ra cảnh báo âm thanh bằng tiếng Việt: *“Phía trước có ổ gà”* khi phát hiện.  
- Tích hợp mô hình **YOLOv8** để đạt độ chính xác cao và tốc độ xử lý phù hợp cho ứng dụng thực tế.  

---

## 3. Phương pháp nghiên cứu
### 3.1. Dữ liệu
- Dữ liệu hình ảnh/ video về ổ gà được thu thập từ nhiều môi trường khác nhau.  
- Dữ liệu được gán nhãn bằng công cụ **LabelImg** trước khi đưa vào huấn luyện.  

### 3.2. Mô hình học sâu
- Sử dụng **YOLOv8 (You Only Look Once phiên bản 8)** của Ultralytics.  
- Mô hình được huấn luyện để phát hiện lớp đối tượng **"pothole" (ổ gà)**.  

### 3.3. Quy trình xử lý
- **Đầu vào:** Camera (webcam hoặc iPhone kết nối).  
- **Xử lý:** Mỗi khung hình được đưa qua YOLOv8 để nhận diện ổ gà.  
- **Đầu ra:**  
  - Hiển thị khung đỏ quanh ổ gà trong video.  
  - Phát âm thanh cảnh báo tiếng Việt.  

### 3.4. Cảnh báo âm thanh
- Sử dụng thư viện **gTTS (Google Text-to-Speech)** để tạo file âm thanh cảnh báo.  
- Khi phát hiện ổ gà, hệ thống sẽ tự động phát giọng nói:  
  👉 *“Phía trước có ổ gà”*.  

---

## 4. Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.10+  
- **Mô hình AI:** YOLOv8 – Ultralytics  
- **Thư viện chính:**  
  - `opencv-python` – xử lý video  
  - `ultralytics` – YOLOv8  
  - `gtts` – tạo giọng nói tiếng Việt  
  - `playsound` – phát âm thanh  

---

## 5. Kiến trúc hệ thống
```text
Camera (điện thoại/iPhone) 
        ↓
Mô hình YOLOv8 → Xử lý khung hình → Phát hiện ổ gà
        ↓
Cảnh báo âm thanh: "Phía trước có ổ gà"
6. Kết quả thực nghiệm

Mô hình YOLOv8 huấn luyện trên tập dữ liệu ổ gà cho độ chính xác tương đối cao trong điều kiện ánh sáng ban ngày.

Hệ thống có khả năng phát hiện và phát cảnh báo trong ~1 giây, đáp ứng yêu cầu thời gian thực.

Kết quả minh chứng khả năng ứng dụng trong hỗ trợ người điều khiển phương tiện.

7. Kết luận và hướng phát triển

Kết luận:

Hệ thống đã chứng minh khả năng ứng dụng YOLOv8 vào bài toán phát hiện ổ gà.

Tín hiệu cảnh báo bằng âm thanh giúp nâng cao độ an toàn giao thông.

Hướng phát triển:

Mở rộng dữ liệu huấn luyện cho nhiều điều kiện môi trường (ban đêm, trời mưa, nhiều phương tiện).

Tích hợp trực tiếp vào ứng dụng di động để sử dụng camera smartphone.

Kết hợp GPS để ghi lại và chia sẻ vị trí ổ gà trên bản đồ giao thông.

8. Thông tin tác giả

Tên: Trịnh Minh Quân

Sinh viên năm 3 – Ngành Công nghệ Thông tin

Trường: Đại học Đại Nam

Chuyên ngành: Hệ thống Thông tin
