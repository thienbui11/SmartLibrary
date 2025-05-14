import cv2
from paddleocr import PaddleOCR
image_file = '3.jpg'
img = cv2.imread(image_file)

    
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)
    
def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)
    
def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

inverted_image = cv2.bitwise_not(img)

# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)

no_noise = noise_removal(inverted_image)

eroded_image = thin_font(no_noise)

dilated_image = thick_font(no_noise)


# ocr = PaddleOCR(lang="en")
# ocr = PaddleOCR(lang="en", det_db_box_thresh=0.2, use_angle_cls=True, rec_algorithm="SVTR_LCNet")
ocr = PaddleOCR(
    lang="vi",  
    det_db_box_thresh=0.3,  # Giảm ngưỡng phát hiện chữ để bắt nhiều chữ nhỏ hơn (mặc định 0.5)
    det_db_unclip_ratio=2.0,  # Mở rộng vùng chữ để tránh cắt mất ký tự (mặc định 1.5)
    use_angle_cls=True,  # Tự động xoay chữ nếu bị nghiêng
    rec_algorithm="SVTR_LCNet",  # Mô hình nhận diện chữ mạnh hơn
    rec_image_shape="3, 32, 320",  # Điều chỉnh kích thước ảnh đầu vào để nhận diện chữ nhỏ tốt hơn
    rec_batch_num=6,  # Xử lý nhiều dòng văn bản cùng lúc
    max_text_length=50,  # Tăng độ dài tối đa của một dòng chữ để tránh cắt mất thông tin
)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Làm rõ chữ
    return thresh

# Tiền xử lý ảnh trước khi OCR
processed_image = preprocess_image(dilated_image)
results = ocr.ocr(processed_image, cls=True)
# results = ocr.ocr(dilated_image, cls=True)
ocr_text = "\n".join([word_info[1][0] for line in results for word_info in line])