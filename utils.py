import os
import cv2
import recognition


def reload():
    """
    Tải lại thư mục 'static', chỉ giữ lại ảnh mặc định
    """
    items = os.listdir('static')
    for i in items:
        if i != 'default_image.png':
            item_del = os.path.join('static', i)
            os.remove(item_del)


def face_recog(path_image):
    """
    Hàm nhận dạng gương mặt có trong ảnh đầu vào. Sử dụng giải thuật PCA
    path_image: đường dẫn ảnh cần nhận dạng
    """
    recog = recognition.recognition_actor(os.getcwd())
    recog.train_with_PCA()
    image, result = recog.predict_with_PCA(path_image)
    result_image = os.path.join('static', 'result_image.jpg')
    cv2.imwrite(result_image, image)


def recog_LBPH(path_image):
    """
    Hàm nhận dạng gương mặt có trong ảnh đầu vào. Sử dụng giải thuật LBPH
    path_image: đường dẫn ảnh cần nhận dạng
    """
    recog = recognition.recognition_actor(os.getcwd())
    recog.train_with_LBPH()
    image = recog.predict_with_LBPH(path_image)
    result_image = os.path.join('static', 'result_image.jpg')
    cv2.imwrite(result_image, image)