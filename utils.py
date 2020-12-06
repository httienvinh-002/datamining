import os
import cv2
import recognition
from os import path, walk

def reload():
    items = os.listdir('static')
    for i in items:
        if i != 'default_image.jpg':
            item_del = os.path.join('static', i)
            os.remove(item_del)


def face_recog(path_image):
    recog = recognition.recognition_actor(os.getcwd())
    recog.train_with_PCA()
    image, result = recog.predict_with_PCA(path_image)
    result_image = os.path.join('static', 'result_image.jpg')
    cv2.imwrite(result_image, image)


def recog_LBPH(path_image):
    recog = recognition.recognition_actor(os.getcwd())
    recog.train_with_LBPH()
    image = recog.predict_with_LBPH(path_image)
    result_image = os.path.join('static', 'result_image.jpg')
    cv2.imwrite(result_image, image)