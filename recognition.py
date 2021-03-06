import google_parser

import os
import re
import cv2
import numpy as np
import pandas as pd

from mtcnn.mtcnn import MTCNN
from icrawler.builtin import GoogleImageCrawler

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score




def convert(text):
    """
    Convert from 'Tieng Viet co dau' thanh 'Tieng Viet khong dau'
    text: input string to be converted
    Return: string converted
    """
    patterns = {
        '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
        '[đ]': 'd',
        '[èéẻẽẹêềếểễệ]': 'e',
        '[ìíỉĩị]': 'i',
        '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
        '[ùúủũụưừứửữự]': 'u',
        '[ỳýỷỹỵ]': 'y'
    }

    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        output = re.sub(regex.upper(), replace.upper(), output)
        output = re.sub(' ', '_', output)
    return output


class recognition_actor(object):
    def __init__(self, image_archive=os.getcwd()):
        self.name = []
        self.id = []
        self.image_archive = image_archive
        self.image_folder = self.image_archive + '/images'
        self.face_folder = self.image_archive + '/faces'

        if os.path.exists(self.image_folder) == False:
            os.mkdir(self.image_folder)
        if os.path.exists(self.face_folder) == False:
            os.mkdir(self.face_folder)
        if len(os.listdir(self.image_folder)) > 0:
            for actor in os.listdir(self.image_folder):
                self.name.append(actor)
            self.name.sort()
        return


    def image_crawl(self, actor, num):
        """
        Crawl hình ảnh của 1 diễn viên từ Google Image và lưu vào thư mục ứng với tên của diễn viên đó
        actor: tên diễn viên
        num: số lượng ảnh
        """
        if convert(actor) in self.name:
            return
        self.name.append(actor)
        self.name.sort()
        path = self.image_folder + "/" + convert(actor)
        google_crawler = GoogleImageCrawler(parser_cls=google_parser.GoogleParser, storage={'root_dir': path})
        google_crawler.crawl(keyword=actor, max_num=num)


    def get_image_path(self, actor, image_theme):
        """
        Truy suất hình ảnh của 1 diễn viên trong thư mục 'images' hoặc 'faces'
        actor: tên diễn viên
        image_theme: images(ảnh ban đầu) hoặc faces(ảnh gương mặt)
        return: đường dẫn của ảnh
        """
        paths = []
        if image_theme == 'images':
            theme = self.image_folder
        if image_theme == 'faces':
            theme = self.face_folder
        if actor not in os.listdir(theme):
            return paths

        root = theme + "/" + actor
        for image_name in os.listdir(root):
            image_path = root + "/" + image_name
            image = cv2.imread(image_path)
            if image is not None:
                paths.append(image_path)
        return paths


    def face_extraction(self, image_path):
        """
        Phát hiện vị trí và trích xuất gương mặt trong ảnh
        image_path: đường dẫn của ảnh ban đầu
        return: ảnh, ảnh gương mặt, tọa độ gương mặt trong ảnh
        """
        detector = MTCNN()
        image = cv2.imread(image_path)
        result = detector.detect_faces(image) # Trích xuất gương mặt có trong ảnh
        faces_image = []
        dict_image = {}

        if len(result) == 0:
            return dict_image

        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            height, width = image.shape[0:2]
            face_x, face_y = bounding_box[0], bounding_box[1]
            face_w, face_h = bounding_box[2], bounding_box[3]
            if face_x < 0:
                face_w = face_w + face_x
                face_x = 0
            if face_y < 0:
                face_h = face_h + face_y
                face_y = 0
            if face_w > width:
                face_w = width - face_x
            if face_h > height:
                face_h = height - face_y

            face = image[face_y:face_y + face_h + 1, face_x:face_x + face_w + 1]
            face_tranfer = cv2.resize(face, (160, 160))

            face = {
                'x': face_x,
                'y': face_y,
                'width': face_w,
                'height': face_h,
                'face': face_tranfer
            }
            faces_image.append(face)

        dict_image = {
            'image': image,
            'faces_image': faces_image
        }
        return dict_image


    def save_face(self, image_path):
        """
        Lưu ảnh gương mặt đã trích xuất vào thư mục 'faces'
        image_path: đường dẫn của ảnh ban đầu
        """
        dict_image = self.face_extraction(image_path)
        if len(dict_image) == 0:
            return

        faces = dict_image['faces_image']
        if len(faces) != 1:
            return None

        face = faces[0]['face']
        name = image_path[image_path.find("images") + 7:-11]
        for n, na in enumerate(self.name):
            if name == na:
                self.id.append(n)

        # Nếu diễn viên chưa có ảnh gương mặt, tạo thư mục chứa ảnh gương mặt của diễn viên đó
        # (../faces/tendienvien)
        face_path = image_path.replace("images", "faces")
        if os.path.exists(face_path[:-11]) == False:
            os.mkdir(face_path[:-11])
        cv2.imwrite(face_path, face)
        return


    def train_with_LBPH(self):
        """
        Huấn luyện mô hình với tập dữ liệu ảnh gương mặt
        """
        faces = []
        self.id = []
        for idx, actor in enumerate(self.name):
            faces_path = self.get_image_path(actor, 'faces')
            for path in faces_path:
                face = cv2.imread(path)
                faces.append(face)
                self.id.append(idx)

        gray_faces = []
        for face in faces:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray_faces.append(gray)

        # Train model để trích xuất đặc trưng các khuôn mặt
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(gray_faces, np.array(self.id))

        # Lưu model
        model_path = self.image_archive + '/' + 'LBPH'
        if os.path.exists(model_path) == False:
            os.mkdir(model_path)
        recognizer.save(model_path + '/trainner.yml')
        return


    def predict_with_LBPH(self, image_path):
        """
        Nhận diện khuôn mặt có trong ảnh
        image_path: ảnh chứa khuôn mặt cần nhận dạng
        image: ảnh chứa khuôn mặt đã được nhận dạng
        """
        dict_image = self.face_extraction(image_path)
        if len(dict_image) == 0:
            return
        image = dict_image['image']
        faces = dict_image['faces_image']

        for face in faces:
            face_x, face_y = face['x'], face['y']
            face_w, face_h = face['width'], face['height']
            face_image = face['face']
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Sử dụng mô hình đã lưu để nhận dạng gương mặt có trong ảnh
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(self.image_archive+'/LBPH/trainner.yml')

            Id, dist = recognizer.predict(face_gray)
            name = convert(self.name[Id])

            # Vẽ vùng chứa gương mặt
            cv2.rectangle(image,
                          (face_x, face_y),
                          (face_x + face_w, face_y + face_h), (0, 255, 25), 2)

            m = (face_x + face_w) * (face_y + face_h) * 3 / 1362060
            if m < 0.5:
                m = 0.5
            cv2.rectangle(image, (face_x, face_y + face_h + int(m * 25)),
                          (face_x + face_w, face_y + face_h),
                          (0, 0, 255), cv2.FILLED)

            # Vẽ tên diện viên đã được nhận dạng
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img=image,
                        text=name,
                        org=(face_x, face_y + face_h + int(m * 20)),
                        fontFace=font,
                        fontScale=m, color=(255, 255, 255), lineType=3)

            cv2.rectangle(image, (face_x, face_y),
                          (face_x + face_w, face_y - int(m * 25)),
                          (0, 0, 255), cv2.FILLED)

            # Vẽ độ sai khác so với trung bình của tập ảnh gương mặt huấn luyện có cùng nhãn với kết quả dự đoán
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img=image,
                        text='%.2f' % dist,
                        org=(face_x, face_y - 0),
                        fontFace=font,
                        fontScale=m, color=(255, 255, 255), lineType=3)
        return image

    def classification_model(self):  # , face_transform_pca):
        """
        Lựa chọn mô hình phân lớp phù hợp trong các mô hình sau:
        'K-NeighborsNear', 'SVM_SVC', 'LogisticRegression'
        """
        scores = []
        models = []
        names = ['K-NeighborsNear', 'SVM_SVC', 'LogisticRegression']

        # K-Neightbors
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score

        k_select = 0
        max_score = 0
        for k in range(1, len(self.train_face)):
            knn = KNeighborsClassifier(n_neighbors=k)
            try:
                score = cross_val_score(knn, self.train_face, self.train_label, cv=5)
            except:
                break
            if np.median(score) > max_score:
                k_select = k
                max_score = np.median(score)

        knn_select = KNeighborsClassifier(n_neighbors=k_select)
        models.append(knn_select)
        scores.append(np.median(max_score))

        # SVM
        from sklearn import svm
        svm_svc = svm.SVC(kernel='linear')
        score = cross_val_score(svm_svc, self.train_face, self.train_label, cv=5)
        models.append(svm_svc)
        scores.append(np.median(score))

        # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        logisticRegr = LogisticRegression(solver='lbfgs')
        score = cross_val_score(logisticRegr, self.train_face, self.train_label, cv=5)
        models.append(logisticRegr)
        scores.append(np.median(score))

        max = 0
        name = ''
        model = None
        for n, s in enumerate(scores):
            if s >= max:
                max = s
                model = models[n]
                name = names[n]
        self.train_model = model
        self.model_name = name
        self.model_score = max
        #return

    def train_with_PCA(self):
        """
        Sử dụng giải thuật PCA giảm chiều dữ liệu tập của dữ liệu và huấn luyện mô hình phân lớp
        """
        paths = []
        id = []
        for name in self.name:
            path = self.get_image_path(name, 'faces')
            for p in path:
                id.append(self.name.index(name))
                paths.append(p)

        # Chuyển ảnh có kích thước 160x160 sang 1x160*160
        img_v1 = []
        for path in paths:
            im = cv2.imread(path)
            imfg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_re = imfg.reshape(1, -1)
            img_v1.append(im_re[0])

        df_face = pd.DataFrame(img_v1)
        train_face = df_face.values
        self.train_label = pd.DataFrame(id).values

        # Tính trung bình của tập ảnh huấn luyện
        self.mean = np.mean(train_face, axis=0)
        # Tính giá trị theta của tập ảnh huấn luyện
        train_face = train_face - self.mean

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        # Fit on training set only.
        self.scaler.fit(train_face)
        # Apply transform to both the training set and the test set.
        train_face = self.scaler.transform(train_face)

        # Sử dụng tập dữ liệu huấn luyện để tạo ra không gian mới có số chiều nhỏ hơn
        self.pca = PCA(n_components=0.99, svd_solver='full')
        self.pca.fit(train_face)
        # Chiếu dữ liệu huấn luyện vào không gian mới
        self.train_face = self.pca.transform(train_face)
        # Lựa chọn mô hình huấn luyện
        self.classification_model()
        #return

    def predict_with_PCA(self, image_path):
        """
        Nhận dạng ảnh với giải thuật PCA
        image_path: đường dẫn của ảnh chứa khuôn mặt cần nhận dạng
        :return: image, results
        image: ảnh chứa khuôn mặt đã được nhận dạng
        results: thông tin kết quả nhận dạng
        """
        dict_image = self.face_extraction(image_path)
        if len(dict_image) == 0:
            image = cv2.imread(image_path)
            # cv2_imshow(image)
            print('Không tìm thấy gương mặt')
            return image, None, None
        image = dict_image['image']
        faces = dict_image['faces_image']
        # names = []
        results = []

        for face in faces:
            result = []
            face_x, face_y = face['x'], face['y']
            face_w, face_h = face['width'], face['height']
            face_image = face['face']

            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Chuyển ảnh sang kênh xám
            face_gray = face_gray.reshape(1, -1)
            # Tính giá trị theta của ảnh cần nhận dạng
            face_gray = face_gray - self.mean
            face_gray = self.scaler.transform(face_gray)
            # Chiếu vào không gian mới
            test_face = self.pca.transform(face_gray)

            self.train_model.fit(self.train_face, self.train_label)
            id_predict = self.train_model.predict(test_face)
            model_name = self.model_name
            cross_score = self.model_score
            name = convert(self.name[id_predict[0]])

            result.append(cross_score)
            result.append(model_name)
            result.append(name)
            results.append(result)
            print("ID: ", id_predict[0], " score: ", cross_score,
                  "Train model: ", model_name)

            # Vẽ vùng giới hạn gương mặt
            cv2.rectangle(image,
                          (face_x, face_y),
                          (face_x + face_w, face_y + face_h), (0, 255, 25), 2)

            m = (face_x + face_w) * (face_y + face_h) * 3 / 1362060
            if m < 0.5:
                m = 0.5
            cv2.rectangle(image, (face_x, face_y + face_h + int(m * 25)),
                          (face_x + face_w, face_y + face_h),
                          (0, 0, 255), cv2.FILLED)

            # Vẽ tên diễn viên đã được nhận dạng
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img=image,
                        text=name,
                        org=(face_x, face_y + face_h + int(m * 20)),
                        fontFace=font,
                        fontScale=m, color=(255, 255, 255), lineType=3)

            cv2.rectangle(image, (face_x, face_y),
                          (face_x + face_w, face_y - int(m * 25)),
                          (0, 0, 255), cv2.FILLED)

            # Vẽ độ chính xác của mô hình nhận dạng
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img=image,
                        text='%.2f' % cross_score,
                        org=(face_x, face_y - 0),
                        fontFace=font,
                        fontScale=m, color=(255, 255, 255), lineType=3)
        return image, results