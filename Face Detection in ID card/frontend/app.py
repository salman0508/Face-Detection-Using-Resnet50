from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

class FaceDetectionApp:
    def __init__(self):

        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = 'static/uploads'
        self.template_path = 'H:/Internship/Face Detection in ID card/Images/idCard.jpeg'
        self.detector = MTCNN()
        self.similarity_threshold = 0.60
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.output_faces_directory = 'static/uploads'
        self.uploaded_image1_url = None
        self.uploaded_image2_url = None
        self.similarity_result = 0.0
        self.eye_distance_result_path = None
        self.distance = None
        self._setup_routes()

    def _setup_routes(self):
        self.app.route('/', methods=['GET', 'POST'])(self.index)

    def run(self):
        self.app.run(debug=True)

    def index(self):
        confusion_matrix_data = None
        if request.method == 'POST':
            uploaded_image1 = request.files['uploaded_image1']
            uploaded_image2 = request.files['uploaded_image2']

            if uploaded_image1 and uploaded_image2:
                filename1 = secure_filename(uploaded_image1.filename)
                filename2 = secure_filename(uploaded_image2.filename)
                
                idCard_image_path, real_image_path = self._process_uploaded_images(uploaded_image1, uploaded_image2)
                output_path =   f'static/uploads/detected_{filename1}.jpg'
                self._detect_id_card_face(idCard_image_path,output_path)

                self._detect_real_image_face(real_image_path)
                
                id_card_face = self.extract_face(idCard_image_path)
                real_image_face = self.extract_face(real_image_path)
                
                if id_card_face is not None and real_image_face is not None:
                    self.similarity_result, _ = self._calculate_similarity(id_card_face, real_image_face)
                    print(self.similarity_result)
                    self.eye_distance_result_path, self.distance = self._calculate_eye_distance(real_image_path)
                           
                confusion_matrix_data = self.generate_confusion_matrix()
                
        return render_template('index.html', uploaded_image1_url=self.uploaded_image1_url,
                               uploaded_image2_url=self.uploaded_image2_url,
                               similarity_result=self.similarity_result,
                               eye_distance_result_path=self.eye_distance_result_path,
                               distance=self.distance, confusion_matrix_data = confusion_matrix_data)

    def _process_uploaded_images(self, uploaded_image1, uploaded_image2):
        filename1 = secure_filename(uploaded_image1.filename)
        filename2 = secure_filename(uploaded_image2.filename)

        image1_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename1)
        image2_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename2)

        uploaded_image1.save(image1_path)
        uploaded_image2.save(image2_path)

        self.uploaded_image1_url = f'static/uploads/{filename1}'
        self.uploaded_image2_url = f'static/uploads/{filename2}'
        return self.uploaded_image1_url, self.uploaded_image2_url

    def _detect_id_card_face(self,image_path,output_path):
        sift = cv2.SIFT_create()

        template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        kp1, des1 = sift.detectAndCompute(template, None)
        print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        kp2, des2 = sift.detectAndCompute(image, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            image = cv2.polylines(image, [np.int32(dst)], True, 255, 3)

            cv2.imwrite(output_path, image) 
            print(f"Detected ID Card saved at {output_path}")

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
            

            os.makedirs(self.output_faces_directory, exist_ok=True)
            for i, (x, y, w, h) in enumerate(faces):
                face = image[y:y+h, x:x+w]
                face_output_path = os.path.join(self.output_faces_directory, f"idCard_face_detected.jpg")
                cv2.imwrite(face_output_path, face)
                print(f"Face {i} saved at {face_output_path}")

    def _detect_real_image_face(self,image_path):
        image = cv2.imread(image_path)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=30, minSize=(30, 30))

        
        for i, (x, y, w, h) in enumerate(faces):
            face = gray_image[y:y+h, x:x+w]
            output_path = os.path.join(self.output_faces_directory, f"real_face_detected.jpg")
            cv2.imwrite(output_path, face)
            print(f"Face {i} saved at {self.output_faces_directory}")

    def _calculate_similarity(self,id_card_face, real_image_face):
        id_card_face_preprocessed = preprocess_input(np.expand_dims(id_card_face, axis=0))
        real_image_face_preprocessed = preprocess_input(np.expand_dims(real_image_face, axis=0))
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        model = Model(inputs=base_model.input, outputs=base_model.output)

        # Extract features
        id_card_features = model.predict(id_card_face_preprocessed)
        real_image_features = model.predict(real_image_face_preprocessed)

        similarity_score = cosine_similarity(id_card_features.reshape(1, -1), real_image_features.reshape(1, -1))[0][0]
        
        
        if similarity_score > self.similarity_threshold:
            print("Similarity Score: ", similarity_score)
            self.similarity_result = "The detected face on the ID card is similar to the detected face in the real image."
        else:
            print("Similarity Score: ", similarity_score)
            self.similarity_result = "The detected face on the ID card is not similar to the detected face in the real image."
        return self.similarity_result, similarity_score

    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        img = image.load_img(image_path, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def extract_face(self, image_path, required_size=(224, 224)):
        img = cv2.imread(image_path)
        faces = self.detector.detect_faces(img)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]['box']
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, required_size)
            return face
        else:
            return None

    def _calculate_eye_distance(self, image_path, required_size=(224, 224)):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 10)
        roi_gray = img
        distance_mm = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # Adjust parameters to reduce false positive eye detections
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(eyes) >= 2:
            eye1_center = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
            eye2_center = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
            pixel_distance = np.sqrt((eye1_center[0] - eye2_center[0])**2 + (eye1_center[1] - eye2_center[1])**2)
            distance_mm = pixel_distance * 0.264
            print(f"Estimated distance between eyes: {distance_mm:.2f} mm")

        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_gray, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
        result_path = os.path.join(self.output_faces_directory, 'eye_distance_result.jpg')
        cv2.imwrite(result_path, roi_gray)
        print(distance_mm)
        return result_path, distance_mm

    def generate_confusion_matrix(self):
        id_card_dir = 'H:/Internship/Face Detection in ID card/Images/testing/id'
        real_face_dir = 'H:/Internship/Face Detection in ID card/Images/testing/face'
        
        # Lists to store actual and predicted labels
        actual_labels = []
        predicted_labels = []
        
        # Iterate through both directories simultaneously
        for id_card_image, real_face_image in zip(os.listdir(id_card_dir), os.listdir(real_face_dir)):
            if id_card_image.endswith('.jpg') and real_face_image.endswith('.jpg'):
                id_card_path = os.path.join(id_card_dir, id_card_image)
                real_face_path = os.path.join(real_face_dir, real_face_image)
                
                id_card_face = self.extract_face(id_card_path)
                real_face = self.extract_face(real_face_path)
                _, similarity_score = self._calculate_similarity(id_card_face, real_face)
                
                predicted_label = 1 if similarity_score >= self.similarity_threshold else 0
                
                actual_label = 1 if id_card_path[-5] == real_face_path[-5] else 0
                
                actual_labels.append(actual_label)
                predicted_labels.append(predicted_label)
             
        for id_card_image in os.listdir(id_card_dir):
            if id_card_image.endswith('.jpg'):
                id_card_path = os.path.join(id_card_dir, id_card_image)
                
                # Choose a random face image from real_face_dir
                random_face_image = random.choice(os.listdir(real_face_dir))
                while not random_face_image.endswith('.jpg'):
                    random_face_image = random.choice(os.listdir(real_face_dir))
                random_face_path = os.path.join(real_face_dir, random_face_image)
                
                id_card_face = self.extract_face(id_card_path)
                random_face = self.extract_face(random_face_path)
                _, similarity_score = self._calculate_similarity(id_card_face, random_face)
                
                predicted_label = 1 if similarity_score >= self.similarity_threshold else 0
                
                actual_label = 1 if id_card_path[-5] == random_face_path[-5] else 0
                
                actual_labels.append(actual_label)
                predicted_labels.append(predicted_label)
                    
        confusion_mat = confusion_matrix(actual_labels, predicted_labels)
    
        confusion_matrix_data = {
            "tn": confusion_mat[0][0],
            "fp": confusion_mat[0][1],
            "fn": confusion_mat[1][0],
            "tp": confusion_mat[1][1]
        }
    
        return confusion_matrix_data     
            

if __name__ == '__main__':
    app = FaceDetectionApp()
    app.run()
    
