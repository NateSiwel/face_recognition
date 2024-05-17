import cv2
import numpy as np
import os
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
import pickle
import dlib
#usbipd attach --busid 2-6 --wsl

class Extract():
    def __init__(self):
        prototxt = 'models/deploy.prototxt.txt'    
        caffemodel='models/res10_300x300_ssd_iter_140000.caffemodel'     
        self.model =  cv2.dnn.readNetFromCaffe( prototxt, caffemodel)     

        self.embedder = cv2.dnn.readNetFromTorch('models/nn4.v2.t7')
        self.imagePaths = list(paths.list_images('faces_test'))

        PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

        try:
            training_data = pickle.load(open("training_data.p", "rb"))
            self.knownNames = training_data["names"]
            self.knownEmbeddings = training_data["embeddings"]
        except FileNotFoundError:
            self.knownNames, self.knownEmbeddings = self.pickle_data()

        self.le = LabelEncoder()

        self.names = self.le.fit_transform(self.knownNames)

        print("[INFO] Training model...")
        self.recognizer = LinearSVC()
        self.recognizer = SVC(gamma="scale")
        self.recognizer.fit(self.knownEmbeddings, self.names)

        print(self.le.inverse_transform(self.recognizer.classes_))

    def pickle_data(self):
        knownNames = []
        knownEmbeddings = []
        for (i, imagePath) in enumerate(self.imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1,
                len(self.imagePaths)))
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            (h, w) = image.shape[:2]

            face_positions = self.get_face_positions(image) 

            if len(face_positions) > 1:
                print(f"More than one face detected in training image @ location {imagePath}")
                continue 

            if len(face_positions) == 1:
                startX, startY, endX, endY = face_positions[0]
                face = image[startY:endY, startX:endX]
                vec = self.get_embedding(face)

                if name != "Nathan":
                    name = "Unknown"
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())

        pickle.dump({"names": knownNames, "embeddings": knownEmbeddings}, open("training_data.p", "wb"))

        return knownNames, knownEmbeddings

    def get_face_positions(self, image, draw=False):
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (300, 300))

        self.model.setInput(blob)
        detections = self.model.forward()
        #returns 4d array - [0][0][:] == detected objects filtered by confidence
        #detections[0][0][0-200] == [0, objectclass, confidence, topleftx, toplefty, brightx, brighty] 

        locations = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  
                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (startX, startY, endX, endY) = box.astype('int')
                locations.append([startX, startY, endX, endY])

        return locations

    def get_embedding(self, face):
        face = self.align_face(face) 
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)    
        resized_face = cv2.resize(face_rgb, (96, 96))
        faceBlob = cv2.dnn.blobFromImage(resized_face, 1.0/155,
                                         (96, 96), (0, 0, 0), swapRB=False, crop=False)
        self.embedder.setInput(faceBlob)
        vec = self.embedder.forward()

        return vec 

    def align_face(self, face):
        left_eye, right_eye = self.get_landmarks(face)
        #face_with_landmarks = face.copy()
        #cv2.imshow("landmarks", annotate_landmarks(face_with_landmarks, landmarks))
        left_eye = np.squeeze(np.asarray(left_eye))
        right_eye = np.squeeze(np.asarray(right_eye))

        #angle between eye corners
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        center = (face.shape[1] // 2, face.shape[0] // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_face = cv2.warpAffine(face, rotation_matrix, (face.shape[1], face.shape[0]))

        return rotated_face

    def get_landmarks(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        landmarks = self.predictor(gray, dlib.rectangle(0, 0, face.shape[1], face.shape[0]))
        landmarks_np = np.matrix([[p.x, p.y] for p in landmarks.parts()])

        #CORNERS = landmarks_np[[36] + [45]]
        return landmarks_np[36], landmarks_np[45] 

    def annotate_landmarks(self, im, landmarks):
        """
        RIGHT_EYE_POINTS = list(range(36, 42))
        LEFT_EYE_POINTS = list(range(42, 48))
        eyes = landmarks[LEFT_EYE_POINTS + RIGHT_EYE_POINTS] 
        """
        CORNERS = landmarks[[36] + [45]]
        for idx, point in enumerate(CORNERS):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(im, pos, 2, color=(0, 255, 255))
        return im

    def predict_face(self, vec):
        # Perform classification to recognize the face
        pred = self.recognizer.predict(vec)[0]
        name = self.le.inverse_transform(self.recognizer.classes_)[pred]

        return name


