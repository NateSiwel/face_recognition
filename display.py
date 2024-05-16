import cv2
from numpy.linalg import svd, norm
from cv2 import dnn
import numpy as np
import os
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import dlib

#usbipd attach --busid 2-6 --wsl

video = cv2.VideoCapture(0)
cv2.namedWindow("Scanner")

video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

prototxt = 'models/deploy.prototxt.txt'    
caffemodel='models/res10_300x300_ssd_iter_140000.caffemodel'     
model =  cv2.dnn.readNetFromCaffe( prototxt, caffemodel)     

embedder = cv2.dnn.readNetFromTorch('models/nn4.small2.v1.t7')
imagePaths = list(paths.list_images('faces_test'))
knownNames=[]
knownEmbeddings=[]

PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, dlib.rectangle(0, 0, face.shape[1], face.shape[0]))
    landmarks_np = np.matrix([[p.x, p.y] for p in landmarks.parts()])

    #CORNERS = landmarks_np[[36] + [45]]
    return landmarks_np[36], landmarks_np[45] 

def annotate_landmarks(im, landmarks):
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

def get_face_positions(image, draw=False):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (300, 300))

    model.setInput(blob)
    detections = model.forward()
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

def getEmbedding(face):
    left_eye, right_eye = get_landmarks(face)
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
    #cv2.imshow("rotface", rotated_face)
    
    faceBlob = cv2.dnn.blobFromImage(rotated_face, 1.0 / 255,
                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    
    return vec

for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
        len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]

    face_positions = get_face_positions(image) 

    if len(face_positions) > 1:
        print(f"More than one face detected in training image @ location {imagePath}")
        continue 

    if len(face_positions) == 1:
        startX, startY, endX, endY = face_positions[0]
        face = image[startY:endY, startX:endX]
        vec = getEmbedding(face)

        if name != "Nathan":
            name = "Unknown"
        knownNames.append(name)
        knownEmbeddings.append(vec.flatten())

le = LabelEncoder()

names = le.fit_transform(knownNames)

print("[INFO] Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(knownEmbeddings, names)

print(le.inverse_transform(recognizer.classes_))

while True:
    ret, frame = video.read()
    
    if not ret:
        break

    face_positions = get_face_positions(frame, draw=True)

    for (i, face) in enumerate(face_positions):

        startX, startY, endX, endY = face_positions[i]
        face = frame[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        if fW < 20 or fH < 20:
            continue

        vec = getEmbedding(face)  

        #perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        name = le.inverse_transform(recognizer.classes_)[j]

        text = "{}: {:.2f}%".format(name, proba * 100)
        print(text)

    cv2.imshow("Scanner", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

video.release()
cv2.destroyAllWindows
