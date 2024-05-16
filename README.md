# Features
* Detects faces in real time using pre-trained caffe model
* Extracts face embeddings using pre-trained pytorch model
* Trains an SVM classifier based on a dataset of known faces
* Automatically aligns face if tilted 
* Recognizes faces in real time from webcam or other video, displaying binary classification 

# Dependencies
* Python 3.x
* OpenCV
* NumPy
* imutils
* scikit-learn

# TODO 
* Replace embedder w/ improved model
* Replace face detetector w/ improved dlib?
* refactor
