import sys
import os
import dlib
import glob
import cv2
import numpy as np
import pickle

encodings = pickle.loads(open('encodings', 'rb').read())
for encoding in encodings:
    print(encoding[1])

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def face_recogniser():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                img = cv2.resize(frame, (224, 224))
                faces = detector(img, 1)
                print("Number of faces detected: {}".format(len(faces)))
                if (len(faces) != 0):
                    for face, d in enumerate(faces):
                        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(face, d.left(), d.top(), d.right(), d.bottom()))
                        # Get the landmarks/parts for the face in box d.
                        shape = sp(img, d)
                        face_descriptor = list(facerec.compute_face_descriptor(img, shape))
                        face_encoding = [np.array(face_descriptor)]
                        L2_distance(face_encoding)
            if cv2.waitKey(50) == ord('q'):
                break

def L2_distance(face_encoding):
    min_distace = 0.52
    print("Calculating Matches.........")
    for encoding in enumerate(encodings):
        # if encodings[index_value] != encoding:
        ref = encoding[1]
        distance = np.linalg.norm(face_encoding - ref)
        if(distance < min_distace):
            min_distace = distance
            name = encoding[0]
    print(name)



if __name__ == '__main__':
    face_recogniser()
