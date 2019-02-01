import cv2
import numpy as np
import os
import sys
import dlib
import glob

faces_folder_path = sys.argv[1]
shape_predictor_file = 'shape_predictor_5_face_landmarks.dat'
face_recog_model_file = 'dlib_face_recognition_resnet_model_v1.dat'

face_detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor_file)
face_recog_model = dlib.face_recognition_model_v1(face_recog_model_file)
win = dlib.image_window()

def calculate_L2_distance(data):
    index = int(input("Enter a index value from 0 to 60: "))
    ref = data[index][1]
    print(len(data))
    for data_item in data:
        distance = np.linalg.norm(data_item[1] - ref)
        print(data_item[0]+" : "+str(distance))


def calculate_encodings():
    print("Preparing database............")
    data = []
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        (img_folder, ext) = os.path.splitext(f)
        (_, img_name) = os.path.split(img_folder)
        print(img_name)
        img = cv2.resize(img, (224, 224))
        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = face_detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        if (len(dets) != 0):
            # Now process each face we found.
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(),
                                                                                   d.bottom()))
                # Get the landmarks/parts for the face in box d.
                shape = sp(img, d)
                # Draw the face landmarks on the screen so we can see what face is currently being processed.
                win.clear_overlay()
                win.add_overlay(d)
                win.add_overlay(shape)
                face_descriptor = list(face_recog_model.compute_face_descriptor(img, shape))
                # print(face_descriptor)
                data.append([img_name, np.array(face_descriptor)])
    print("Database finished.....")
    data.sort(key=lambda x:x[0])
    return data


if __name__ == '__main__':
    data = calculate_encodings()
    calculate_L2_distance(data)
