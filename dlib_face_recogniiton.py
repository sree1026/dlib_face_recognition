import sys
import os
import dlib
import glob
import cv2
import numpy as np
import pickle
import screeninfo

data = pickle.loads(open('encodings', 'rb').read())
train_image_encodings = data
# database_list = []

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


def L2_distance(face_encoding):
    min_distance = 0.52
    name = 'unknown'
    database_list = []
    print("Calculating Matches.........")
    for index, train_image_encoding in enumerate(train_image_encodings):
        # if encodings[index_value] != encoding:
        ref = train_image_encoding[1]
        distance = np.linalg.norm(face_encoding - ref)
        # if(distance < min_distance):
        #     min_distance = distance
        name = train_image_encoding[0]
        database_tuple = tuple([name, distance])
        database_list.append(database_tuple)
    return database_list


def show_details(database_list):
    img = cv2.imread('/home/soliton/Downloads/blank.png', 1)
    x = 0
    y = 0
    for index, item in enumerate(database_list):
        value = np.float(item[1])
        value = str(value)
        detail = item[0]+" : "+value[:8]
        h = y + 20
        cv2.rectangle(img, (x, y), (600, h), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, detail, (x, y+15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('imd_details', img)
        cv2.waitKey(500)
        y = h + 20
    cv2.destroyAllWindows()


def face_recogniser():
    database_list = []
    screen_detail = screeninfo.get_monitors()[0]
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        (frame_height, frame_width, channels) = frame.shape
        if ret:
            img = cv2.GaussianBlur(frame, (5, 5), 0)
            img = cv2.resize(img, (224, 224))
            (img_height, img_width, img_channels) = img.shape
            # img = frame
            faces = detector(img, 1)
            print("Number of faces detected: {}".format(len(faces)))
            if (len(faces) != 0):
                for face, d in enumerate(faces):
                    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(face, d.left(), d.top(), d.right(), d.bottom()))
                    x = d.left()
                    y = d.top()
                    w = d.right()
                    z = d.bottom()
                    x1 = int(x*frame_width / img_width)
                    y1 = int(y*frame_height / img_height)
                    w1 = int(w*frame_width / img_width)
                    z1 = int(z*frame_height / img_height)
                    # Get the landmarks/parts for the face in box d.
                    shape = sp(img, d)
                    face_descriptor = list(facerec.compute_face_descriptor(img, shape))
                    face_encoding = [np.array(face_descriptor)]
                    # call L2_distance function to recognise face.
                    # name = L2_distance(face_encoding)
                    database_list = L2_distance(face_encoding)
                    # sorting the database
                    database_list.sort(key=lambda x:x[1])
                    if database_list[0][0] != 'unknown':
                        cv2.rectangle(frame, (x1, y1), (w1, z1), (0, 255, 0), 2)
                        cv2.rectangle(frame, (x1, y1-30), (w1, y1), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, database_list[0][0], (x1+6, y1-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.rectangle(frame, (x1, y1), (w1, z1), (0, 0, 255), 2)
                        cv2.rectangle(frame, (x1, y1 - 30), (w1, y1), (0, 0, 255), cv2.FILLED)
                        cv2.putText(frame, 'unknown', (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            window_name = 'Find your face :p '
            # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            # cv2.moveWindow(window_name, screen_detail.x - 1, screen_detail.y - 1)
            # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, frame)
            # if len(faces) != 0:
            #     show_details(database_list)

        if cv2.waitKey(50) == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

# def L2_distance(face_encoding):
#     min_distace = 0.45
#     database = []
#     # name = 'unknown'
#     print("Calculating Matches.........")
#     for index, train_image_encoding in enumerate(train_image_encodings):
#         # if encodings[index_value] != encoding:
#         ref = train_image_encoding[1]
#         distance = np.linalg.norm(face_encoding - ref)
#         if(distance < min_distace):
#             min_distace = distance
#             name = train_image_encoding[0]
#             database = [min_distace, name]
#     return database


if __name__ == '__main__':
    face_recogniser()
