from imutils.video import FPS
import cv2
import numpy as np
import dlib
import time

tracker = cv2.TrackerKCF_create()
cap = cv2.VideoCapture(0)
initBB = None
init_once = False
fps = None
face_detector = dlib.get_frontal_face_detector()
while True:
    ret, frame = cap.read()
    if ret:
        frame_height, frame_width, _ = frame.shape
        image = cv2.resize(frame, (224, 244))
        image_height, image_width, _ = image.shape
        if not init_once:
            faces = face_detector(image, 1)
            print("No of faces detected: "+str(len(faces)))
            if len(faces) == 1:
                for d in faces:
                    left = d.left()
                    top = d.top()
                    right = d.right()
                    bottom = d.bottom()
                    cal_left = left * frame_width / image_width
                    cal_top = top * frame_height / image_height
                    cal_right = right * frame_width / image_width
                    cal_bottom = bottom * frame_height / image_height
                    # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    initBB = tuple([cal_left, cal_top, cal_right, cal_bottom])
                    print("in init_once....")
                    success = tracker.init(frame, initBB)
                    fps = FPS().start()
                    init_once = True
        success, new_box = tracker.update(frame)
        print(success)
        if success:
            (x, y, w, h) = [int(v) for v in new_box]
            print(new_box)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()