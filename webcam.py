#!/usr/bin/env python

import cv2
from time import sleep
from skimage import transform
import numpy as np

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)


def normalize(img, size=200):
    return transform.resize(img, [size, size], mode='constant')


def MSE(img, img2):
    img2 = normalize(img2).reshape([-1, 1])
    img = normalize(img).reshape([-1, 1])
    return np.sum(np.power(img - img2, 2))
    

def read():
    my_face = cv2.imread('face.png', cv2.IMREAD_GRAYSCALE)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display frame
        cv2.imshow('Video', frame)
        
        # Detect face
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw a rectangle around the faces
        if len(faces):
            (x, y, w, h) = faces[0]
            face = frame[y:y+h,x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face2 = face.copy()
            face = normalize(face)
            error = 10 ** 10
            if my_face is not None:
                error = MSE(my_face, face)
            color = (0, 0, 255)
            if error < 1000:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.imshow('Video', frame)
            print(f'w: {w:<5}h: {h:<5}{error}')
            # cv2.imshow('Face', face)
        else:
            cv2.destroyWindow('Face')

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite('face.png', face2)
            return True

        elif key == ord('q'):
            return False

def main():
    r = True
    while r:
        r = read()

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
