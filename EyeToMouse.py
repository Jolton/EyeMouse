


import cv2
import numpy as np
from pymouse import PyMouse
import time

def main():
    webCam = cv2.VideoCapture(0)

    global mouse
    mouse = PyMouse()
    mouse.move(1,1)
    
    while (True):
        ret, img = webCam.read()

        cv2.namedWindow('img')
        cv2.moveWindow('img',0,0)

        #cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

        eyePos = detectEyes(img, face_cascade, eye_cascade)
        

        if eyePos is not None:
            print("Move Mouse!!!")
            print("x=",eyePos[0],"y=",eyePos[1])
            mouse.move(eyePos[0],eyePos[1])
        
        cv2.imshow('img',img)
    
    webCam.release()
    cv2.destroyAllWindows()

def detectEyes(img, faceCascade, eyeCascade):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    except Exception as e:
        print('img not loaded')
        cv2.imshow('img',img)
    else:
        pass

    try:
        faces = faceCascade.detectMultiScale(gray, 1.1, 7)
    except Exception as e:
        print('No faces found')
        cv2.imshow('img',img)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 6)
        if(isinstance(eyes, tuple)):
            continue
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        eyeRect = getLeftMostEye(eyes)
        (ex, ey, ew, eh) = eyeRect
        eye = roi_gray[ey:ey+eh, ex:ex+ew] #face rect
        eyeColor =roi_color[ey:ey+eh, ex:ex+ew] #face rect with color
        eye = cv2.equalizeHist(eye)
        circles = cv2.HoughCircles(eye, cv2.HOUGH_GRADIENT, 1, 20, 1, 300, 7, 7, 12)
        if(len(circles) > 0):
            eyeball = getEyeBall(eye, circles)
            if(type(eyeball) == np.ndarray):
        #		print(ex)
        # 		print(eyeball[0])
        # 		print(ey)
        # 		print(eyeball[1])
                 cv2.circle(eyeColor, (int(eyeball[0]), int(eyeball[1])),int(eyeball[2]), (0, 0, 255), 2 )
                 target_x = (x+ex+x+ex+ew) / 2
                 target_y = (y+ey+y+ey+eh) / 2 + ey/2
                 return np.array([target_x,target_y])

    return None


def getLeftMostEye(eyes):
    leftmost = 99999999
    leftmostIndex = -1
    i = 0
    for (x, y, w, h) in eyes:
        if(x < leftmost):
            leftmost = x
            leftmostIndex = i
        i += 1
    return eyes[leftmostIndex]

def getEyeBall(eye, circles):
    sums = np.zeros(len(circles), np.int)
    for y in range(0, eye.shape[0]):
        for x in range(0, eye.shape[1]):
            value = eye[x][y]
            for i in range(0, circles.shape[0]):
                try:
                    center = (circles[0][i][0], circles[0][i][1])
                    radius = circles[0][i][2]
                    if((((x-center[0]) * (x-center[0])) +( (y - center[1]) * (y - center[1]))) < (radius * radius)):
                        sums[i] += value
                except IndexError:
                    return None

    smallestSum = 9999999
    smallestSumIndex = -1

    for i in range(0, circles.shape[0]):
        if(sums[i] < smallestSum):
            smallestSum = sums[i]
            smallestSumIndex = i
    
    return circles[0][smallestSumIndex]


    


main()
