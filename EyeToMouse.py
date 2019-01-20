# With reference to this tutorial
# https://picoledelimao.github.io/blog/2017/01/28/eyeball-tracking-for-mouse-control-in-opencv/
# And various openCV tutorials

import cv2
import numpy as np
import win32api
from win32api import GetSystemMetrics
import win32con

def main():
	webCam = cv2.VideoCapture(0)
	while (True):
		ret, img = webCam.read()


		#cv2.imshow('frame', frame)
		if cv2.waitKey(30) & 0xFF == ord('q'):
			break

		face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
		eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

		detectEyes(img, face_cascade, eye_cascade)
		click(mousePoint[0], mousePoint[1])
	
		cv2.imshow('img',img)
	
	webCam.release()
	cv2.destroyAllWindows()

def detectEyes(img, faceCascade, eyeCascade):
	global lastPoint
	global mousePoint
	try:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)
	except Exception as e:
		print('img not loaded')
		cv2.imshow('img',img)
	else:
		pass

	try:
		faces = faceCascade.detectMultiScale(gray, 1.1, 3)
	except Exception as e:
		print('No faces found')
		cv2.imshow('img',img)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 3)
		if(isinstance(eyes, tuple)):
			continue
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		eyeRect = getLeftMostEye(eyes)
		(ex, ey, ew, eh) = eyeRect
		eye = roi_gray[ey:ey+eh, ex:ex+ew] #face rect
		eyeColor =roi_color[ey:ey+eh, ex:ex+ew] #face rect with color
		eye = cv2.equalizeHist(eye)
		circles = cv2.HoughCircles(eye, cv2.HOUGH_GRADIENT, 1, 6, 1, 300, 7, 8, 13)
		if(len(circles) > 0):
			eyeball = getEyeBall(eye, circles)
			if(type(eyeball) == np.ndarray):
		#		print(ex)
		# 		print(eyeball[0])
		# 		print(ey)
		# 		print(eyeball[1])
				centers.append((int(eyeball[0]), int(eyeball[1])))
				stableCenter = stabilize(centers, 5)
				cv2.circle(eyeColor, stableCenter , int(eyeball[2]), (0, 0, 255), 2 )
				if(len(centers) > 0):
					xVal = int((stableCenter[0] / ew) * GetSystemMetrics(0))
					yVal = int((stableCenter[1] / eh) * GetSystemMetrics(1))
					mousePoint = (xVal, yVal)
					print(mousePoint)
				lastPoint = stableCenter

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

def stabilize(points, period):
	sumX = 0
	sumY = 0
	count = 0
	while(len(points) > period):
		points.pop(0)
	for i in range(max(0, len(points) - period), len(points)):
		sumX += points[i][0]
		sumY += points[i][1]
		count += 1
	if(count > 0):
		sumX = sumX / count
		sumY = sumY / count
	return (int(sumX), int(sumY))

def click(x, y):
	if(x > GetSystemMetrics(0)): x = GetSystemMetrics(0)
	if(x < 0): x = 0
	if(y > GetSystemMetrics(1)): y = GetSystemMetrics(1)
	if(y < 0): y = 0
	print("{:d} {:d}".format(x, y))
	win32api.SetCursorPos((x, y))
	# win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
	# win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

centers = []
lastPoint = (500, 500)
mousePoint = (500, 500)

main()