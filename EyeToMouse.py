


import cv2

webCam = cv2.VideoCapture(0)

while (True):
	ret, img = webCam.read()


	#cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

	try:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	except Exception as e:
		print('img not loaded')
		continue
	else:
		pass


	try:
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	except Exception as e:
		print('No faces found')
		cv2.imshow('img',img)
		continue
	
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	cv2.imshow('img',img)




webCam.release()
cv2.destroyAllWindows()