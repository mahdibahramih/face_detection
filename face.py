import cv2
import numpy as np

face = cv2.CascadeClassifier('eye.xml')
eye = cv2.CascadeClassifier('ffffffff.xml')
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face.detectMultiScale(gray)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h,x:x+h]
		roi_color = frame[y:y+h,x:x+h]
		eyes = eye.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
		    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
	cv2.imshow('Frame',frame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break


cap.release() 
cv2.destroyAllWindows()
