import cv2
import time
import numpy as np

face = cv2.CascadeClassifier('face.xml')
eye = cv2.CascadeClassifier('eye.xml')
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face.detectMultiScale(gray)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.circle(frame,(x+w,y-20),6,(255,255,255),-1,20)
		cv2.circle(frame,(x+w+14,y-20-12),8,(255,255,255),-1,20)
		cv2.circle(frame,(x+w+32,y-20-29),10,(255,255,255),-1,20)
		cv2.circle(frame,(x+w+42,y-8),20,(255,0,0),1,20)
		cv2.imshow('Frame',frame)
		time.sleep(2)                
		"""roi_gray = gray[y:y+h,x:x+h]
		roi_color = frame[y:y+h,x:x+h]
		eyes = eye.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)"""
	cv2.imshow('Frame',frame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break


cap.release() 
cv2.destroyAllWindows()
