# // <!--
# //  ============================
# //  @Author  :        Raja Durai M
# //  @Version :        1.0
# //  @Date    :        20 Jul 2021
# //  @Detail  :        Automatic Registration Number Detector
# //  ============================
# //  -->

import cv2
import pytesseract
import numpy as np

plate_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
input_video = cv2.VideoCapture('rec.mp4')

if(input_video.isOpened() == False):
	print('Error! Unable to open file')

registration_plate = None
hashTable = []
isAlreadyDisplayed = False

while True:
	check,frame = input_video.read()
	gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	allDetectedPlates = plate_detector.detectMultiScale(gray_scale_frame,scaleFactor=1.2,minNeighbors=5,minSize=(25,25))	#list of 4 coordinates

	for (x,y,w,h) in allDetectedPlates:
		# Marking the registration plate & displaying as detected
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
		cv2.putText(frame,text='Detected',org=(x-3,y-3),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,255,0),thickness=1,fontScale=0.6)

		# Saving the registration plate image
		registration_plate = gray_scale_frame[y:y + h, x:x + w]

		# Noise Cancellation for better performance
		registration_plate = cv2.bilateralFilter(registration_plate, 11, 17, 17)
		(thresh, registration_plate) = cv2.threshold(registration_plate, 150, 180, cv2.THRESH_BINARY)

		# Text Recognition by Pytesseract & conversion from image to text
		text = pytesseract.image_to_string(registration_plate, config="--psm 7")
		
		# Checking for duplicates & printing each registration number only once!
		for t1 in hashTable:
			if t1==text:
				isAlreadyDisplayed = True
				break
		if not isAlreadyDisplayed :
		   	print('Vehicle Registration Number :', text,"\n")
		   	hashTable.append(text)
		isAlreadyDisplayed = False
		
	# Terminating condition. Checking whether video is completed or not.
	if check == True:
		cv2.imshow('Processed video', frame)

		if cv2.waitKey(25) & 0xFF == ord("q"):
			break
	else:
		break
input_video.release()
cv2.destroyAllWindows()