import cv2
from random import *

#load some pre-trained data on face frontals from opencv (haar cascade algorithm)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')

#getting webcam (u can add any vieo instead of 0)
webCam = cv2.VideoCapture(0)

#iterate forever over frames
while True:
    #Read the current frame
    successful_frame_read, frame = webCam.read()

    if not successful_frame_read:
        break

    #grayscaled_image
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces
    faces_coordinates = face_detector.detectMultiScale(grey_image)
    

    #run face detection within each of the faces
    for (x,y,w,h) in faces_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

        #get the sub frame using numpy N-dimensional array slicing
        the_face = frame[y:y+h,x:x+w]
        #grayscaled_face
        grey_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        #detect smiles
        smile_coordinates = smile_detector.detectMultiScale(grey_face,scaleFactor=1.7,minNeighbors=70)

        # #find all the smiles in the face
        # for (x2,y2,w2,h2) in smile_coordinates:
        #     cv2.rectangle(the_face,(x2,y2),(x2+w2,y2+h2),(randrange(256),randrange(256),randrange(256)),2)
        
        #label this face as smiling
        if len(smile_coordinates)>0:
            cv2.putText(frame,'smiling',(x,y+h+40),fontScale=3,
            fontFace=cv2.FONT_HERSHEY_PLAIN,color=[255,255,255])

    #display the image with rectangle around the face
    cv2.imshow('Smile Detector',frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break
webCam.release()
cv2.destroyAllWindows()

"""
#choose an image to detect faces in 
img = cv2.imread('faces.jpg')


#grayscaled_image
grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_faced_data.detectMultiScale(grey_image)

#draw a rectangle around the image
for (x,y,w,h) in face_coordinates:
    print(x)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#display the image with rectangle around the face
cv2.imshow('face',img)
cv2.waitKey()

"""

