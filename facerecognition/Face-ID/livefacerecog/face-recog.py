import cv2
import os

# for face detection
face_cascade = cv2.CascadeClassifier('*insert your own path here*/facerecognition/Face-ID/livefacerecog/haarcascade_frontalface_alt2.xml')

# resolution of the webcam
screen_width = 1280       # try 640 if code fails
screen_height = 720

# default webcam
stream = cv2.VideoCapture(0)         #for mac
count = 1
while True:
    # capture frame-by-frame
    (grabbed, frame) = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)

    # for each faces found
    for (x, y, w, h) in faces:
        #save face
        if count < 50:
            cv2.imwrite(os.path.join("*insert your own path here*/facerecognition/Face-ID/oneshotfacerecog/unknown", f'{count}.jpg'), frame)
            count += 1
        # Draw a rectangle around the face
        color = (255, 0, 0) # in BGR
        stroke = 3
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
        


    # show the frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break                  #q to exit out of the loop

# cleanup
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
