# face-recognitionV2
This is my updated face recognition system, coded in python.
Add the faces of the people to be added to the model into the known folder.
Make sure to go through the code and change the file names and patterns to that of your own machine.

Dependencies:
cv2
pillow
face_recognition
matplotlib
shutil

To acquire data for the unknown section, simply run the recog.py file. It will detect only for faces and save the files to unknown. 
Then, run face_recog once you have gathered the files and the recognised images will show up in recognised. 
(it auto caps at 50, change the for loop variable for more)

