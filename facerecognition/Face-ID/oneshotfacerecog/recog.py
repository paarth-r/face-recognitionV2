from PIL import Image, ImageDraw
import face_recognition
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os, shutil
from matplotlib.patches import PathPatch
import cv2
import warnings
warnings.filterwarnings('ignore')

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f



def recognise(url3, savename):
  foundfaces = []
  known_face_encodings = [
      face_recognition.face_encodings(face_recognition.load_image_file(f"/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/known/{n}"))[0] for n in listdir_nohidden("/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/known")
  ]
  known_face_names = [
      n for n in listdir_nohidden("/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/known")
  ]

  # Load an image with an unknown face
  unknown_image = face_recognition.load_image_file(url3)

  # Find all the faces and face encodings in the unknown image
  face_locations = face_recognition.face_locations(unknown_image)
  face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

  # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
  # See http://pillow.readthedocs.io/ for more about PIL/Pillow
  pil_image = Image.fromarray(unknown_image)
  # Create a Pillow ImageDraw Draw instance to draw with
  draw = ImageDraw.Draw(pil_image)

  # Loop through each face found in the unknown image
  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      
      
      # See if the face is a match for the known face(s)
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

      name = "Unknown"

      # If a match was found in known_face_encodings, just use the first one.
      try:
          first_match_index = matches.index(True)
          name = known_face_names[first_match_index][0:-4]
          foundfaces.append(name)
          
      except:
          name = "Unknown"
          foundfaces.append(name)

      # Draw a box around the face using the Pillow module
      draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

      # Draw a label with a name below the face
      text_width, text_height = draw.textsize(name)
      draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
      draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


  # Remove the drawing library from memory as per the Pillow docs
  del draw

  # Display the resulting image
  plt.imshow(pil_image, aspect='auto')

  # Save the resulting image
  found = '_'.join(foundfaces)
  pil_image.save(f"/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/recognised/{found+savename}.jpg")

images = [img for img in os.listdir("/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/unknown") if img.endswith(".jpg")]
print(images)
for n, imagename in enumerate(images):
    print("recognising")
    recognise(f"/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/unknown/{imagename}", f"frame_{n}")




