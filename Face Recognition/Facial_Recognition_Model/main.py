import numpy as np
import pandas
import cv2
import face_recognition
from datetime import datetime
from keras.models import model_from_json
from keras.preprocessing import image

# load dataset and tuples
df=pandas.DataFrame(columns=["Name","Start","End","Emotions"])
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#load models, weights, data frame, and tuples
model = model_from_json(open("Models_And_Weights/model_unit.json", "r").read())
model.load_weights('Models_And_Weights/weights.h5')
face_haar_cascade = cv2.CascadeClassifier('Models_And_Weights/haarcascade_frontalface_default.xml')

# Initilize some variables
predicted_emotion = "Unknown"
name = "Unknown"
img_pixels = 0
first_frame=None
status_list=[None,None]
face_locations = []
face_names = []
second_face_names = []
emotion = []
second_emotion = []
address_locations = []
times=[]
process_this_frame = True

# Load a sample picture and learn how to recognize it.
sample_image = face_recognition.load_image_file("Input_Images/ganesh.jpg")
sample_face_encoding = face_recognition.face_encodings(ganesh_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    sample_face_encoding, sample_face_encoding_2
]
known_face_names = [
    "Sample Image Name", "Sample Image Name"
]

video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video

    ret, frame = video_capture.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray, 1.32, 5) 
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy = 0.25)

    rgb_small_frame = small_frame[:, :, ::-1]
    
    
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        second_face_names = []
        second_emotion = []
        for (x,y,w,h), face_encoding in zip( faces_detected, face_encodings):
            name = "Unknown"
            predicted_emotion = "Unknown"
            roi_gray=gray[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255
            
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                predicted_emotion = emotions[max_index]
                
            second_face_names.append(name)
            second_emotion.append(predicted_emotion)

    process_this_frame = not process_this_frame
    
    times.append(datetime.now())
    face_names.append(name)
    emotion.append(predicted_emotion)

    # Display the results
    for (top, right, bottom, left), name, predicted_emotion in zip(face_locations, second_face_names, second_emotion):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, predicted_emotion, ( left + 6, top - 6), font, 1.0, (0,0,255), 1)
        

    # Display the resulting imag.63-+.63-+.63-+.63-+.63-+.63-+.63-+ e
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
            times.append(datetime.now())
            face_names.append(name)
            emotion.append(predicted_emotion)
            break


for i in range(0, len(times) - 1,2):
    df=df.append({"Name":face_names[i],"Start":times[i],"End":times[i+1], "Emotions": emotion[i]},ignore_index=True)
    
df.to_csv("Output_Files/Dataset_File/Times.csv")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()