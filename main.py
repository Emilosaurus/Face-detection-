import face_recognition
import cv2
import numpy as np
import os
import csv
from datetime import datetime
import time
import pyttsx3  # Text-to-speech library

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech rate (words per minute)
tts_engine.setProperty('volume', 0.9)  # Adjust volume (0.0 to 1.0)

# Load known faces and their names from the 'faces' folder
known_face_encodings = []
known_face_names = []

for filename in os.listdir('faces'):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        known_name = os.path.splitext(filename)[0]
        known_image = face_recognition.load_image_file(f'faces/{filename}')
        known_encoding = face_recognition.face_encodings(known_image)[0]
        known_face_encodings.append(known_encoding)
        known_face_names.append(known_name)

# Initialize variables for face recognition
face_locations = []
face_encodings = []
face_names = []

# Track announced names to avoid repeated announcements
announced_names = set()
last_announcement_time = {}

# Start capturing video from the default camera (0)
video_capture = cv2.VideoCapture(0)

# Open a CSV file for writing
with open('face_log.csv', mode='w', newline='') as csv_file:
    fieldnames = ['Name', 'Time']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    last_write_time = time.time()
    frame_counter = 0

    while True:
        # Capture a single frame from the camera
        ret, frame = video_capture.read()
        
        frame_counter += 1
        if frame_counter % 5 == 0:  # Process every 5th frame
            frame_counter = 0

            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_names = []
            current_time = time.time()
            
            for face_encoding in face_encodings:
                # Compare the face encoding to known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = "Unknown"

                # Calculate face distance (lower is better)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = f"{100 - face_distances[best_match_index] * 100:.2f}%"
                    
                    # Announce the name if it hasn't been announced recently
                    # Wait at least 5 seconds between announcements for the same person
                    if (name not in last_announcement_time or 
                        current_time - last_announcement_time[name] > 5):
                        
                        # Announce the detected person's name
                        announcement = f"{name}, you are recognized."
                        print(f"Announcing: {announcement}")
                        tts_engine.say(announcement)
                        tts_engine.runAndWait()
                        
                        # Update the last announcement time
                        last_announcement_time[name] = current_time

                face_names.append(f"{name} ({confidence})")

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name and confidence above the face
                cv2.rectangle(frame, (left, top - 20), (right, top), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, top - 6), font, 0.5, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Write to the CSV file every 10 seconds
            if time.time() - last_write_time >= 10:
                for name in face_names:
                    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow({'Name': name, 'Time': current_time_str})
                last_write_time = time.time()

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

# Stop the TTS engine
tts_engine.stop()