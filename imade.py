import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read an image or access a video stream
image = cv2.imread('image.jpg')

# Convert the image to grayscale (face detection works better in grayscale)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Display the image with detected faces
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


import face_recognition

# Load known face encodings and their corresponding names
known_face_encodings = [...]
known_face_names = [...]

# Load the image to be recognized
unknown_image = face_recognition.load_image_file("unknown.jpg")

# Find face locations in the image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for face_encoding in face_encodings:
    # Compare the face encoding to the known face encodings
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"  # Default name if no match is found

    # If a match is found, use the name of the known person
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    print(f"Found {name} in the image.")
