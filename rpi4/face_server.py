import pickle
import socket

import cv2
import face_recognition
import numpy as np

localIP = "127.0.0.1"
localPort = 20001
bufferSize = 2048
msgFromServer = "Test"
bytesToSend = str.encode(msgFromServer)
# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))
print("UDP server up and listening")
# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("./assets/obama.png")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("assets/biden.png")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
yunqi_image = face_recognition.load_image_file("assets/yunqi.jpg")
yunqi_face_encoding = face_recognition.face_encodings(yunqi_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    yunqi_face_encoding
]

known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Yunqi Guo"
]


while True:
    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)

    face_encoding = pickle.loads(bytesAddressPair[0])
    address = bytesAddressPair[1]
    clientMsg = "Message from Client:{}".format(face_encoding)
    clientIP = "Client IP Address:{}".format(address)

    print(clientMsg)
    print(clientIP)
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Sending a reply to client
    UDPServerSocket.sendto(str.encode(name), address)
