import socket

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


UDP_IP = "192.168.0.70"
UDP_PORT = 5006
sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP
def udp_send(message):
    sock.sendto(message, (UDP_IP, UDP_PORT))

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Pose', image)
        # [mp_pose.PoseLandmark.RIGHT_ELBOW].z
        # print(results.pose_landmarks.landmark)
        print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z)
        # udp_send(results.pose_landmarks.landmark)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
