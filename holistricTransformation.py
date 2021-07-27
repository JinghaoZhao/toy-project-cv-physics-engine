# https://colab.research.google.com/drive/16UOYQ9hPM6L5tkq7oQBl1ULJ8xuK5Lae?usp=sharing#scrollTo=vp-ohtBNSFkj
import socket
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

UDP_IP = "192.168.0.70"
UDP_PORT = 5006
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def send_xyz(landmark):
    return str(landmark.x) + "," + str(landmark.y) + "," + str(landmark.z)

def udp_send(message):
    sock.sendto(message, (UDP_IP, UDP_PORT))

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as holistic:
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
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Holistic', image)


        LEFT_HAND = results.left_hand_landmarks
        RIGHT_HAND = results.right_hand_landmarks
        FACE = results.face_landmarks
        BODY = results.pose_landmarks

        # print(mp_holistic.HandLandmark)
        if LEFT_HAND is not None:
            LEFT_WRIST = LEFT_HAND.landmark[mp_holistic.HandLandmark.WRIST]
            print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].z)
            # udp_send(LEFT_WRIST)

        # if results.left_hand_landmarks.landmark != None:
        #
        #     LWRIST = str(round(results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x,3)) + "," \
        #             + str(round(results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y, 3)) + "," \
        #             + str(round(results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].z, 3))
            # print(LWRIST)




        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()
