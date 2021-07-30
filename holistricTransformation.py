# https://colab.research.google.com/drive/16UOYQ9hPM6L5tkq7oQBl1ULJ8xuK5Lae?usp=sharing#scrollTo=vp-ohtBNSFkj
import socket
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

UDP_IP = "192.168.0.70"
UDP_PORT = 5006
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def send_xyz(landmark):
    return str(landmark.x) + "," + str(landmark.y) + "," + str(landmark.z)

def udp_send(message):
    sock.sendto(message, (UDP_IP, UDP_PORT))

def diff(landmark1, landmark2):
    return [landmark1.x - landmark2.x, landmark1.y - landmark2.y, landmark1.z - landmark2.z]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def get_direction(landmark1, landmark2):
    return unit_vector(np.asarray(diff(landmark1, landmark2)))

def middle(l1,l2):
    mid = l1
    mid.x = l1.x + l2.x


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as holistic:
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

        up = np.asarray([0, 1, 0])
        LEFT_HAND = []
        RIGHT_HAND = []
        FACE = []
        BODY = []

        LEFT_HAND_MESSAGE = []
        RIGHT_HAND_MESSAGE = []
        FACE_MESSAGE = []
        BODY_MESSAGE = []

            # udp_send(LEFT_WRIST)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                LEFT_HAND.append(landmark)
            wristDirection = get_direction(LEFT_HAND[0], LEFT_HAND[9])
            wristRotationV = np.cross(up, wristDirection)
            wristRotationA = angle_between(up, wristDirection)
            wristRotation = "{}, {}, {}, {}".format(wristRotationA, *wristRotationV)
            LEFT_HAND_MESSAGE.append(wristRotation)
            for i in range(1, 21):
                if i in [1, 5, 9, 13, 17]:
                    direction = get_direction(LEFT_HAND[0], LEFT_HAND[i])
                    V = np.cross(wristDirection, direction)
                    A = angle_between(wristDirection, direction)
                if i in [2, 6, 10, 14, 18]:
                    direction = get_direction(LEFT_HAND[i - 1], LEFT_HAND[i])
                    V = np.cross(unit_vector(np.asarray(diff(LEFT_HAND[0], LEFT_HAND[i - 1]))), direction)
                    A = angle_between(unit_vector(np.asarray(diff(LEFT_HAND[0], LEFT_HAND[i - 1]))), direction)
                else:
                    direction = get_direction(LEFT_HAND[i - 1], LEFT_HAND[i])
                    V = np.cross(unit_vector(np.asarray(diff(LEFT_HAND[i - 2], LEFT_HAND[i - 1]))), direction)
                    A = angle_between(unit_vector(np.asarray(diff(LEFT_HAND[i - 2], LEFT_HAND[i - 1]))), direction)
                rotation = "{}, {}, {}, {}".format(A, *V)
                LEFT_HAND_MESSAGE.append(rotation)

        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                RIGHT_HAND.append(landmark)
            wristDirection = get_direction(RIGHT_HAND[0], RIGHT_HAND[9])
            wristRotationV = np.cross(up, wristDirection)
            wristRotationA = angle_between(up, wristDirection)
            wristRotation = "{}, {}, {}, {}".format(wristRotationA, *wristRotationV)
            RIGHT_HAND_MESSAGE.append(wristRotation)
            for i in range(1, 21):
                if i in [1, 5, 9, 13, 17]:
                    direction = get_direction(RIGHT_HAND[0], RIGHT_HAND[i])
                    V = np.cross(wristDirection, direction)
                    A = angle_between(wristDirection, direction)
                if i in [2, 6, 10, 14, 18]:
                    direction = get_direction(RIGHT_HAND[i - 1], RIGHT_HAND[i])
                    V = np.cross(unit_vector(np.asarray(diff(RIGHT_HAND[0], RIGHT_HAND[i - 1]))), direction)
                    A = angle_between(unit_vector(np.asarray(diff(RIGHT_HAND[0], RIGHT_HAND[i - 1]))), direction)
                else:
                    direction = get_direction(RIGHT_HAND[i - 1], RIGHT_HAND[i])
                    V = np.cross(unit_vector(np.asarray(diff(RIGHT_HAND[i - 2], RIGHT_HAND[i - 1]))), direction)
                    A = angle_between(unit_vector(np.asarray(diff(RIGHT_HAND[i - 2], RIGHT_HAND[i - 1]))), direction)
                rotation = "{}, {}, {}, {}".format(A, *V)
                RIGHT_HAND_MESSAGE.append(rotation)

        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                FACE.append(landmark)

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                BODY.append(landmark)
            # Face



        print(LEFT_HAND_MESSAGE)
        print(RIGHT_HAND_MESSAGE)



        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()
