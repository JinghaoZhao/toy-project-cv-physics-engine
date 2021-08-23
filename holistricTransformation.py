# https://colab.research.google.com/drive/16UOYQ9hPM6L5tkq7oQBl1ULJ8xuK5Lae?usp=sharing#scrollTo=vp-ohtBNSFkj
import socket

import cv2
import mediapipe as mp
import numpy as np
import time

from util import to_trans_dict, angle_axis_to_string

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

UDP_IP = "192.168.0.34"
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
    Returns the angle in degrees between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def get_direction(landmark1, landmark2):
    return unit_vector(np.asarray(diff(landmark1, landmark2)))


def middle(l1, l2):
    mid = l1
    mid.x = (l1.x + l2.x) / 2
    mid.y = (l1.y + l2.y) / 2
    mid.z = (l1.z + l2.z) / 2
    return mid


def get_rotation(prvDirection, direction):
    V = np.cross(prvDirection, direction)
    A = angle_between(prvDirection, direction)
    return "{},{},{},{}".format(A, *V)


# For video input:
# cap = cv2.VideoCapture("media/dance.mov")
cap = cv2.VideoCapture(1)

with mp_holistic.Holistic(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():
        print(time.time())
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

        # mp_drawing.draw_landmarks(
        #     image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

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
        POSE = []

        handIndex = open("HAND_INDEX.txt", "r")
        poseIndex = open("BODY_INDEX.txt", "r")

        LEFT_HAND_MESSAGE = ["LHM", [], handIndex]
        RIGHT_HAND_MESSAGE = ["RHM", [], handIndex]
        FACE_MESSAGE = ["FACE", []]
        BODY_MESSAGE = ["BODY", []]

        # udp_send(LEFT_WRIST)
        if results.left_hand_landmarks and False:
            for landmark in results.left_hand_landmarks.landmark:
                LEFT_HAND.append(landmark)
            wristDirection = get_direction(LEFT_HAND[0], LEFT_HAND[9])
            wristRotationV = np.cross(up, wristDirection)
            wristRotationA = angle_between(up, wristDirection)
            wristRotation = "{},{},{},{}".format(wristRotationA, *wristRotationV)
            LEFT_HAND_MESSAGE[1].append(wristRotation)
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
                rotation = "{},{},{},{}".format(A, *V)
                LEFT_HAND_MESSAGE[1].append(rotation)

        if results.right_hand_landmarks and False:
            for landmark in results.right_hand_landmarks.landmark:
                RIGHT_HAND.append(landmark)
            wristDirection = get_direction(RIGHT_HAND[0], RIGHT_HAND[9])
            wristRotationV = np.cross(up, wristDirection)
            wristRotationA = angle_between(up, wristDirection)
            wristRotation = "{},{},{},{}".format(wristRotationA, *wristRotationV)
            RIGHT_HAND_MESSAGE[1].append(wristRotation)
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
                rotation = "{},{},{},{}".format(A, *V)
                RIGHT_HAND_MESSAGE[1].append(rotation)

        if results.pose_world_landmarks:
            # This steps takes 6.3ms to process
            trans_dict = to_trans_dict(results.pose_world_landmarks.landmark,
                                       results.left_hand_landmarks,
                                       results.right_hand_landmarks,)
            for k in trans_dict.keys():
                msg = k + "]" + angle_axis_to_string(trans_dict[k])
                BODY_MESSAGE[1].append(msg)
            # for landmark in results.pose_world_landmarks.landmark:
            #     POSE.append(landmark)
            # # Head
            # Head = get_direction(middle(POSE[10], POSE[9]), POSE[0])
            # HeadR = get_rotation(up, Head)
            # # HeadTop_End
            # HeadT = get_direction(POSE[0], middle(POSE[4], POSE[1]))
            # HeadTR = get_rotation(Head, HeadT)
            # # Neck
            # Neck = get_direction(middle(POSE[10], POSE[9]), middle(POSE[12], POSE[11]))
            # NeckR = get_rotation(up, Neck)
            # for i in range(11, 33):
            #     msg = ""
            #     # Shoulder
            #     # if i == 11 or i == 12:
            #     #     Shoulder = get_direction(POSE[i+12],POSE[i])
            #     #     ShoulderR = get_rotation(up,Shoulder)
            #     #     msg = "Shoulder]" + ShoulderR
            #     # Arm
            #     if i == 13 or i == 14:
            #         Elbow = get_direction(POSE[i - 2], POSE[i])
            #         if i == 13:
            #             ElbowR = get_rotation(get_direction(POSE[i - 1], POSE[i - 2]), Elbow)
            #         else:
            #             ElbowR = get_rotation(get_direction(POSE[i - 2], POSE[i - 3]), Elbow)
            #         msg = "Arm]" + ElbowR
            #     # ForeArm
            #     if i == 15 or i == 16:
            #         Wrist = get_direction(POSE[i - 2], POSE[i])
            #         WristR = get_rotation(get_direction(POSE[i - 4], POSE[i - 2]), Elbow)
            #         msg = "ForeArm]" + WristR
            #     # Hand
            #     if i == 19 or i == 20:
            #         Hand = get_direction(POSE[i - 4], POSE[i])
            #         HandR = get_rotation(get_direction(POSE[i - 6], POSE[i - 4]), Hand)
            #         msg = "Hand]" + HandR
            #     # Hips
            #     if i == 23:
            #         Hips = get_direction(middle(POSE[23], POSE[24]), middle(POSE[11], POSE[12]))
            #         HipsR = get_rotation(up, Hips)
            #         msg = "Hips]" + HipsR
            #     # Upleg
            #     if i == 25 or i == 26:
            #         Knee = get_direction(POSE[i - 2], POSE[i])
            #         KneeR = get_rotation(get_direction(middle(POSE[23], POSE[24]), middle(POSE[11], POSE[12])), Knee)
            #         msg = "UpLeg]" + KneeR
            #     # foot
            #     if i == 27 or i == 28:
            #         Ankle = get_direction(POSE[i - 2], POSE[i])
            #         AnkleR = get_rotation(get_direction(POSE[i - 4], POSE[i - 2]), Ankle)
            #         msg = "Foot]" + AnkleR
            #     # toe base
            #     if i == 31 or i == 32:
            #         Toe = get_direction(POSE[i - 4], POSE[i])
            #         ToeR = get_rotation(get_direction(POSE[i - 6], POSE[i - 4]), Toe)
            #         msg = "Toe_Base]" + ToeR
            #     # toe end
            #     if i == 29 or i == 30:
            #         Heel = get_direction(POSE[i - 2], POSE[i])
            #         HeelR = get_rotation(get_direction(POSE[i - 4], POSE[i - 2]), Heel)
            #         msg = "Toe_End]" + HeelR
            #     if msg != "":
            #         msg = msg + " "
            #         if i != 23:
            #             if i % 2 == 0:
            #                 msg = "Right" + msg
            #             else:
            #                 msg = "Left" + msg
            #     if i in [14]:
            #         BODY_MESSAGE[1].append(msg)

        # print(handIndex.readline().split(". ")[1].split("\n")[0])
        # """
        # for i in [LEFT_HAND_MESSAGE, RIGHT_HAND_MESSAGE, FACE_MESSAGE]:
        #     if i[1] != []:
        #         MSG = i[0] + ")"
        #         for j in i[1]:
        #             try:
        #                 MSG += (i[2].readline().split(". ")[1].split("\n")[0] + ":" + j + " ")
        #             except:
        #                 pass
        #         print(MSG)
        # udp_send(MSG.encode())
        try:
            MSG = "BODY)" + " ".join(BODY_MESSAGE[1])
            print(MSG)
            udp_send(MSG.encode())
        except:
            pass

        # try:
        #     print(middle(results.face_landmarks.landmark[10],results.face_landmarks.landmark[9]))
        # except:
        #     passf

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
