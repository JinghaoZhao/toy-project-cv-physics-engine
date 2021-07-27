import socket
import time

import cv2
import mediapipe as mp
import numpy as np

UDP_IP = "192.168.0.70"
UDP_PORT = 5006
sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP


def udp_send(message):
    sock.sendto(message, (UDP_IP, UDP_PORT))


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        # print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            # print('hand_landmarks:', hand_landmarks)
            # print(
            #     f'Index finger tip coordinates: (',
            #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            # )
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite(
            '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))

# For webcam input:
cap = cv2.VideoCapture(0)


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


with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
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
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        '''
        Normalized X gives 0 to 1 where x-origin is origin of the image x-coordinate
        Normalized Y gives 0 to 1 where y-origin is origin of the image y-coordinate
        Normalized Z where z-origin is relative to the wrist z-origin. I.e if Z is positive, 
        the z-la ndmark coordinate is out of the page with respect to the wrist. Z is negative, 
        the z-landmark coordinate is into the page with respect of the wrist.
        '''
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # print(hand_landmarks)
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Calculate the hand direction
                landmark = list(hand_landmarks.landmark)

                hand_direction = np.asarray(diff(landmark[0], landmark[9]))
                hand_direction = unit_vector(hand_direction)
                up = np.asarray([0, 1, 0])
                rotation_vec = np.cross(up, hand_direction)
                rotation_angle = angle_between(up, hand_direction)

                # print("\n", hand_landmarks, "\n")

                """
                wrist = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,3)) + ", " \
                      + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,3)) + ", " \
                      + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z,3))
                thumbCMC = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x, 3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y, 3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z, 3))
                thumbMCP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z,3))
                thumbIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,3)) + ", " \
                        + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,3)) + ", " \
                        + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z,3))
                thumbTIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z,3))
                indexMCP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z,3))
                indexPIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z,3))           
                indexDIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z,3))
                indexTIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,3))
                middleMCP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z,3))           
                middlePIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z,3))           
                middleDIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z,3))           
                middleTIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z,3))                           
                ringMCP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z, 3))
                ringPIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z, 3))          
                ringDIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z, 3))          
                ringTIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z, 3))          
                pinkyMCP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z, 3))
                pinkyPIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x, 3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y, 3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z, 3))
                pinkyDIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x, 3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y, 3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z, 3))
                pinkyTIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x, 3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y, 3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z, 3))          
                 
                """

                # handHelp = open("UDP_HAND_HELP.txt","r")
                # for i in range(1,21):
                #     print(handHelp.readline().split(". ")[1])

                message = []
                wristDirection = unit_vector(np.asarray(diff(landmark[0], landmark[9])))
                wristRotationV = np.cross(up, wristDirection)
                wristRotationA = angle_between(up, wristDirection)
                wristRotation = "{}, {}, {}, {}".format(wristRotationA, *wristRotationV)
                message.append(wristRotation)
                for i in range(1,21):
                    if i in [1,5,9,13,17]:
                        direction = unit_vector(np.asarray(diff(landmark[0], landmark[i])))
                        V = np.cross(wristDirection, direction)
                        A = angle_between(wristDirection, direction)
                    if i in [2,6,10,14,18]:
                        direction = unit_vector(np.asarray(diff(landmark[i - 1], landmark[i])))
                        V = np.cross(unit_vector(np.asarray(diff(landmark[0], landmark[i-1]))), direction)
                        A = angle_between(unit_vector(np.asarray(diff(landmark[0], landmark[i-1]))), direction)
                    else:
                        direction = unit_vector(np.asarray(diff(landmark[i-1], landmark[i])))
                        V = np.cross(unit_vector(np.asarray(diff(landmark[i-2], landmark[i-1]))), direction)
                        A = angle_between(unit_vector(np.asarray(diff(landmark[i-2], landmark[i-1]))), direction)
                    rotation = "{}, {}, {}, {}".format(A, *V)
                    message.append(rotation)


                """
                thumbCMCDirection = unit_vector(np.asarray(diff(landmark[0], landmark[1])))
                thumbCMCRotationV = np.cross(wristDirection, thumbCMCDirection)
                thumbCMCRotationA = angle_between(wristDirection, thumbCMCDirection)
                thumbRotationCMC = "{}, {}, {}, {}".format(round(thumbCMCRotationA,3), *thumbCMCRotationV)
                thumbMCPDirection = unit_vector(np.asarray(diff(landmark[1], landmark[2])))
                thumbMCPRotationV = np.cross(thumbCMCDirection, thumbMCPDirection)
                thumbMCPRotationA = angle_between(thumbCMCDirection, thumbMCPDirection)
                thumbRotationMCP = "{}, {}, {}, {}".format(round(thumbMCPRotationA,3), *thumbMCPRotationV)
                thumbIPDirection = unit_vector(np.asarray(diff(landmark[2], landmark[3])))
                thumbIPRotationV = np.cross(thumbMCPDirection, thumbIPDirection)
                thumbIPRotationA = angle_between(thumbMCPDirection, thumbIPDirection)
                thumbRotationIP = "{}, {}, {}, {}".format(round(thumbIPRotationA,3), *thumbIPRotationV)
                thumbTIPDirection = unit_vector(np.asarray(diff(landmark[3], landmark[4])))
                thumbTIPRotationV = np.cross(thumbIPDirection, thumbTIPDirection)
                thumbTIPRotationA = angle_between(thumbIPDirection, thumbTIPDirection)
                thumbRotationTIP = "{}, {}, {}, {}".format(round(thumbTIPRotationA,3), *thumbTIPRotationV)

                # index finger

                indexMCPDirection = unit_vector(np.asarray(diff(landmark[0], landmark[5])))
                indexMCPRotationV = np.cross(up, indexMCPDirection)
                indexMCPRotationA = angle_between(up, indexMCPDirection)
                indexRotationMCP = "{}, {}, {}, {}".format(round(indexMCPRotationA,3), *indexMCPRotationV)
                indexPIPDirection = unit_vector(np.asarray(diff(landmark[5], landmark[6])))
                indexPIPRotationV = np.cross(indexMCPDirection, indexPIPDirection)
                indexPIPRotationA = angle_between(indexMCPDirection, indexPIPDirection)
                indexRotationPIP = "{}, {}, {}, {}".format(round(indexPIPRotationA,3), *indexPIPRotationV)
                indexDIPDirection = unit_vector(np.asarray(diff(landmark[6], landmark[7])))
                indexDIPRotationV = np.cross(indexPIPDirection, indexDIPDirection)
                indexDIPRotationA = angle_between(indexPIPDirection, indexDIPDirection)
                indexRotationDIP = "{}, {}, {}, {}".format(round(indexDIPRotationA,3), *indexDIPRotationV)
                indexTIPDirection = unit_vector(np.asarray(diff(landmark[7], landmark[8])))
                indexTIPRotationV = np.cross(indexDIPDirection, indexTIPDirection)
                indexTIPRotationA = angle_between(indexDIPDirection, indexTIPDirection)
                indexRotationTIP = "{}, {}, {}, {}".format(round(indexTIPRotationA,3), *indexTIPRotationV)

                # middle finger

                middleMCPDirection = unit_vector(np.asarray(diff(landmark[0], landmark[9])))
                middleMCPRotationV = np.cross(up, middleMCPDirection)
                middleMCPRotationA = angle_between(up, middleMCPDirection)
                middleRotationMCP = "{}, {}, {}, {}".format(round(middleMCPRotationA,3), *middleMCPRotationV)
                middlePIPDirection = unit_vector(np.asarray(diff(landmark[9], landmark[10])))
                middlePIPRotationV = np.cross(middleMCPDirection, middlePIPDirection)
                middlePIPRotationA = angle_between(middleMCPDirection, middlePIPDirection)
                middleRotationPIP = "{}, {}, {}, {}".format(round(middlePIPRotationA,3), *middlePIPRotationV)
                middleDIPDirection = unit_vector(np.asarray(diff(landmark[10], landmark[11])))
                middleDIPRotationV = np.cross(middlePIPDirection, middleDIPDirection)
                middleDIPRotationA = angle_between(middlePIPDirection, middleDIPDirection)
                middleRotationDIP = "{}, {}, {}, {}".format(round(middleDIPRotationA,3), *middleDIPRotationV)
                middleTIPDirection = unit_vector(np.asarray(diff(landmark[11], landmark[12])))
                middleTIPRotationV = np.cross(middleDIPDirection, middleTIPDirection)
                middleTIPRotationA = angle_between(middleDIPDirection, middleTIPDirection)
                middleRotationTIP = "{}, {}, {}, {}".format(round(middleTIPRotationA,3), *middleTIPRotationV)

                # ring finger

                ringMCPDirection = unit_vector(np.asarray(diff(landmark[0], landmark[13])))
                ringMCPRotationV = np.cross(up, ringMCPDirection)
                ringMCPRotationA = angle_between(up, ringMCPDirection)
                ringRotationMCP = "{}, {}, {}, {}".format(round(ringMCPRotationA,3), *ringMCPRotationV)
                ringPIPDirection = unit_vector(np.asarray(diff(landmark[13], landmark[14])))
                ringPIPRotationV = np.cross(ringMCPDirection, ringPIPDirection)
                ringPIPRotationA = angle_between(ringMCPDirection, ringPIPDirection)
                ringRotationPIP = "{}, {}, {}, {}".format(round(ringPIPRotationA,3), *ringPIPRotationV)
                ringDIPDirection = unit_vector(np.asarray(diff(landmark[14], landmark[15])))
                ringDIPRotationV = np.cross(ringPIPDirection, ringDIPDirection)
                ringDIPRotationA = angle_between(ringPIPDirection, ringDIPDirection)
                ringRotationDIP = "{}, {}, {}, {}".format(round(ringDIPRotationA,3), *ringDIPRotationV)
                ringTIPDirection = unit_vector(np.asarray(diff(landmark[15], landmark[16])))
                ringTIPRotationV = np.cross(ringDIPDirection, ringTIPDirection)
                ringTIPRotationA = angle_between(ringDIPDirection, ringTIPDirection)
                ringRotationTIP = "{}, {}, {}, {}".format(round(ringTIPRotationA,3), *ringTIPRotationV)

                # pinky

                pinkyMCPDirection = unit_vector(np.asarray(diff(landmark[0], landmark[17])))
                pinkyMCPRotationV = np.cross(up, pinkyMCPDirection)
                pinkyMCPRotationA = angle_between(up, pinkyMCPDirection)
                pinkyRotationMCP = "{}, {}, {}, {}".format(round(pinkyMCPRotationA,3), *pinkyMCPRotationV)
                pinkyPIPDirection = unit_vector(np.asarray(diff(landmark[17], landmark[18])))
                pinkyPIPRotationV = np.cross(pinkyMCPDirection, pinkyPIPDirection)
                pinkyPIPRotationA = angle_between(pinkyMCPDirection, pinkyPIPDirection)
                pinkyRotationPIP = "{}, {}, {}, {}".format(round(pinkyPIPRotationA,3), *pinkyPIPRotationV)
                pinkyDIPDirection = unit_vector(np.asarray(diff(landmark[18], landmark[19])))
                pinkyDIPRotationV = np.cross(pinkyPIPDirection, pinkyDIPDirection)
                pinkyDIPRotationA = angle_between(pinkyPIPDirection, pinkyDIPDirection)
                pinkyRotationDIP = "{}, {}, {}, {}".format(round(pinkyDIPRotationA,3), *pinkyDIPRotationV)
                pinkyTIPDirection = unit_vector(np.asarray(diff(landmark[19], landmark[20])))
                pinkyTIPRotationV = np.cross(pinkyDIPDirection, pinkyTIPDirection)
                pinkyTIPRotationA = angle_between(pinkyDIPDirection, pinkyTIPDirection)
                pinkyRotationTIP = "{}, {}, {}, {}".format(round(pinkyTIPRotationA, 3), *pinkyTIPRotationV)
                

                message = \
                "WR" + wristRotation + "WR" + \
                "thumb" + thumbRotationCMC + "x" + thumbRotationMCP + "x" + thumbRotationIP + "x" + thumbRotationTIP + "thumb" + \
                "index" + "x" + indexRotationMCP + "x" + indexRotationPIP + "x" + indexRotationDIP + "x" + indexRotationTIP + "index" + \
                "middle" + middleRotationMCP + "x" + middleRotationPIP + "x" + middleRotationDIP + "x" + middleRotationTIP + "middle" + \
                "ring" + ringRotationMCP + "x" + ringRotationPIP + "x" + ringRotationDIP + "x" + ringRotationTIP + "ring" + \
                "pinky" + pinkyRotationMCP + "x" + pinkyRotationPIP + "x" + pinkyRotationDIP + "x" + pinkyRotationTIP + "pinky"
                """

                udp_send(message.encode())
                print(message)
        time.sleep(0.1)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
