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
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
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


                wrist = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,3)) + ", " \
                      + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,3)) + ", " \
                      + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z,3))

                wristDirection = unit_vector(np.asarray(diff(landmark[0], landmark[9])))
                wristRotationV = np.cross(up, wristDirection)
                wristRotationA = angle_between(up, wristDirection)
                wristRotation = "{}, {}, {}, {}".format(round(wristRotationA,3), *wristRotationV)

                thumbCMC = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,3)) + ", " \
                         + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,3)) + ", " \
                         + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z,3))
                thumbCMCDirection = unit_vector(np.asarray(diff(landmark[0], landmark[1])))
                thumbCMCRotationV = np.cross(wristDirection, thumbCMCDirection)
                thumbCMCRotationA = angle_between(wristDirection, thumbCMCDirection)
                thumbRotationCMC = "{}, {}, {}, {}".format(round(thumbCMCRotationA,3), *thumbCMCRotationV)
                thumbMCP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z,3))
                thumbMCPDirection = unit_vector(np.asarray(diff(landmark[1], landmark[2])))
                thumbMCPRotationV = np.cross(thumbCMCDirection, thumbMCPDirection)
                thumbMCPRotationA = angle_between(thumbCMCDirection, thumbMCPDirection)
                thumbRotationMCP = "{}, {}, {}, {}".format(round(thumbMCPRotationA,3), *thumbMCPRotationV)
                thumbIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,3)) + ", " \
                        + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,3)) + ", " \
                        + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z,3))
                thumbIPDirection = unit_vector(np.asarray(diff(landmark[2], landmark[3])))
                thumbIPRotationV = np.cross(thumbMCPDirection, thumbIPDirection)
                thumbIPRotationA = angle_between(thumbMCPDirection, thumbIPDirection)
                thumbRotationIP = "{}, {}, {}, {}".format(round(thumbIPRotationA,3), *thumbIPRotationV)
                thumbTIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z,3))
                thumbTIPDirection = unit_vector(np.asarray(diff(landmark[3], landmark[4])))
                thumbTIPRotationV = np.cross(thumbIPDirection, thumbTIPDirection)
                thumbTIPRotationA = angle_between(thumbIPDirection, thumbTIPDirection)
                thumbRotationTIP = "{}, {}, {}, {}".format(round(thumbTIPRotationA,3), *thumbTIPRotationV)

                # index finger
                indexMCP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z,3))
                indexMCPDirection = unit_vector(np.asarray(diff(landmark[0], landmark[5])))
                indexMCPRotationV = np.cross(up, indexMCPDirection)
                indexMCPRotationA = angle_between(up, indexMCPDirection)
                indexRotationMCP = "{}, {}, {}, {}".format(round(indexMCPRotationA,3), *indexMCPRotationV)
                indexPIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z,3))
                indexPIPDirection = unit_vector(np.asarray(diff(landmark[5], landmark[6])))
                indexPIPRotationV = np.cross(indexMCPDirection, indexPIPDirection)
                indexPIPRotationA = angle_between(indexMCPDirection, indexPIPDirection)
                indexRotationPIP = "{}, {}, {}, {}".format(round(indexPIPRotationA,3), *indexPIPRotationV)
                indexDIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z,3))
                indexDIPDirection = unit_vector(np.asarray(diff(landmark[6], landmark[7])))
                indexDIPRotationV = np.cross(indexPIPDirection, indexDIPDirection)
                indexDIPRotationA = angle_between(indexPIPDirection, indexDIPDirection)
                indexRotationDIP = "{}, {}, {}, {}".format(round(indexDIPRotationA,3), *indexDIPRotationV)
                indexTIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,3))
                indexTIPDirection = unit_vector(np.asarray(diff(landmark[7], landmark[8])))
                indexTIPRotationV = np.cross(indexDIPDirection, indexTIPDirection)
                indexTIPRotationA = angle_between(indexDIPDirection, indexTIPDirection)
                indexRotationTIP = "{}, {}, {}, {}".format(round(indexTIPRotationA,3), *indexTIPRotationV)

                # middle finger

                middleMCP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z,3))
                middleMCPDirection = unit_vector(np.asarray(diff(landmark[0], landmark[9])))
                middleMCPRotationV = np.cross(up, middleMCPDirection)
                middleMCPRotationA = angle_between(up, middleMCPDirection)
                middleRotationMCP = "{}, {}, {}, {}".format(round(middleMCPRotationA,3), *middleMCPRotationV)

                middlePIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z,3))
                middlePIPDirection = unit_vector(np.asarray(diff(landmark[9], landmark[10])))
                middlePIPRotationV = np.cross(middleMCPDirection, middlePIPDirection)
                middlePIPRotationA = angle_between(middleMCPDirection, middlePIPDirection)
                middleRotationPIP = "{}, {}, {}, {}".format(round(middlePIPRotationA,3), *middlePIPRotationV)

                middleDIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z,3))
                middleDIPDirection = unit_vector(np.asarray(diff(landmark[10], landmark[11])))
                middleDIPRotationV = np.cross(middlePIPDirection, middleDIPDirection)
                middleDIPRotationA = angle_between(middlePIPDirection, middleDIPDirection)
                middleRotationDIP = "{}, {}, {}, {}".format(round(middleDIPRotationA,3), *middleDIPRotationV)

                middleTIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,3)) + ", " \
                           + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z,3))
                middleTIPDirection = unit_vector(np.asarray(diff(landmark[11], landmark[12])))
                middleTIPRotationV = np.cross(middleDIPDirection, middleTIPDirection)
                middleTIPRotationA = angle_between(middleDIPDirection, middleTIPDirection)
                middleRotationTIP = "{}, {}, {}, {}".format(round(middleTIPRotationA,3), *middleTIPRotationV)

                # ring finger

                ringMCP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z, 3))
                ringMCPDirection = unit_vector(np.asarray(diff(landmark[0], landmark[13])))
                ringMCPRotationV = np.cross(up, ringMCPDirection)
                ringMCPRotationA = angle_between(up, ringMCPDirection)
                ringRotationMCP = "{}, {}, {}, {}".format(round(ringMCPRotationA,3), *ringMCPRotationV)

                ringPIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z, 3))
                ringPIPDirection = unit_vector(np.asarray(diff(landmark[13], landmark[14])))
                ringPIPRotationV = np.cross(ringMCPDirection, ringPIPDirection)
                ringPIPRotationA = angle_between(ringMCPDirection, ringPIPDirection)
                ringRotationPIP = "{}, {}, {}, {}".format(round(ringPIPRotationA,3), *ringPIPRotationV)

                ringDIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z, 3))
                ringDIPDirection = unit_vector(np.asarray(diff(landmark[14], landmark[15])))
                ringDIPRotationV = np.cross(ringPIPDirection, ringDIPDirection)
                ringDIPRotationA = angle_between(ringPIPDirection, ringDIPDirection)
                ringRotationDIP = "{}, {}, {}, {}".format(round(ringDIPRotationA,3), *ringDIPRotationV)

                ringTIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z, 3))
                ringTIPDirection = unit_vector(np.asarray(diff(landmark[15], landmark[16])))
                ringTIPRotationV = np.cross(ringDIPDirection, ringTIPDirection)
                ringTIPRotationA = angle_between(ringDIPDirection, ringTIPDirection)
                ringRotationTIP = "{}, {}, {}, {}".format(round(ringTIPRotationA,3), *ringTIPRotationV)

                # pinky

                pinkyMCP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z, 3))
                pinkyMCPDirection = unit_vector(np.asarray(diff(landmark[0], landmark[17])))
                pinkyMCPRotationV = np.cross(up, pinkyMCPDirection)
                pinkyMCPRotationA = angle_between(up, pinkyMCPDirection)
                pinkyRotationMCP = "{}, {}, {}, {}".format(round(pinkyMCPRotationA,3), *pinkyMCPRotationV)

                pinkyPIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z, 3))
                pinkyPIPDirection = unit_vector(np.asarray(diff(landmark[17], landmark[18])))
                pinkyPIPRotationV = np.cross(pinkyMCPDirection, pinkyPIPDirection)
                pinkyPIPRotationA = angle_between(pinkyMCPDirection, pinkyPIPDirection)
                pinkyRotationPIP = "{}, {}, {}, {}".format(round(pinkyPIPRotationA,3), *pinkyPIPRotationV)

                pinkyDIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z, 3))
                pinkyDIPDirection = unit_vector(np.asarray(diff(landmark[18], landmark[19])))
                pinkyDIPRotationV = np.cross(pinkyPIPDirection, pinkyDIPDirection)
                pinkyDIPRotationA = angle_between(pinkyPIPDirection, pinkyDIPDirection)
                pinkyRotationDIP = "{}, {}, {}, {}".format(round(pinkyDIPRotationA,3), *pinkyDIPRotationV)

                pinkyTIP = str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y, 3)) + ", " \
                          + str(round(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z, 3))
                pinkyTIPDirection = unit_vector(np.asarray(diff(landmark[19], landmark[20])))
                pinkyTIPRotationV = np.cross(pinkyDIPDirection, pinkyTIPDirection)
                pinkyTIPRotationA = angle_between(pinkyDIPDirection, pinkyTIPDirection)
                pinkyRotationTIP = "{}, {}, {}, {}".format(round(pinkyTIPRotationA, 3), *pinkyTIPRotationV)

                # "TC" + thumbCMC + "TC" + \
                # "TCR" + thumbRotationCMC + "TCR" + \
                # "TM" + thumbMCP + "TM" + \
                # "TMR" + thumbRotationMCP + "TMR" + \
                # "TI" + thumbIP + "TI" + \
                # "TIR" + thumbRotationIP + "TIR" + \
                # "TT" + thumbTIP + "TT" + \
                # "TTR" + thumbRotationTIP + "TTR" + \
                # "IM" + indexMCP + "IM" + \
                # "IRM" + indexRotationMCP + "IRM" + \
                # "IP" + indexPIP + "IP" + \
                # "IPR" + indexRotationPIP + "IPR" + \
                # "ID" + indexDIP + "ID" + \
                # "IDR" + indexRotationDIP + "IDR" + \
                # "IT" + indexTIP + "IT" + \
                # "ITR" + indexRotationTIP + "ITR" + \
                # "MM" + middleMCP + "MM" + \
                # "MRM" + middleRotationMCP + "MRM" + \
                # "MP" + middlePIP + "MP" + \
                # "MPR" + middleRotationPIP + "MPR" + \
                # "MD" + middleDIP + "MD" + \
                # "MDR" + middleRotationDIP + "MDR" + \
                # "MT" + middleTIP + "MT" + \
                # "MRT" + middleRotationTIP + "MRT" + \
                # "RM" + ringMCP + "RM" + \
                # "RRM" + ringRotationMCP + "RRM" + \
                # "RP" + ringPIP + "RP" + \
                # "RRP" + ringRotationPIP + "RRP" + \
                # "RD" + ringDIP + "RD" + \
                # "RDR" + ringRotationDIP + "RDR" + \
                # "RT" + ringTIP + "RT" + \
                # "RTR" + ringRotationTIP + "RTR" + \
                # "PM" + pinkyMCP + "PM" + \
                # "PMR" + pinkyRotationMCP + "PMR" + \
                # "PP" + pinkyPIP + "PP" + \
                # "PPR" + pinkyRotationPIP + "PPR" + \
                # "PD" + pinkyDIP + "PD" + \
                # "PDR" + pinkyRotationDIP + "PDR" + \
                # "PT" + pinkyTIP + "PT" + \
                # "PTR" + pinkyRotationTIP + "PTR"

                message = \
                    "W" + wrist + "W" + \
                    "WR" + wristRotation + "WR" + \
                    "thumb" + thumbCMC + "x" + thumbRotationCMC + "x" + thumbMCP + "x" + thumbRotationMCP + "x" + \
                    thumbIP + "x" + thumbRotationIP + "x" + thumbTIP + "x" + thumbRotationTIP + "thumb" + \
                    "index" + indexMCP + "x" + indexRotationMCP + "x" + indexPIP + "x" + indexRotationPIP + "x" + \
                    indexDIP + "x" + indexRotationDIP + "x" + indexTIP + "x" + indexRotationTIP + "index" + \
                    "middle" + middleMCP + "x" + middleRotationMCP + "x" + middlePIP + "x" + middleRotationPIP + "x" + \
                    middleDIP + "x" + middleRotationDIP + "x" + middleTIP + "x" + middleRotationTIP + "middle" + \
                    "ring" + ringMCP + "x" + ringRotationMCP + "x" + ringPIP + "x" + ringRotationPIP + "x" + \
                    ringDIP + "x" + ringRotationDIP + "x" + ringTIP + "x" + ringRotationTIP + "ring" + \
                    "pinky" + pinkyMCP + "x" + pinkyRotationMCP + "x" + pinkyPIP + "x" + pinkyRotationPIP + "x" + \
                    pinkyDIP + "x" + pinkyRotationDIP + "x" + pinkyTIP + "x" + pinkyRotationTIP + "pinky"

                udp_send(message.encode())
                print(message)
        time.sleep(0.1)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
