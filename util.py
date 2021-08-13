import numpy as np
from numpy.linalg import inv
from pyquaternion import Quaternion

# pip install pyquaternion
LEG_LENGTH = 0


def normalize(v):
    return v / np.linalg.norm(v)


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


def to_angle_axis(rotationM):
    q = Quaternion(matrix=rotationM.transpose())
    axis = q.axis
    return [q.degrees, axis[0], axis[1], axis[2]]


def middle(v1, v2):
    return (v1 + v2) / 2.


def to_trans_dict(pose_landmark, left_hand_landmarks, right_hand_landmarks):
    global LEG_LENGTH
    visibility = np.asarray([l.visibility for l in pose_landmark])
    landmarkList = np.asarray([[l.x, -l.y, -l.z] for l in pose_landmark])
    transDict = {}
    # Spine CS
    orig = ((landmarkList[11] + landmarkList[12]) / 2.)
    x = normalize(landmarkList[11] - landmarkList[12])

    up = (landmarkList[11] + landmarkList[12]) / 2. - (landmarkList[23] + landmarkList[24]) / 2.
    up = normalize(up)

    z = normalize(np.cross(x, up))
    y = np.cross(z, x)

    # spineCS = np.append(np.transpose(np.asarray([x, y, z, orig])), [[0,0,0,1]], axis=0)
    spineCS = np.asarray([x, y, z])
    spineRotation = spineCS

    # Left shoulder
    leftShoulderRotation = np.asarray([[0, 0, -1],
                                       [1, 0, 0],
                                       [0, -1, 0]])
    leftShoulderCS = np.matmul(leftShoulderRotation, spineCS)

    # Left arm
    # leftArmCS = leftArmRotation * leftShouldCS
    y = normalize(landmarkList[13] - landmarkList[11])
    z = normalize(np.cross(y, landmarkList[15] - landmarkList[13]))
    x = np.cross(y, z)
    leftArmCS = np.asarray([x, y, z])
    leftArmRotation = np.matmul(leftArmCS, inv(leftShoulderCS))

    # leftForeArm
    y = normalize(landmarkList[15] - landmarkList[13])
    # z does not change here
    x = np.cross(y, z)
    leftForeArmCS = np.asarray([x, y, z])
    leftForeArmRotation = np.matmul(leftForeArmCS, inv(leftArmCS))

    transDict["LeftArm"] = to_angle_axis(leftArmRotation)
    transDict["LeftForeArm"] = to_angle_axis(leftForeArmRotation)
    # Right shoulder
    rightShoulderRotation = np.asarray([[0, 0, 1],
                                        [-1, 0, 0],
                                        [0, -1, 0]])
    rightShoulderCS = np.matmul(rightShoulderRotation, spineCS)

    # Right arm
    # rightArmCS = rightArmRotation * rightShouldCS
    y = normalize(landmarkList[14] - landmarkList[12])
    z = normalize(np.cross(landmarkList[16] - landmarkList[14], y))
    x = np.cross(y, z)
    rightArmCS = np.asarray([x, y, z])
    rightArmRotation = np.matmul(rightArmCS, inv(rightShoulderCS))

    # rightForeArm
    y = normalize(landmarkList[16] - landmarkList[14])
    # z does not change here
    x = np.cross(y, z)
    rightForeArmCS = np.asarray([x, y, z])
    rightForeArmRotation = np.matmul(rightForeArmCS, inv(rightArmCS))

    transDict["RightArm"] = to_angle_axis(rightArmRotation)
    transDict["RightForeArm"] = to_angle_axis(rightForeArmRotation)

    if visibility[25] > .5 and visibility[26] > .5:
        # Left up leg
        y = normalize(landmarkList[25] - landmarkList[23])
        x = normalize(np.cross(landmarkList[27] - landmarkList[25], y))
        z = np.cross(x, y)
        leftUpLegCS = np.asarray([x, y, z])
        leftUpLegRotation = np.matmul(leftUpLegCS, inv(spineCS))

        # Left leg
        y = normalize(landmarkList[27] - landmarkList[25])
        # x does not change here
        z = np.cross(x, y)
        leftLegCS = np.asarray([x, y, z])
        leftLegRotation = np.matmul(leftLegCS, inv(leftUpLegCS))

        transDict["LeftUpLeg"] = to_angle_axis(leftUpLegRotation)
        transDict["LeftLeg"] = to_angle_axis(leftLegRotation)

        # Right up leg
        y = normalize(landmarkList[26] - landmarkList[24])
        x = normalize(np.cross(landmarkList[28] - landmarkList[26], y))
        z = np.cross(x, y)
        rightUpLegCS = np.asarray([x, y, z])
        rightUpLegRotation = np.matmul(rightUpLegCS, inv(spineCS))

        # right leg
        y = normalize(landmarkList[28] - landmarkList[26])
        # x does not change here
        z = np.cross(x, y)
        rightLegCS = np.asarray([x, y, z])
        rightLegRotation = np.matmul(rightLegCS, inv(rightUpLegCS))

        transDict["RightUpLeg"] = to_angle_axis(rightUpLegRotation)
        transDict["RightLeg"] = to_angle_axis(rightLegRotation)

        # Distance to the ground
        # Translation (ratio) = (hip_point - lower_point)/ (2. * up_leg_length)
        if LEG_LENGTH == 0:
            LEG_LENGTH = 2. * np.linalg.norm(landmarkList[26] - landmarkList[24])

        translation = ((landmarkList[23] + landmarkList[24]) / 2)[1] - np.min(landmarkList[:, 1]) / LEG_LENGTH

        transDict["Hips"] = to_angle_axis(spineRotation)
        transDict["HipsTrans"] = [translation, 0, 0, 0]
    else:
        transDict["Spine1"] = to_angle_axis(spineRotation)

    # left hand
    if visibility[19] > 0.5:
        handMiddle = (landmarkList[17] + landmarkList[19]) / 2.
        y = normalize(handMiddle - landmarkList[15])
        z = normalize(np.cross(landmarkList[17] - landmarkList[15], landmarkList[19] - landmarkList[15]))
        x = np.cross(y, z)
        leftHandCS = np.asarray([x, y, z])
        leftHandRotation = np.matmul(leftHandCS, inv(leftForeArmCS))
        transDict["LeftHand"] = to_angle_axis(leftHandRotation)

    # right hand
    if visibility[20] > 0.5:
        handMiddle = (landmarkList[18] + landmarkList[20]) / 2.
        y = normalize(handMiddle - landmarkList[16])
        z = normalize(np.cross(landmarkList[20] - landmarkList[16], landmarkList[18] - landmarkList[16]))
        x = np.cross(y, z)
        rightHandCS = np.asarray([x, y, z])
        rightHandRotation = np.matmul(rightHandCS, inv(rightForeArmCS))
        transDict["RightHand"] = to_angle_axis(rightHandRotation)

    # head
    if visibility[0] > 0.5:
        neck = middle(landmarkList[11], landmarkList[12])
        y = normalize(middle(landmarkList[7], landmarkList[8]) - neck)
        z = normalize(np.cross(landmarkList[7] - neck, landmarkList[8] - neck))
        x = np.cross(y, z)
        headCS = np.asarray([x, y, z])
        headRotation = np.matmul(headCS, inv(spineCS))
        transDict["Head"] = to_angle_axis(headRotation)

    # If the hand landmarks are given, we optimize the gestures with the hand landmarks
    # Left hand
    if left_hand_landmarks:
        leftHandVisibility = np.asarray([l.visibility for l in left_hand_landmarks.landmark])
        leftHandLandmarkList = np.asarray([[l.x, -l.y, -l.z] for l in left_hand_landmarks.landmark])
        handMiddle = (leftHandLandmarkList[5] + leftHandLandmarkList[17]) / 2.
        y = normalize(handMiddle - leftHandLandmarkList[0])
        z = normalize(np.cross(leftHandLandmarkList[17] - leftHandLandmarkList[0],
                               leftHandLandmarkList[5] - leftHandLandmarkList[0]))
        x = np.cross(y, z)
        leftHandCS = np.asarray([x, y, z])
        leftHandRotation = np.matmul(leftHandCS, inv(leftForeArmCS))
        transDict["LeftHand"] = to_angle_axis(leftHandRotation)

        # Left hand fingers
        # Index1
        y = normalize(leftHandLandmarkList[6] - leftHandLandmarkList[5])
        x = normalize(np.cross(y, leftHandLandmarkList[7] - leftHandLandmarkList[6]))
        z = np.cross(x, y)
        leftHandIndex1CS = np.asarray([x, y, z])
        leftHandIndex1Rotation = np.matmul(leftHandIndex1CS, inv(leftHandCS))
        transDict["LeftHandIndex1"] = to_angle_axis(leftHandIndex1Rotation)
        # Index2. Since the knuckles only rotate according to the x axis, we use the angle between function.
        angle = angle_between(leftHandLandmarkList[7] - leftHandLandmarkList[6],
                              leftHandLandmarkList[6] - leftHandLandmarkList[5])
        transDict["LeftHandIndex2"] = [angle, 1.0, 0.0, 0.0]
        # Index3
        angle = angle_between(leftHandLandmarkList[8] - leftHandLandmarkList[7],
                              leftHandLandmarkList[7] - leftHandLandmarkList[6])
        transDict["LeftHandIndex3"] = [angle, 1.0, 0.0, 0.0]

        # Middle1
        y = normalize(leftHandLandmarkList[10] - leftHandLandmarkList[9])
        x = normalize(np.cross(y, leftHandLandmarkList[11] - leftHandLandmarkList[10]))
        z = np.cross(x, y)
        leftHandMiddle1CS = np.asarray([x, y, z])
        leftHandMiddle1Rotation = np.matmul(leftHandMiddle1CS, inv(leftHandCS))
        transDict["LeftHandMiddle1"] = to_angle_axis(leftHandMiddle1Rotation)
        # Middle2
        angle = angle_between(leftHandLandmarkList[11] - leftHandLandmarkList[10],
                              leftHandLandmarkList[10] - leftHandLandmarkList[9])
        transDict["LeftHandMiddle2"] = [angle, 1.0, 0.0, 0.0]
        # Middle3
        angle = angle_between(leftHandLandmarkList[12] - leftHandLandmarkList[11],
                              leftHandLandmarkList[11] - leftHandLandmarkList[10])
        transDict["LeftHandMiddle3"] = [angle, 1.0, 0.0, 0.0]

        # Ring1
        y = normalize(leftHandLandmarkList[14] - leftHandLandmarkList[13])
        x = normalize(np.cross(y, leftHandLandmarkList[15] - leftHandLandmarkList[14]))
        z = np.cross(x, y)
        leftHandRing1CS = np.asarray([x, y, z])
        leftHandRing1Rotation = np.matmul(leftHandRing1CS, inv(leftHandCS))
        transDict["LeftHandRing1"] = to_angle_axis(leftHandRing1Rotation)
        # Ring2
        angle = angle_between(leftHandLandmarkList[15] - leftHandLandmarkList[14],
                              leftHandLandmarkList[14] - leftHandLandmarkList[13])
        transDict["LeftHandRing2"] = [angle, 1.0, 0.0, 0.0]
        # Ring3
        angle = angle_between(leftHandLandmarkList[16] - leftHandLandmarkList[15],
                              leftHandLandmarkList[15] - leftHandLandmarkList[14])
        transDict["LeftHandRing3"] = [angle, 1.0, 0.0, 0.0]

        # Pinky1
        y = normalize(leftHandLandmarkList[18] - leftHandLandmarkList[17])
        x = normalize(np.cross(y, leftHandLandmarkList[19] - leftHandLandmarkList[18]))
        z = np.cross(x, y)
        leftHandPinky1CS = np.asarray([x, y, z])
        leftHandPinky1Rotation = np.matmul(leftHandPinky1CS, inv(leftHandCS))
        transDict["LeftHandPinky1"] = to_angle_axis(leftHandPinky1Rotation)
        # Pinky2
        angle = angle_between(leftHandLandmarkList[19] - leftHandLandmarkList[18],
                              leftHandLandmarkList[18] - leftHandLandmarkList[17])
        transDict["LeftHandPinky2"] = [angle, 1.0, 0.0, 0.0]
        # Pinky3
        angle = angle_between(leftHandLandmarkList[20] - leftHandLandmarkList[19],
                              leftHandLandmarkList[19] - leftHandLandmarkList[18])
        transDict["LeftHandPinky3"] = [angle, 1.0, 0.0, 0.0]

        # Thumb1
        y = normalize(leftHandLandmarkList[2] - leftHandLandmarkList[1])
        z = normalize(np.cross(leftHandLandmarkList[3] - leftHandLandmarkList[2], y))
        x = np.cross(y, z)
        leftHandThumb1CS = np.asarray([x, y, z])
        leftHandThumb1Rotation = np.matmul(leftHandThumb1CS, inv(leftHandCS))
        transDict["LeftHandThumb1"] = to_angle_axis(leftHandThumb1Rotation)
        # Thumb2
        angle = angle_between(leftHandLandmarkList[3] - leftHandLandmarkList[2],
                              leftHandLandmarkList[2] - leftHandLandmarkList[1])
        transDict["LeftHandThumb2"] = [angle, 0.0, 0.0, -1.0]
        # Thumb3
        angle = angle_between(leftHandLandmarkList[4] - leftHandLandmarkList[3],
                              leftHandLandmarkList[3] - leftHandLandmarkList[2])
        transDict["LeftHandThumb3"] = [angle, 0.0, 0.0, -1.0]

    # Right hand
    if right_hand_landmarks:
        rightHandVisibility = np.asarray([l.visibility for l in right_hand_landmarks.landmark])
        rightHandLandmarkList = np.asarray([[l.x, -l.y, -l.z] for l in right_hand_landmarks.landmark])
        handMiddle = (rightHandLandmarkList[5] + rightHandLandmarkList[17]) / 2.
        y = normalize(handMiddle - rightHandLandmarkList[0])
        z = normalize(np.cross(rightHandLandmarkList[5] - rightHandLandmarkList[0],
                               rightHandLandmarkList[17] - rightHandLandmarkList[0]))
        x = np.cross(y, z)
        rightHandCS = np.asarray([x, y, z])
        rightHandRotation = np.matmul(rightHandCS, inv(rightForeArmCS))
        transDict["RightHand"] = to_angle_axis(rightHandRotation)

        # Right hand fingers
        # Index1
        y = normalize(rightHandLandmarkList[6] - rightHandLandmarkList[5])
        x = normalize(np.cross(y, rightHandLandmarkList[7] - rightHandLandmarkList[6]))
        z = np.cross(x, y)
        rightHandIndex1CS = np.asarray([x, y, z])
        rightHandIndex1Rotation = np.matmul(rightHandIndex1CS, inv(rightHandCS))
        transDict["RightHandIndex1"] = to_angle_axis(rightHandIndex1Rotation)
        # Index2. Since the knuckles only rotate according to the x axis, we use the angle between function.
        angle = angle_between(rightHandLandmarkList[7] - rightHandLandmarkList[6],
                              rightHandLandmarkList[6] - rightHandLandmarkList[5])
        transDict["RightHandIndex2"] = [angle, 1.0, 0.0, 0.0]
        # Index3
        angle = angle_between(rightHandLandmarkList[8] - rightHandLandmarkList[7],
                              rightHandLandmarkList[7] - rightHandLandmarkList[6])
        transDict["RightHandIndex3"] = [angle, 1.0, 0.0, 0.0]

        # Middle1
        y = normalize(rightHandLandmarkList[10] - rightHandLandmarkList[9])
        x = normalize(np.cross(y, rightHandLandmarkList[11] - rightHandLandmarkList[10]))
        z = np.cross(x, y)
        rightHandMiddle1CS = np.asarray([x, y, z])
        rightHandMiddle1Rotation = np.matmul(rightHandMiddle1CS, inv(rightHandCS))
        transDict["RightHandMiddle1"] = to_angle_axis(rightHandMiddle1Rotation)
        # Middle2
        angle = angle_between(rightHandLandmarkList[11] - rightHandLandmarkList[10],
                              rightHandLandmarkList[10] - rightHandLandmarkList[9])
        transDict["RightHandMiddle2"] = [angle, 1.0, 0.0, 0.0]
        # Middle3
        angle = angle_between(rightHandLandmarkList[12] - rightHandLandmarkList[11],
                              rightHandLandmarkList[11] - rightHandLandmarkList[10])
        transDict["RightHandMiddle3"] = [angle, 1.0, 0.0, 0.0]

        # Ring1
        y = normalize(rightHandLandmarkList[14] - rightHandLandmarkList[13])
        x = normalize(np.cross(y, rightHandLandmarkList[15] - rightHandLandmarkList[14]))
        z = np.cross(x, y)
        rightHandRing1CS = np.asarray([x, y, z])
        rightHandRing1Rotation = np.matmul(rightHandRing1CS, inv(rightHandCS))
        transDict["RightHandRing1"] = to_angle_axis(rightHandRing1Rotation)
        # Ring2
        angle = angle_between(rightHandLandmarkList[15] - rightHandLandmarkList[14],
                              rightHandLandmarkList[14] - rightHandLandmarkList[13])
        transDict["RightHandRing2"] = [angle, 1.0, 0.0, 0.0]
        # Ring3
        angle = angle_between(rightHandLandmarkList[16] - rightHandLandmarkList[15],
                              rightHandLandmarkList[15] - rightHandLandmarkList[14])
        transDict["RightHandRing3"] = [angle, 1.0, 0.0, 0.0]

        # Pinky1
        y = normalize(rightHandLandmarkList[18] - rightHandLandmarkList[17])
        x = normalize(np.cross(y, rightHandLandmarkList[19] - rightHandLandmarkList[18]))
        z = np.cross(x, y)
        rightHandPinky1CS = np.asarray([x, y, z])
        rightHandPinky1Rotation = np.matmul(rightHandPinky1CS, inv(rightHandCS))
        transDict["RightHandPinky1"] = to_angle_axis(rightHandPinky1Rotation)
        # Pinky2
        angle = angle_between(rightHandLandmarkList[19] - rightHandLandmarkList[18],
                              rightHandLandmarkList[18] - rightHandLandmarkList[17])
        transDict["RightHandPinky2"] = [angle, 1.0, 0.0, 0.0]
        # Pinky3
        angle = angle_between(rightHandLandmarkList[20] - rightHandLandmarkList[19],
                              rightHandLandmarkList[19] - rightHandLandmarkList[18])
        transDict["RightHandPinky3"] = [angle, 1.0, 0.0, 0.0]

        # Thumb1
        y = normalize(rightHandLandmarkList[2] - rightHandLandmarkList[1])
        z = normalize(np.cross(y, rightHandLandmarkList[3] - rightHandLandmarkList[2]))
        x = np.cross(y, z)
        rightHandThumb1CS = np.asarray([x, y, z])
        rightHandThumb1Rotation = np.matmul(rightHandThumb1CS, inv(rightHandCS))
        transDict["RightHandThumb1"] = to_angle_axis(rightHandThumb1Rotation)
        # Thumb2
        angle = angle_between(rightHandLandmarkList[3] - rightHandLandmarkList[2],
                              rightHandLandmarkList[2] - rightHandLandmarkList[1])
        transDict["RightHandThumb2"] = [angle, 0.0, 0.0, 1.0]
        # Thumb3
        angle = angle_between(rightHandLandmarkList[4] - rightHandLandmarkList[3],
                              rightHandLandmarkList[3] - rightHandLandmarkList[2])
        transDict["RightHandThumb3"] = [angle, 0.0, 0.0, 1.0]
    return transDict


def angle_axis_to_string(angle_axis):
    return "{:.3f},{:.5f},{:.5f},{:.5f}".format(*angle_axis)
