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


def vector_to_angle_axis(v):
    """
    Calculate the rotation according to the x and z axis transform y to v
    :param v: target vector
    :return: angle, axis
    """
    y = np.asarray([0., 1., 0.])
    angle = angle_between(v, y)
    axis = np.cross(y, v)
    return [angle, axis[0], axis[1], axis[2]]


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
        yP = np.matmul(y, inv(leftHandCS))
        transDict["LeftHandIndex1"] = vector_to_angle_axis(yP)
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
        yP = np.matmul(y, inv(leftHandCS))
        transDict["LeftHandMiddle1"] = vector_to_angle_axis(yP)
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
        yP = np.matmul(y, inv(leftHandCS))
        transDict["LeftHandRing1"] = vector_to_angle_axis(yP)
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
        yP = np.matmul(y, inv(leftHandCS))
        transDict["LeftHandPinky1"] = vector_to_angle_axis(yP)
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
        yP = np.matmul(y, inv(leftHandCS))
        transDict["LeftHandThumb1"] = vector_to_angle_axis(yP)
        angle_axis = transDict["LeftHandThumb1"]
        r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
        leftThumb1CS = np.matmul(r.rotation_matrix.transpose(), leftHandCS)
        # Thumb2
        angle = angle_between(leftHandLandmarkList[3] - leftHandLandmarkList[2],
                              leftHandLandmarkList[2] - leftHandLandmarkList[1])
        y = normalize(leftHandLandmarkList[3] - leftHandLandmarkList[2])
        yP = np.matmul(y, inv(leftThumb1CS))
        if yP[0] < 0:
            transDict["LeftHandThumb2"] = [angle, 0.0, 0.0, 1.0]
        else:
            transDict["LeftHandThumb2"] = [-angle, 0.0, 0.0, 1.0]
        angle_axis = transDict["LeftHandThumb2"]
        r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
        leftThumb2CS = np.matmul(r.rotation_matrix.transpose(), leftThumb1CS)
        # Thumb3
        angle = angle_between(leftHandLandmarkList[4] - leftHandLandmarkList[3],
                              leftHandLandmarkList[3] - leftHandLandmarkList[2])
        y = normalize(leftHandLandmarkList[4] - leftHandLandmarkList[3])
        yP = np.matmul(y, inv(leftThumb2CS))
        if yP[0] < 0:
            transDict["LeftHandThumb3"] = [angle, 0.0, 0.0, 1.0]
        else:
            transDict["LeftHandThumb3"] = [-angle, 0.0, 0.0, 1.0]

        # # Thumb1
        # y = normalize(leftHandLandmarkList[2] - leftHandLandmarkList[1])
        # yP = np.matmul(y, inv(leftHandCS))
        # transDict["LeftHandThumb1"] = vector_to_angle_axis(yP)
        # # Thumb2
        # angle = angle_between(leftHandLandmarkList[3] - leftHandLandmarkList[2],
        #                       leftHandLandmarkList[2] - leftHandLandmarkList[1])
        # transDict["LeftHandThumb2"] = [angle, 0.0, 0.0, -1.0]
        # # Thumb3
        # angle = angle_between(leftHandLandmarkList[4] - leftHandLandmarkList[3],
        #                       leftHandLandmarkList[3] - leftHandLandmarkList[2])
        # transDict["LeftHandThumb3"] = [angle, 0.0, 0.0, -1.0]

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
        yP = np.matmul(y, inv(rightHandCS))
        transDict["RightHandIndex1"] = vector_to_angle_axis(yP)
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
        yP = np.matmul(y, inv(rightHandCS))
        transDict["RightHandMiddle1"] = vector_to_angle_axis(yP)
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
        yP = np.matmul(y, inv(rightHandCS))
        transDict["RightHandRing1"] = vector_to_angle_axis(yP)
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
        yP = np.matmul(y, inv(rightHandCS))
        transDict["RightHandPinky1"] = vector_to_angle_axis(yP)
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
        yP = np.matmul(y, inv(rightHandCS))
        transDict["RightHandThumb1"] = vector_to_angle_axis(yP)
        angle_axis = transDict["RightHandThumb1"]
        r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
        rightThumb1CS = np.matmul(r.rotation_matrix.transpose(), rightHandCS)
        # Thumb2
        angle = angle_between(rightHandLandmarkList[3] - rightHandLandmarkList[2],
                              rightHandLandmarkList[2] - rightHandLandmarkList[1])
        y = normalize(rightHandLandmarkList[3] - rightHandLandmarkList[2])
        yP = np.matmul(y, inv(rightThumb1CS))
        if yP[0] < 0:
            transDict["RightHandThumb2"] = [angle, 0.0, 0.0, 1.0]
        else:
            transDict["RightHandThumb2"] = [-angle, 0.0, 0.0, 1.0]
        angle_axis = transDict["RightHandThumb2"]
        r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
        rightThumb2CS = np.matmul(r.rotation_matrix.transpose(), rightThumb1CS)
        # Thumb3
        angle = angle_between(rightHandLandmarkList[4] - rightHandLandmarkList[3],
                              rightHandLandmarkList[3] - rightHandLandmarkList[2])
        y = normalize(rightHandLandmarkList[4] - rightHandLandmarkList[3])
        yP = np.matmul(y, inv(rightThumb2CS))
        if yP[0] < 0:
            transDict["RightHandThumb3"] = [angle, 0.0, 0.0, 1.0]
        else:
            transDict["RightHandThumb3"] = [-angle, 0.0, 0.0, 1.0]
    return transDict


def angle_axis_to_string(angle_axis):
    return "{:.3f},{:.5f},{:.5f},{:.5f}".format(*angle_axis)
