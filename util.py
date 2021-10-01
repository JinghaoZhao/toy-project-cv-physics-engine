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


def to_angle_axis(rotation_m):
    """
    Transform the rotation matrix to degrees and axis
    :param rotation_m: rotation matrix
    :return: [angle, x, y, z]
    """
    q = Quaternion(matrix=rotation_m.transpose())
    axis = q.axis
    return [q.degrees, axis[0], axis[1], axis[2]]


def middle(v1, v2):
    return (v1 + v2) / 2.


def vector_to_angle_axis(v):
    """
    Calculate the rotation according to the x and z axis transform y to v
    :param v: target vector
    :return: [angle, x, y, z]
    """
    y = np.asarray([0., 1., 0.])
    angle = angle_between(v, y)
    axis = np.cross(y, v)
    return [angle, axis[0], axis[1], axis[2]]


def angle_axis_to_matrix(vec4):
    rotation = Quaternion(axis=[vec4[1], vec4[2], vec4[3]], degrees=vec4[0])
    return rotation.rotation_matrix


def to_trans_dict(pose_landmark, left_hand_landmarks, right_hand_landmarks):
    """
    An IK module to transform the pose landmarks to kinematic transformations
    :param pose_landmark: 33 pose landmarks from Mediapipe
    :param left_hand_landmarks: 21 hand landmarks from Mediapipe
    :param right_hand_landmarks: 21 hand landmarks from Mediapipe
    :return: the transformation dictionary used in the AR application
    """
    global LEG_LENGTH
    visibility = np.asarray([lmk.visibility for lmk in pose_landmark])
    landmark_list = np.asarray([[lmk.x, -lmk.y, -lmk.z] for lmk in pose_landmark])
    trans_dict = {}
    # Spine CS
    # orig = ((landmark_list[11] + landmark_list[12]) / 2.)
    x = normalize(landmark_list[11] - landmark_list[12])

    up = (landmark_list[11] + landmark_list[12]) / 2. - (landmark_list[23] + landmark_list[24]) / 2.
    up = normalize(up)

    z = normalize(np.cross(x, up))
    y = np.cross(z, x)

    # spineCS = np.append(np.transpose(np.asarray([x, y, z, orig])), [[0,0,0,1]], axis=0)
    spine_cs = np.asarray([x, y, z])
    spine_rotation = spine_cs

    # Left shoulder
    left_shoulder_rotation = np.asarray([[0, 0, -1],
                                         [1, 0, 0],
                                         [0, -1, 0]])
    left_shoulder_cs = np.matmul(left_shoulder_rotation, spine_cs)

    # Left arm has 2 DoF, rotation according to x and z. It cannot rotate according to Y

    # leftArmCS = leftArmRotation * leftShouldCS
    # y in the shoulder CS
    y = normalize(landmark_list[13] - landmark_list[11])
    y_p = np.matmul(y, inv(left_shoulder_cs))
    trans_dict["LeftArm"] = vector_to_angle_axis(y_p)
    left_arm_rotation = angle_axis_to_matrix(trans_dict["LeftArm"])
    left_arm_cs = np.matmul(left_arm_rotation.T, left_shoulder_cs)

    # left fore Arm CS also have 2 DoF
    y = normalize(landmark_list[15] - landmark_list[13])
    y_p = np.matmul(y, inv(left_arm_cs))
    trans_dict["LeftForeArm"] = vector_to_angle_axis(y_p)
    left_fore_arm_rotation = angle_axis_to_matrix(trans_dict["LeftForeArm"])
    left_fore_arm_cs = np.matmul(left_fore_arm_rotation.T, left_arm_cs)

    # Right shoulder
    right_shoulder_rotation = np.asarray([[0, 0, 1],
                                          [-1, 0, 0],
                                          [0, -1, 0]])
    right_shoulder_cs = np.matmul(right_shoulder_rotation, spine_cs)

    # Right arm
    # Right arm has 2 DoF, rotation according to x and z. It cannot rotate according to Y
    # rightArmCS = rightArmRotation * rightShouldCS
    # y in the shoulder CS
    y = normalize(landmark_list[14] - landmark_list[12])
    y_p = np.matmul(y, inv(right_shoulder_cs))
    trans_dict["RightArm"] = vector_to_angle_axis(y_p)
    right_arm_rotation = angle_axis_to_matrix(trans_dict["RightArm"])
    right_arm_cs = np.matmul(right_arm_rotation.T, right_shoulder_cs)

    # right fore Arm CS also have 2 DoF
    y = normalize(landmark_list[16] - landmark_list[14])
    y_p = np.matmul(y, inv(right_arm_cs))
    trans_dict["RightForeArm"] = vector_to_angle_axis(y_p)
    right_fore_arm_rotation = angle_axis_to_matrix(trans_dict["RightForeArm"])
    right_fore_arm_cs = np.matmul(right_fore_arm_rotation.T, right_arm_cs)

    if visibility[25] > .5 and visibility[26] > .5:
        # Left up leg
        y = normalize(landmark_list[25] - landmark_list[23])
        x = normalize(np.cross(landmark_list[27] - landmark_list[25], y))
        z = np.cross(x, y)
        left_up_leg_cs = np.asarray([x, y, z])
        left_up_leg_rotation = np.matmul(left_up_leg_cs, inv(spine_cs))

        # Left leg
        y = normalize(landmark_list[27] - landmark_list[25])
        # x does not change here
        z = np.cross(x, y)
        left_leg_cs = np.asarray([x, y, z])
        left_leg_rotation = np.matmul(left_leg_cs, inv(left_up_leg_cs))

        trans_dict["LeftUpLeg"] = to_angle_axis(left_up_leg_rotation)
        trans_dict["LeftLeg"] = to_angle_axis(left_leg_rotation)

        # Right up leg
        y = normalize(landmark_list[26] - landmark_list[24])
        x = normalize(np.cross(landmark_list[28] - landmark_list[26], y))
        z = np.cross(x, y)
        right_up_leg_cs = np.asarray([x, y, z])
        right_up_leg_rotation = np.matmul(right_up_leg_cs, inv(spine_cs))

        # right leg
        y = normalize(landmark_list[28] - landmark_list[26])
        # x does not change here
        z = np.cross(x, y)
        right_leg_cs = np.asarray([x, y, z])
        right_leg_rotation = np.matmul(right_leg_cs, inv(right_up_leg_cs))

        trans_dict["RightUpLeg"] = to_angle_axis(right_up_leg_rotation)
        trans_dict["RightLeg"] = to_angle_axis(right_leg_rotation)

        # Distance to the ground
        # Translation (ratio) = (hip_point - lower_point)/ (2. * up_leg_length)
        if LEG_LENGTH == 0:
            LEG_LENGTH = 2. * np.linalg.norm(landmark_list[26] - landmark_list[24])

        translation = ((landmark_list[23] + landmark_list[24]) / 2)[1] - np.min(landmark_list[:, 1]) / LEG_LENGTH

        trans_dict["Hips"] = to_angle_axis(spine_rotation)
        trans_dict["HipsTrans"] = [translation, 0, 0, 0]
    else:
        trans_dict["Spine1"] = to_angle_axis(spine_rotation)

    # left hand
    if visibility[19] > 0.5:
        hand_middle = (landmark_list[17] + landmark_list[19]) / 2.
        y = normalize(hand_middle - landmark_list[15])
        z = normalize(np.cross(landmark_list[17] - landmark_list[15], landmark_list[19] - landmark_list[15]))
        x = np.cross(y, z)
        left_hand_cs = np.asarray([x, y, z])
        left_hand_rotation = np.matmul(left_hand_cs, inv(left_fore_arm_cs))
        trans_dict["LeftHand"] = to_angle_axis(left_hand_rotation)

    # right hand
    if visibility[20] > 0.5:
        hand_middle = (landmark_list[18] + landmark_list[20]) / 2.
        y = normalize(hand_middle - landmark_list[16])
        z = normalize(np.cross(landmark_list[20] - landmark_list[16], landmark_list[18] - landmark_list[16]))
        x = np.cross(y, z)
        right_hand_cs = np.asarray([x, y, z])
        right_hand_rotation = np.matmul(right_hand_cs, inv(right_fore_arm_cs))
        trans_dict["RightHand"] = to_angle_axis(right_hand_rotation)

    # head
    if visibility[0] > 0.5:
        neck = middle(landmark_list[11], landmark_list[12])
        y = normalize(middle(landmark_list[7], landmark_list[8]) - neck)
        z = normalize(np.cross(landmark_list[7] - neck, landmark_list[8] - neck))
        x = np.cross(y, z)
        head_cs = np.asarray([x, y, z])
        head_rotation = np.matmul(head_cs, inv(spine_cs))
        trans_dict["Head"] = to_angle_axis(head_rotation)

    # If the hand landmarks are given, we optimize the gestures with the hand landmarks
    # Left hand
    if left_hand_landmarks:
        # left_hand_visibility = np.asarray([l.visibility for l in left_hand_landmarks.landmark])
        left_hand_landmark_list = np.asarray([[lmk.x, -lmk.y, -lmk.z] for lmk in left_hand_landmarks.landmark])
        hand_middle = (left_hand_landmark_list[5] + left_hand_landmark_list[17]) / 2.
        y = normalize(hand_middle - left_hand_landmark_list[0])
        z = normalize(np.cross(left_hand_landmark_list[17] - left_hand_landmark_list[0],
                               left_hand_landmark_list[5] - left_hand_landmark_list[0]))
        x = np.cross(y, z)
        left_hand_cs = np.asarray([x, y, z])
        left_hand_rotation = np.matmul(left_hand_cs, inv(left_fore_arm_cs))
        trans_dict["LeftHand"] = to_angle_axis(left_hand_rotation)

        # Left hand fingers
        # Index1
        y = normalize(left_hand_landmark_list[6] - left_hand_landmark_list[5])
        y_p = np.matmul(y, inv(left_hand_cs))
        trans_dict["LeftHandIndex1"] = vector_to_angle_axis(y_p)
        # Index2. Since the knuckles only rotate according to the x axis, we use the angle between function.
        angle = angle_between(left_hand_landmark_list[7] - left_hand_landmark_list[6],
                              left_hand_landmark_list[6] - left_hand_landmark_list[5])
        trans_dict["LeftHandIndex2"] = [angle, 1.0, 0.0, 0.0]
        # Index3
        angle = angle_between(left_hand_landmark_list[8] - left_hand_landmark_list[7],
                              left_hand_landmark_list[7] - left_hand_landmark_list[6])
        trans_dict["LeftHandIndex3"] = [angle, 1.0, 0.0, 0.0]

        # Middle1
        y = normalize(left_hand_landmark_list[10] - left_hand_landmark_list[9])
        y_p = np.matmul(y, inv(left_hand_cs))
        trans_dict["LeftHandMiddle1"] = vector_to_angle_axis(y_p)
        # Middle2
        angle = angle_between(left_hand_landmark_list[11] - left_hand_landmark_list[10],
                              left_hand_landmark_list[10] - left_hand_landmark_list[9])
        trans_dict["LeftHandMiddle2"] = [angle, 1.0, 0.0, 0.0]
        # Middle3
        angle = angle_between(left_hand_landmark_list[12] - left_hand_landmark_list[11],
                              left_hand_landmark_list[11] - left_hand_landmark_list[10])
        trans_dict["LeftHandMiddle3"] = [angle, 1.0, 0.0, 0.0]

        # Ring1
        y = normalize(left_hand_landmark_list[14] - left_hand_landmark_list[13])
        y_p = np.matmul(y, inv(left_hand_cs))
        trans_dict["LeftHandRing1"] = vector_to_angle_axis(y_p)
        # Ring2
        angle = angle_between(left_hand_landmark_list[15] - left_hand_landmark_list[14],
                              left_hand_landmark_list[14] - left_hand_landmark_list[13])
        trans_dict["LeftHandRing2"] = [angle, 1.0, 0.0, 0.0]
        # Ring3
        angle = angle_between(left_hand_landmark_list[16] - left_hand_landmark_list[15],
                              left_hand_landmark_list[15] - left_hand_landmark_list[14])
        trans_dict["LeftHandRing3"] = [angle, 1.0, 0.0, 0.0]

        # Pinky1
        y = normalize(left_hand_landmark_list[18] - left_hand_landmark_list[17])
        y_p = np.matmul(y, inv(left_hand_cs))
        trans_dict["LeftHandPinky1"] = vector_to_angle_axis(y_p)
        # Pinky2
        angle = angle_between(left_hand_landmark_list[19] - left_hand_landmark_list[18],
                              left_hand_landmark_list[18] - left_hand_landmark_list[17])
        trans_dict["LeftHandPinky2"] = [angle, 1.0, 0.0, 0.0]
        # Pinky3
        angle = angle_between(left_hand_landmark_list[20] - left_hand_landmark_list[19],
                              left_hand_landmark_list[19] - left_hand_landmark_list[18])
        trans_dict["LeftHandPinky3"] = [angle, 1.0, 0.0, 0.0]

        # Thumb1
        y = normalize(left_hand_landmark_list[2] - left_hand_landmark_list[1])
        y_p = np.matmul(y, inv(left_hand_cs))
        trans_dict["LeftHandThumb1"] = vector_to_angle_axis(y_p)
        angle_axis = trans_dict["LeftHandThumb1"]
        r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
        left_thumb1_cs = np.matmul(r.rotation_matrix.transpose(), left_hand_cs)
        # Thumb2
        angle = angle_between(left_hand_landmark_list[3] - left_hand_landmark_list[2],
                              left_hand_landmark_list[2] - left_hand_landmark_list[1])
        y = normalize(left_hand_landmark_list[3] - left_hand_landmark_list[2])
        y_p = np.matmul(y, inv(left_thumb1_cs))
        if y_p[0] < 0:
            trans_dict["LeftHandThumb2"] = [angle, 0.0, 0.0, 1.0]
        else:
            trans_dict["LeftHandThumb2"] = [-angle, 0.0, 0.0, 1.0]
        angle_axis = trans_dict["LeftHandThumb2"]
        r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
        left_thumb2_cs = np.matmul(r.rotation_matrix.transpose(), left_thumb1_cs)
        # Thumb3
        angle = angle_between(left_hand_landmark_list[4] - left_hand_landmark_list[3],
                              left_hand_landmark_list[3] - left_hand_landmark_list[2])
        y = normalize(left_hand_landmark_list[4] - left_hand_landmark_list[3])
        y_p = np.matmul(y, inv(left_thumb2_cs))
        if y_p[0] < 0:
            trans_dict["LeftHandThumb3"] = [angle, 0.0, 0.0, 1.0]
        else:
            trans_dict["LeftHandThumb3"] = [-angle, 0.0, 0.0, 1.0]

        # # Thumb1
        # y = normalize(left_hand_landmark_list[2] - left_hand_landmark_list[1])
        # yP = np.matmul(y, inv(left_hand_cs))
        # transDict["LeftHandThumb1"] = vector_to_angle_axis(yP)
        # # Thumb2
        # angle = angle_between(left_hand_landmark_list[3] - left_hand_landmark_list[2],
        #                       left_hand_landmark_list[2] - left_hand_landmark_list[1])
        # transDict["LeftHandThumb2"] = [angle, 0.0, 0.0, -1.0]
        # # Thumb3
        # angle = angle_between(left_hand_landmark_list[4] - left_hand_landmark_list[3],
        #                       left_hand_landmark_list[3] - left_hand_landmark_list[2])
        # transDict["LeftHandThumb3"] = [angle, 0.0, 0.0, -1.0]

    # Right hand
    if right_hand_landmarks:
        # right_hand_visibility = np.asarray([lmk.visibility for lmk in right_hand_landmarks.landmark])
        right_hand_landmark_list = np.asarray([[lmk.x, -lmk.y, -lmk.z] for lmk in right_hand_landmarks.landmark])
        hand_middle = (right_hand_landmark_list[5] + right_hand_landmark_list[17]) / 2.
        y = normalize(hand_middle - right_hand_landmark_list[0])
        z = normalize(np.cross(right_hand_landmark_list[5] - right_hand_landmark_list[0],
                               right_hand_landmark_list[17] - right_hand_landmark_list[0]))
        x = np.cross(y, z)
        right_hand_cs = np.asarray([x, y, z])
        right_hand_rotation = np.matmul(right_hand_cs, inv(right_fore_arm_cs))
        trans_dict["RightHand"] = to_angle_axis(right_hand_rotation)

        # Right hand fingers
        # Index1
        y = normalize(right_hand_landmark_list[6] - right_hand_landmark_list[5])
        y_p = np.matmul(y, inv(right_hand_cs))
        trans_dict["RightHandIndex1"] = vector_to_angle_axis(y_p)
        # Index2. Since the knuckles only rotate according to the x axis, we use the angle between function.
        angle = angle_between(right_hand_landmark_list[7] - right_hand_landmark_list[6],
                              right_hand_landmark_list[6] - right_hand_landmark_list[5])
        trans_dict["RightHandIndex2"] = [angle, 1.0, 0.0, 0.0]
        # Index3
        angle = angle_between(right_hand_landmark_list[8] - right_hand_landmark_list[7],
                              right_hand_landmark_list[7] - right_hand_landmark_list[6])
        trans_dict["RightHandIndex3"] = [angle, 1.0, 0.0, 0.0]

        # Middle1
        y = normalize(right_hand_landmark_list[10] - right_hand_landmark_list[9])
        y_p = np.matmul(y, inv(right_hand_cs))
        trans_dict["RightHandMiddle1"] = vector_to_angle_axis(y_p)
        # Middle2
        angle = angle_between(right_hand_landmark_list[11] - right_hand_landmark_list[10],
                              right_hand_landmark_list[10] - right_hand_landmark_list[9])
        trans_dict["RightHandMiddle2"] = [angle, 1.0, 0.0, 0.0]
        # Middle3
        angle = angle_between(right_hand_landmark_list[12] - right_hand_landmark_list[11],
                              right_hand_landmark_list[11] - right_hand_landmark_list[10])
        trans_dict["RightHandMiddle3"] = [angle, 1.0, 0.0, 0.0]

        # Ring1
        y = normalize(right_hand_landmark_list[14] - right_hand_landmark_list[13])
        y_p = np.matmul(y, inv(right_hand_cs))
        trans_dict["RightHandRing1"] = vector_to_angle_axis(y_p)
        # Ring2
        angle = angle_between(right_hand_landmark_list[15] - right_hand_landmark_list[14],
                              right_hand_landmark_list[14] - right_hand_landmark_list[13])
        trans_dict["RightHandRing2"] = [angle, 1.0, 0.0, 0.0]
        # Ring3
        angle = angle_between(right_hand_landmark_list[16] - right_hand_landmark_list[15],
                              right_hand_landmark_list[15] - right_hand_landmark_list[14])
        trans_dict["RightHandRing3"] = [angle, 1.0, 0.0, 0.0]

        # Pinky1
        y = normalize(right_hand_landmark_list[18] - right_hand_landmark_list[17])
        y_p = np.matmul(y, inv(right_hand_cs))
        trans_dict["RightHandPinky1"] = vector_to_angle_axis(y_p)
        # Pinky2
        angle = angle_between(right_hand_landmark_list[19] - right_hand_landmark_list[18],
                              right_hand_landmark_list[18] - right_hand_landmark_list[17])
        trans_dict["RightHandPinky2"] = [angle, 1.0, 0.0, 0.0]
        # Pinky3
        angle = angle_between(right_hand_landmark_list[20] - right_hand_landmark_list[19],
                              right_hand_landmark_list[19] - right_hand_landmark_list[18])
        trans_dict["RightHandPinky3"] = [angle, 1.0, 0.0, 0.0]

        # Thumb1
        y = normalize(right_hand_landmark_list[2] - right_hand_landmark_list[1])
        y_p = np.matmul(y, inv(right_hand_cs))
        trans_dict["RightHandThumb1"] = vector_to_angle_axis(y_p)
        angle_axis = trans_dict["RightHandThumb1"]
        r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
        right_thumb1_cs = np.matmul(r.rotation_matrix.transpose(), right_hand_cs)
        # Thumb2
        angle = angle_between(right_hand_landmark_list[3] - right_hand_landmark_list[2],
                              right_hand_landmark_list[2] - right_hand_landmark_list[1])
        y = normalize(right_hand_landmark_list[3] - right_hand_landmark_list[2])
        y_p = np.matmul(y, inv(right_thumb1_cs))
        if y_p[0] < 0:
            trans_dict["RightHandThumb2"] = [angle, 0.0, 0.0, 1.0]
        else:
            trans_dict["RightHandThumb2"] = [-angle, 0.0, 0.0, 1.0]
        angle_axis = trans_dict["RightHandThumb2"]
        r = Quaternion(axis=[angle_axis[1], angle_axis[2], angle_axis[3]], degrees=angle_axis[0])
        right_thumb2_cs = np.matmul(r.rotation_matrix.transpose(), right_thumb1_cs)
        # Thumb3
        angle = angle_between(right_hand_landmark_list[4] - right_hand_landmark_list[3],
                              right_hand_landmark_list[3] - right_hand_landmark_list[2])
        y = normalize(right_hand_landmark_list[4] - right_hand_landmark_list[3])
        y_p = np.matmul(y, inv(right_thumb2_cs))
        if y_p[0] < 0:
            trans_dict["RightHandThumb3"] = [angle, 0.0, 0.0, 1.0]
        else:
            trans_dict["RightHandThumb3"] = [-angle, 0.0, 0.0, 1.0]
    return trans_dict


def angle_axis_to_string(angle_axis):
    return "{:.3f},{:.5f},{:.5f},{:.5f}".format(*angle_axis)
