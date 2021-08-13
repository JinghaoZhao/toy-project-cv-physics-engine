import numpy as np
from numpy.linalg import inv
from pyquaternion import Quaternion


def normalize(v):
    return v / np.linalg.norm(v)


def to_angle_axis(rotationM):
    q = Quaternion(matrix=rotationM.transpose())
    return [q.degrees, q.axis[0], q.axis[1], q.axis[2]]


def to_trans_dict(pose_landmark):
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

    transDict["Hips"] = to_angle_axis(spineRotation)
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
    leftUpLegAngle = to_angle_axis(leftUpLegCS)[0]
    leftLegAngle = to_angle_axis(leftLegCS)[0]
    translation = np.sum(np.cos(np.radians([leftUpLegAngle, leftLegAngle]))) / -2.
    transDict["HipsTrans"] = [translation, 0, 0, 0]
    return transDict


def angle_axis_to_string(angle_axis):
    return "{:.3f},{:.5f},{:.5f},{:.5f}".format(*angle_axis)
