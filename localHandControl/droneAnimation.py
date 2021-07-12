import socket
import time

import cv2
import mediapipe as mp
import numpy as np
import pyglet
import trimesh
from pyrender import PerspectiveCamera, \
    DirectionalLight, SpotLight, PointLight, \
    Mesh, Scene, \
    OffscreenRenderer, RenderFlags
import socket
import time

import cv2
import mediapipe as mp
import numpy as np
import pyglet
import trimesh
from pyrender import PerspectiveCamera, \
    DirectionalLight, SpotLight, PointLight, \
    Mesh, Scene, \
    OffscreenRenderer, RenderFlags

UDP_IP = "192.168.1.197"
UDP_PORT = 5005
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
cap = cv2.VideoCapture(1)


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


c_time = time.time()

pyglet.options['shadow_window'] = False
drill_trimesh = trimesh.load('./models/drill.obj')
drill_mesh = Mesh.from_trimesh(drill_trimesh)
drill_pose = np.eye(4)
drill_pose[0, 3] = 0.1
drill_pose[2, 3] = -np.min(drill_trimesh.vertices[:, 2])

drill_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                   innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)
point_l = PointLight(color=np.ones(3), intensity=10.0)
cam = PerspectiveCamera(yfov=(np.pi / 3.0))
cam_pose = np.array([
    [0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0.2],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0.2],
    [0.0, 0.0, 0.0, 1.0]
])
scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
drill_node = scene.add(drill_mesh, pose=drill_pose)
direc_l_node = scene.add(direc_l, pose=cam_pose)
spot_l_node = scene.add(spot_l, pose=cam_pose)

# ==============================================================================
cam_node = scene.add(cam, pose=cam_pose)

r = OffscreenRenderer(viewport_width=640, viewport_height=480)
color, depth = r.render(scene)

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
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
        rotation_angle = 0.
        up = np.asarray([0., 1., 0.])
        rotation_vec = up
        tx = 0.
        ty = 0.

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # print(hand_landmarks)
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Calculate the hand direction
                landmark = list(hand_landmarks.landmark)

                hand_direction = np.asarray(diff(landmark[0], landmark[9]))
                tx = landmark[0].x
                ty = landmark[0].y
                hand_direction = unit_vector(hand_direction)
                # print("Hand direction, ", hand_direction)

                rotation_vec = np.cross(up, hand_direction)
                rotation_angle = angle_between(up, hand_direction)
                print("Rotation information: {}, {}, {}, {}".format(rotation_angle, *rotation_vec).encode())
                udp_send("{}, {}, {}, {}".format(rotation_angle, *rotation_vec).encode())
        time.sleep(0.1)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        rotate = trimesh.transformations.rotation_matrix(
            angle=np.radians(rotation_angle),
            direction=rotation_vec,
            point=[0, 0, 0])
        rotate2 = trimesh.transformations.rotation_matrix(
            angle=np.radians(90),
            direction=[1, 0, 0],
            point=[0, 0, 0])
        translate = trimesh.transformations.translation_matrix(
            [(tx-0.5)/5, 0., (ty-0.5)/5]
        )
        scene.set_pose(drill_node, np.dot(np.dot(rotate, rotate2), translate))
        color, depth = r.render(scene, flags=RenderFlags.RGBA)
        color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)
        cv2.imshow("RGB", color.astype("uint8"))
        cv2.imshow("Depth", depth * 256)
        print(np.max(depth), depth.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)

cap.release()
