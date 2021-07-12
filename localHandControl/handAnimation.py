"""Examples of using pyrender for viewing and offscreen rendering.
"""
import matplotlib.pyplot as plt
import numpy as np
import pyglet
import trimesh
from pyrender import PerspectiveCamera, \
    DirectionalLight, SpotLight, PointLight, \
    Mesh, Scene, \
    Viewer, OffscreenRenderer, RenderFlags

pyglet.options['shadow_window'] = False

# ==============================================================================
# Mesh creation
# ==============================================================================

# ------------------------------------------------------------------------------
# Creating textured meshes from trimeshes
# ------------------------------------------------------------------------------

# hand trimesh
# hand_gltf = trimesh.load('./models/hand.glb')
# hand_trimesh = hand_gltf.geometry[list(hand_gltf.geometry.keys())[0]]
# hand_mesh = Mesh.from_trimesh(hand_trimesh)

drone_gltf = trimesh.load('./models/drone.glb')
drone_parts = list(drone_gltf.geometry.keys())
drone_trimeshes = [drone_gltf.geometry[x] for x in drone_parts]
drone_meshes = [Mesh.from_trimesh(x) for x in drone_trimeshes]
poses = [x.principal_inertia_transform for x in drone_trimeshes]

# hand_pose = np.array([
#     [1.0, 0.0, 0.0, 0.1],
#     [0.0, 0.0, -1.0, -0.16],
#     [0.0, 1.0, 0.0, 0.13],
#     [0.0, 0.0, 0.0, 1.0],
# ])

# ==============================================================================
# Light creation
# ==============================================================================
direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                   innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)
point_l = PointLight(color=np.ones(3), intensity=10.0)

# ==============================================================================
# Camera creation
# ==============================================================================

cam = PerspectiveCamera(yfov=(np.pi / 3.0))
cam_pose = np.array([
    [0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0.5],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0.4],
    [0.0, 0.0, 0.0, 1.0]
])

# ==============================================================================
# Scene creation
# ==============================================================================

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

# ==============================================================================
# Adding objects to the scene
# ==============================================================================

# ------------------------------------------------------------------------------
# By using the add() utility function
# ------------------------------------------------------------------------------
# hand_node = scene.add(hand_mesh, pose=hand_pose)
for i in range(len(drone_meshes)):
    scene.add(drone_meshes[i], pose=poses[i])
direc_l_node = scene.add(direc_l, pose=cam_pose)
spot_l_node = scene.add(spot_l, pose=cam_pose)

# ==============================================================================
# Using the viewer with a default camera
# ==============================================================================

v = Viewer(scene, shadows=True)

# ==============================================================================
# Using the viewer with a pre-specified camera
# ==============================================================================
cam_node = scene.add(cam, pose=cam_pose)

# ==============================================================================
# Rendering offscreen from that camera
# ==============================================================================

r = OffscreenRenderer(viewport_width=640 * 2, viewport_height=480 * 2)
color, depth = r.render(scene)

plt.figure()
plt.imshow(color)
plt.show()

# ==============================================================================
# Segmask rendering
# ==============================================================================

nm = {node: 20 * (i + 1) for i, node in enumerate(scene.mesh_nodes)}
seg = r.render(scene, RenderFlags.SEG, nm)[0]
plt.figure()
plt.imshow(seg)
plt.show()

r.delete()
