import argparse
import os
from pathlib import Path
import numpy as np

import blenderproc as bproc


parser = argparse.ArgumentParser()

parser.add_argument("mesh", nargs="?", default="", help="Path to mesh file to load")

args = parser.parse_args()


bproc.init()

objs = bproc.loader.load_obj(Path(args.mesh))

for obj in objs:
    pass


# Setup Camera
fx = 1600.1475830078125
fy = 1600.1475830078125
cx = 960.0
cy = 720.0
image_width = cx * 2.0
image_height = cy * 2.0

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

bproc.camera.set_intrinsics_from_K_matrix(K, image_width, image_height)

RT = np.array([])

# bproc.camera.add_camera_pose(RT)
data = bproc.renderer.render()
