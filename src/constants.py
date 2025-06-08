import numpy as np

# we scale the depth value so that it can fit in np.uint16 and use image compression algorithms
DEPTH_IMG_SCALE = 16384

# simulation initialization settings
TABLE_HEIGHT = 0.5
OBJ_INIT_TRANS = np.array([0.45, 0.2, 0.6])
OBJ_RAND_RANGE = 0.3
OBJ_RAND_SCALE = 0.05

# clip the point cloud to a box
PC_MIN = np.array(
    [
        0.2,
        0,
        0.735,
    ]
)
PC_MAX = np.array(
    [
        1,
        0.8,
        0.85,
    ]
)

OBSERVING_QPOS_DELTA = np.array([0.01,0,0.25,0,0,0,0.15])