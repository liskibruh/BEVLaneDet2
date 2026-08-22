import numpy as np

def get_camera_matrix(cam_h, cam_p):
    """
    cam_h: camera height in meters
    cam_p: camera pitch in radians (?)
    """
    theta = (np.pi/2) + cam_p
    
    rot = np.array([
        [1,     0,                  0             ],
        [0,     np.cos(theta),      -np.sin(theta)],
        [0,     np.sin(theta),      np.cos(theta) ],
        [0,     0,                  1             ]
    ])
    
    trans = np.array([[0, cam_h, 0, 0]]).T
    
    homo = np.concatenate([rot, trans], axis=1)
    
    return homo