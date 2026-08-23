import numpy as np
import cv2

def get_camera_matrix(cam_h, cam_p):
    """
    cam_h: camera height in meters
    cam_p: camera pitch in radians
    """
    theta = np.pi / 2 + cam_p
    
    homo = np.array([
        [1, 0,              0,               0    ],
        [0, np.cos(theta), -np.sin(theta),   cam_h],
        [0, np.sin(theta),  np.cos(theta),   0    ],
        [0, 0,              0,               1    ]
    ])
    
    return homo

class StandardCamera:
    def __init__(
        self,
        vc_height,
        vc_pitch,
        vc_im_shape,
        cam_im_shape,
        cam_K,
        ground2cam,
    ):
        self.vc_K = cam_K
        self.vc_ground2cam = get_camera_matrix(vc_height, vc_pitch)
        
        self.cam_K = cam_K
        self.ground2cam = ground2cam
        
        self.vc_im_shape = vc_im_shape
        self.cam_im_shape = cam_im_shape
    
    def get_H(self, height):
        u1, u2 = int(0.6 * self.cam_im_shape[0]), int(0.8 * self.cam_im_shape[0])
        v1, v2 = int(0.33 * self.cam_im_shape[1]), int(0.66 * self.cam_im_shape[1])
        
        base_pts = np.array([[u1, v1, 1], 
                            [u1, v2, 1], 
                            [u2, v1, 1], 
                            [u2, v2, 1]])
        
        H = self.get_project_matrix(base_pts, height)
        
        return H
    
    def get_project_matrix(self, base_pts, height):
        pts_ego = self.img2ego(base_pts, height)
        
        vc_ground2cam = self.vc_ground2cam
        vc_ground2cam_R = vc_ground2cam[:3, :3]
        vc_ground2cam_T = vc_ground2cam[:3, 3]
        
        pts_vc_cam = np.dot(vc_ground2cam_R, pts_ego) + vc_ground2cam_T.reshape(3, 1)
        pts_vc_im = self.vc_K @ pts_vc_cam
        pts_vc_im = pts_vc_im / pts_vc_im[2]
        
        pts_im = base_pts[:, :2].astype(np.float32)
        pts_vc_im = pts_vc_im[:2].T.astype(np.float32)
        
        H = cv2.getPerspectiveTransform(pts_im, pts_vc_im)
        
        return H
    
    def img2ego(self, pts_img, height):
        cam_K_inv = np.linalg.inv(self.cam_K)
        R_inv = self.ground2cam[:3, :3].T       # R.T = R_inverse
        T = self.ground2cam[:3, 3]
        
        mat1 = np.dot(np.dot(R_inv, cam_K_inv), pts_img.T)  # ((3,3) @ (3,3)) @ (3,4) -> (3,4)
        mat2 = np.dot(R_inv, T)
        
        Zc = (height + mat2[2]) / mat1[2]
        
        pts_ego = Zc * mat1 - np.expand_dims(mat2, 1)
        
        return pts_ego