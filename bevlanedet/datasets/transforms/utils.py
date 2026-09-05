import os
import cv2
import numpy as np

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

def get_ground2ipm(x_range, y_range, m_per_pixel=0.5):
    """
    x_range: longitudnal
    y_range: lateral
    """
    s = m_per_pixel
    tx = int(x_range[1]/s)
    ty = int(-y_range[0]/s)
    
    ground2ipm = np.array([
        [-1/s,   0,    tx],
        [0,     1/s,   ty]
    ])
    
    return ground2ipm

def dedup_and_sort_x(x, y, z):
    """
    x, y, z are IPM (BEV) lane coordinates
        x: longitudnal
        y: lateral
        z: vertical/height
    """
    
    order = np.argsort(x)
    
    sorted_x = x[order]
    sorted_y = y[order]
    sorted_z = z[order]
    
    _, first_idx = np.unique(sorted_x, return_index=True)
    first_idx = np.sort(first_idx)  # restore ascend. order
    
    x = sorted_x[first_idx]
    y = sorted_y[first_idx]
    z = sorted_z[first_idx]
    
    return x, y, z

def dedup_and_sort_x_old(x, y, z):
    order = np.argsort(x)  # ascending
    
    x_, y_, z_ = [], [], []
    
    for idx in range(len(order)):
        if idx != 0 and x[order[idx]] <= x[order[idx-1]]:
            continue
        else:
            x_.append(x[order[idx]])
            y_.append(y[order[idx]])
            z_.append(z[order[idx]])
    
    return x_, y_, z_

def calculate_dist(
    center,
    lanes_xy_dense,
    lanes_xz_dense,
    lanes_xy_dense_bin
):
    cond = np.where(
        (lanes_xy_dense_bin[0] == int(center[0])) &
        (lanes_xy_dense_bin[1] == int(center[1]))
    )
    
    if len(cond[0]) == 0:
        return None, None
    
    lane_xys_selected = lanes_xy_dense.T[cond]  # (M, 3)
    lane_xzs_selected = lanes_xz_dense.T[cond]
    
    offset_y = np.mean(lane_xys_selected[:, 1] - center[1])
    z        = np.mean(lane_xzs_selected[:, 1])
    
    return offset_y, z

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
        
        vc_ground2cam   = self.vc_ground2cam
        vc_ground2cam_R = vc_ground2cam[:3, :3]
        vc_ground2cam_T = vc_ground2cam[:3, 3]
        
        pts_vc_cam = np.dot(vc_ground2cam_R, pts_ego) + vc_ground2cam_T.reshape(3, 1)
        pts_vc_im  = self.vc_K @ pts_vc_cam
        pts_vc_im  = pts_vc_im / pts_vc_im[2]
        
        pts_im    = base_pts[:, :2].astype(np.float32)
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
    
def save_gt_vis(gt, fname, out_dir="./"):
    
    def visualize_offsets_arrows(idx, ipm_image, offset_map, out_path='offset_arrows'):
        import matplotlib.pyplot as plt
        
        rows, cols = np.where(ipm_image == idx)
        fig, ax = plt.subplots(figsize=(8, 10))
        
        ax.imshow(ipm_image == idx, cmap='gray', alpha=0.3)
        
        for row, col in zip(rows, cols):
            off = offset_map[row, col]
            # arrow from cell center to true sub-pixel position
            ax.arrow(col, row, off, 0, head_width=0.15, head_length=0.1,
                    fc='blue', ec='blue', length_includes_head=False)
            ax.plot(col + off, row, 'yo', markersize=1  )  # true position
        
        ax.set_xlabel('column (lateral)')
        ax.set_ylabel('row (longitudinal)')
        ax.set_title(f'Lane {idx}: per-cell offset arrows')
        plt.tight_layout()
        plt.savefig(out_path+f"_{idx}.png", dpi=300)
        plt.close()
    
    if gt.get('ipm_lanes', None) is not None:
        ipm_image = None
        ipm_h = int((gt['x_range'][1] - gt['x_range'][0]) / gt['m_per_pixel'])
        ipm_w = int((gt['y_range'][1] - gt['y_range'][0]) / gt['m_per_pixel'])
        
        ipm_image = np.zeros((ipm_h, ipm_w))
        ipm_lanes = gt['ipm_lanes']
        
        for idx in ipm_lanes:
            ipm_lane = ipm_lanes[idx]
            ipm_x = ipm_lane[0, :]
            ipm_y = ipm_lane[1, :]
            ipm_z = ipm_lane[2, :]
            
            for nz_x, nz_y, nz_z in zip(ipm_x, ipm_y, ipm_z):
                
                if (ipm_image.shape[0] > int(nz_x) > 0) \
                and (ipm_image.shape[1] > int(nz_y) > 0):
                    
                    cv2.circle(ipm_image, (int(nz_y), int(nz_x)), radius=1, color=(255, 0, 0), thickness=-1)
        
        cv2.imwrite(os.path.join(out_dir, f"bev_gt_{fname}"), ipm_image)
    
    if gt.get('ipm_image', None) is not None:
        # bev image
        bev_image = gt['ipm_image'].copy()
        sparse_xy_lanes = gt['ipm_sparse_xy_lanes'].copy()
        nonzero_gt_idxs = np.where(bev_image>0)
        bev_image[nonzero_gt_idxs] = 255
        cv2.imwrite(os.path.join(out_dir, f"bev_gt_final_{fname}"), bev_image)
        
        # y_offset
        bev_image = gt['ipm_image'].copy()
        y_offset_map = gt['ipm_y_offset_map'].copy()
        sparse_xy_lanes = gt['ipm_sparse_xy_lanes'].copy()
        dense_xy_lanes = gt['ipm_dense_xy_lanes'].copy()
        
        for idx in sparse_xy_lanes:
            visualize_offsets_arrows(idx, bev_image, y_offset_map)
    
    if gt.get('image_gt', None) is not None:
        image = gt['image']
        image_gt = gt['image_gt']
        
        nonzero_gt_idxs = np.where(image_gt>0)
        image_gt[nonzero_gt_idxs] = 255
        
        nz_xs = nonzero_gt_idxs[1]
        nz_ys = nonzero_gt_idxs[0]
        
        for nz_x, nz_y in zip(nz_xs, nz_ys):
            cv2.circle(image, (nz_x, nz_y), radius=1, color=(255, 0, 0), thickness=-1)
        
        cv2.imwrite(os.path.join(out_dir, f"gt_{fname}"), image_gt)
        cv2.imwrite(os.path.join(out_dir, f"rgb_{fname}"), image)
