import numpy as np
import cv2

from .utils import get_camera_matrix, StandardCamera

from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadLanes(BaseTransform):
    def __init__(self,
                ):
        pass
    
    def transform(self, results):
        return results

@TRANSFORMS.register_module()
class GenerateLaneTargets(BaseTransform):
    def __init__(self,
                x_range,
                y_range,
                meter_per_pixel,
                vc_config,
                use_virtual_cam=True,
                ):
        self.x_range = x_range
        self.y_range = y_range
        self.meter_per_pixel = meter_per_pixel
        self.vc_config = vc_config
        self.use_virtual_cam = use_virtual_cam
        
        # camera intrinsic
        self.K = np.array([
            [2015.,     0.,     960.],
            [0.,        2015.,  540.],
            [0.,        0.,     1.  ]
        ])
        
    def draw_lines(self, lane_ground, ground2cam, image_gt, lane_idx):
        """
        lane_ground: homogeneous lane coordinates in ground coordinate of shape (4, M)
        ground2cam: homography matrix for ground-to-cam transformation
        """
        
        # ground -> cam -> image
        lane_cam = ground2cam @ lane_ground
        lane_img = self.K @ lane_cam[ :3]
        lane_img = lane_img / lane_img[2]
        lane_uv  = lane_img[ :2].T
        
        # draw lanes on image
        cv2.polylines(
            img=        image_gt,
            pts=        [lane_uv.astype(int)],
            isClosed=   False,
            color=      lane_idx + 1,
            thickness=  3
        )
        
        return image_gt
    
    def get_bev_targets(self):
        pass
    
    def transform(self, results):
        
        cam_height = results['cam_height']
        cam_pitch = results['cam_pitch']
        lanes_ground = results['laneLines']
        image = results['img']
        image_gt = np.zeros(image.shape[:2], np.uint8)
        
        ground2cam = get_camera_matrix(cam_height, cam_pitch)
        cam2ground = np.linalg.inv(ground2cam)
        
        for idx in range(len(lanes_ground)):
            lane_visibility = np.array(results['laneLines_visibility'][idx])
            lane_ground = np.array(lanes_ground[idx])
            
            assert lane_ground.shape[0] == lane_visibility.shape[0]
            
            lane_ground = lane_ground[lane_visibility > 0.5]
            lane_ground = np.concatenate([lane_ground, np.ones([lane_ground.shape[0], 1])], axis=1).T   # (M, 3) -> (M,4) -> (4, M)
            
            image_gt = self.draw_lines(lane_ground, ground2cam, image_gt, idx)
            
            
        # virtual camera
        if self.vc_config.use_virtual_cam:    # todo: implement virtual camera
            
            sc = StandardCamera(
                self.vc_config.vc_height,
                self.vc_config.vc_pitch,
                self.vc_config.vc_im_shape,
                image.shape[:2],
                self.K,
                ground2cam
            )
            
            H = sc.get_H(height=0)
            image = cv2.warpPerspective(image, H, self.vc_config.vc_im_shape)
            image_gt = cv2.warpPerspective(image_gt, H, self.vc_config.vc_im_shape)
        
        print(f"iamge_gt.shape: {image_gt.shape}")
        print(f"np.unique(image_gt): {np.unique(image_gt)}")
        
        
        
        # results['image_gt'] = image_gt
        # results['bev_gt'] = bev_gt
        
        return results