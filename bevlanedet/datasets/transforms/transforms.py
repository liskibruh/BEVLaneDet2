import numpy as np
import cv2
from scipy.interpolate import make_interp_spline

from .utils import  get_camera_matrix, get_ground2ipm, \
                    dedup_and_sort_x, dedup_and_sort_x_old, \
                    calculate_dist, save_gt_vis, StandardCamera

from mmcv.transforms import BaseTransform, to_tensor
from mmdet.registry import TRANSFORMS
from mmengine.structures import BaseDataElement, PixelData

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
        
        self.ipm_h = int((self.x_range[1] - self.x_range[0]) / meter_per_pixel)
        self.ipm_w = int((self.y_range[1] - self.y_range[0]) / meter_per_pixel)
        
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
    
    def get_yz_offsets(self, ipm_lanes):
        """
        ipm_lanes: 
            type:   dict
            keys:   lane indices
            values: lane points in shape (3, M)
        """
        
        ipm_h = self.ipm_h
        ipm_w = self.ipm_w
        
        lanes_xy_dense, lanes_xz_dense = {}, {}
        lanes_xy_sparse_bin, lanes_xy_dense_bin = {}, {}
        
        for idx in ipm_lanes:
            
            ipm_points = np.array(ipm_lanes[idx])
            
            # filter out-of-bounds points
            
            # inbounds = np.where(
            #     (ipm_points[0] >= 0) & (ipm_points[0] < ipm_h) &
            #     (ipm_points[1] >= 0) & (ipm_points[1] < ipm_w)
            #     )[0]
            
            inbounds = np.where(
                (ipm_points[0] >= 0) & (ipm_points[0] < ipm_h)
                )[0]
            
            ipm_points = ipm_points.T[inbounds].T
            
            if len(ipm_points[0]) <= 1:
                continue
            
            x_ = ipm_points[0]
            y_ = ipm_points[1]
            z_ = ipm_points[2]
            
            # sort points and remove dups
            x, y, z = dedup_and_sort_x(x_, y_, z_)
            # x, y, z = dedup_and_sort_x_old(x_, y_, z_)
            
            if len(x) <= 1:
                continue
            
            # interpolate to dense lane points
            x_dense = np.linspace(x.min(), x.max(), 
                                    int((x.max() - x.min()) // 0.05))
            x_bin = np.linspace(int(x.min()), int(x.max()), 
                                    int(int(x.max()) - int(x.min()))+1)
            
            if len(x) <=2:
                y_fn = make_interp_spline(x, y, k=1)    # linear
                z_fn = make_interp_spline(x, z, k=1)
            
            elif len(x) <=3:
                y_fn = make_interp_spline(x, y, k=2)   # quadratic                
                z_fn = make_interp_spline(x, z, k=2)
            
            else:
                y_fn = make_interp_spline(x, y, k=3)    # cubic                
                z_fn = make_interp_spline(x, z, k=3)
            
            y_dense = y_fn(x_dense)
            y_bin   = y_fn(x_bin) 
            z_dense = z_fn(x_dense, extrapolate=False)
            
            lanes_xy_dense[idx]      = np.array([x_dense, y_dense])
            lanes_xz_dense[idx]      = np.array([x_dense, z_dense])
            lanes_xy_dense_bin[idx]  = np.array([x_dense, y_dense]).astype(int)
            lanes_xy_sparse_bin[idx] = np.array([x_bin, y_bin]).astype(int)
        
        # create offset maps
        ipm_image   = np.zeros((ipm_h, ipm_w), dtype=np.float32)
        offset_map  = np.zeros((ipm_h, ipm_w), dtype=np.float32)
        height_map  = np.zeros((ipm_h, ipm_w), dtype=np.float32)
        
        for idx in lanes_xy_sparse_bin:
            lane_sparse_bin = lanes_xy_sparse_bin[idx].T    # (2,M) -> (M,2)
            
            for pt in lane_sparse_bin:
                row, col = pt[0], pt[1]
                
                # if not(0 < row < ipm_h and 0 <= col < ipm_w):
                if not(0 <= row < ipm_h and 0 < col < ipm_w):
                    continue
                
                ipm_image[row, col] = idx
                center = np.array([row, col])
                
                offset_y, z = calculate_dist(
                    center,
                    lanes_xy_dense[idx],
                    lanes_xz_dense[idx],
                    lanes_xy_dense_bin[idx]
                )
                
                if offset_y is None:
                    ipm_image[row, col] = 0
                    continue
                
                if offset_y > 1:
                    offset_y = 1
                
                if offset_y < 0:
                    offset_y = 0
                
                offset_map[row][col] = offset_y
                height_map[row][col] = z
        
        return ipm_image, offset_map, height_map, lanes_xy_sparse_bin, lanes_xy_dense
    
    def transform(self, results):
        
        cam_height = results['cam_height']
        cam_pitch = results['cam_pitch']
        lanes_ground = results['laneLines']
        image = results['img']
        image_gt = np.zeros(image.shape[:2], np.uint8)
        
        ground2cam = get_camera_matrix(cam_height, cam_pitch)
        cam2ground = np.linalg.inv(ground2cam)
        ground2ipm = get_ground2ipm(
            self.x_range,
            self.y_range,
            self.meter_per_pixel
        )
        
        gt_lanes = []
        ipm_lanes = {}
        for idx in range(len(lanes_ground)):
            lane_visibility = np.array(results['laneLines_visibility'][idx])
            lane_ground = np.array(lanes_ground[idx])
            
            assert lane_ground.shape[0] == lane_visibility.shape[0]
            
            lane_ground = lane_ground[lane_visibility > 0.5]
            
            gt_lanes.append(lane_ground.tolist())
            
            # (M, 3) -> (M,4) -> (4, M)
            lane_ground = np.concatenate([
                lane_ground,
                np.ones([lane_ground.shape[0], 1])
            ], axis=1).T    
            
            # generate PV target
            image_gt = self.draw_lines(lane_ground, ground2cam, image_gt, idx)
            
            # generate BEV targets
            x = lane_ground[0, :]
            y = lane_ground[1, :]
            
            # ipm projection
            lane_ipm = ground2ipm @ np.vstack([
                y,
                x,
                np.ones_like(x)
            ])
            
            # concat height (z) dim
            lane_ipm = np.concatenate([
                lane_ipm,
                np.expand_dims(lane_ground[2, :], axis=0)
            ], axis=0)
            
            ipm_lanes[idx+1] = lane_ipm
        
        # generate BEV targets with offset
        # bev_gt, y_offset_map, z_map, sparse_xy_lanes, dense_xy_lanes = self.get_yz_offsets(ipm_lanes)
        bev_gt, y_offset_map, z_map, _, _ = self.get_yz_offsets(ipm_lanes)
        
        # virtual camera projection for PV
        if self.vc_config.use_virtual_cam:
            
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
        
        # debug
        # print(f"iamge_gt.shape: {image_gt.shape}")
        # print(f"np.unique(image_gt): {np.unique(image_gt)}")
        # print(f"image.shape: {image.shape}")
        # print(f"len(ipm_lanes): {len(ipm_lanes)}")
        
        # debug_gt = {
        #     'image': image,
        #     'image_gt': image_gt,
        #     'ipm_image': bev_gt,
        #     'ipm_lanes': ipm_lanes,
        #     'ipm_y_offset_map': y_offset_map,
        #     'ipm_z_map': z_map,
        #     'ipm_sparse_xy_lanes': sparse_xy_lanes,
        #     'ipm_dense_xy_lanes': dense_xy_lanes,
        #     'x_range': self.x_range,
        #     'y_range': self.y_range,
        #     'm_per_pixel': self.meter_per_pixel
        # }
        
        # fname = results['img_path'].split('/')[-1]
        
        # save_gt_vis(gt=debug_gt, fname=fname, out_dir="./")
        
        image_gt_instance = image_gt.copy()
        image_gt_segment = image_gt.copy()
        image_gt_segment[image_gt_segment > 0] = 1
        
        bev_gt_instance = bev_gt.copy()
        bev_gt_segment = bev_gt.copy()
        bev_gt_segment[bev_gt_segment > 0] = 1
        
        # results['image_gt'] = image_gt
        # results['bev_gt'] = bev_gt
        
        results['gt_lanes'] = gt_lanes
        
        results['img'] = image        
        results['image_gt_segment'] = image_gt_segment
        results['image_gt_instance'] = image_gt_instance
        
        results['bev_gt_segment'] = bev_gt_segment
        results['bev_gt_instance'] = bev_gt_instance
        results['bev_y_offset_map'] = y_offset_map
        results['bev_z_map'] = z_map
        
        results['bev_gt_shape'] = bev_gt.shape
        results['ground2cam'] = ground2cam
        results['K'] = self.K
        
        return results

@TRANSFORMS.register_module()
class MultiTargetResize(BaseTransform):
    def __init__(self, image_size, image_gt_size):
        self.im_h = image_size[0]
        self.im_w = image_size[1]
        
        self.im_gt_h = image_gt_size[0]
        self.im_gt_w = image_gt_size[1]
    
    def transform(self, results):
        
        assert results['img'] is not None
        assert results['image_gt_segment'] is not None
        
        ori_im_h = results['img'].shape[0]
        ori_im_w = results['img'].shape[1]
        
        image = results['img']
        results['img'] = cv2.resize(
            image, 
            (self.im_w, self.im_h),
            interpolation=cv2.INTER_LINEAR
        )
        
        results['img_shape'] = (self.im_h, self.im_w)   # update 'image_shape' to the resized shape
        
        image_gt_segment = results['image_gt_segment']
        results['image_gt_segment'] = cv2.resize(
            image_gt_segment,
            (self.im_gt_w, self.im_gt_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        image_gt_instance = results['image_gt_instance']
        results['image_gt_instance'] = cv2.resize(
            image_gt_instance,
            (self.im_gt_w, self.im_gt_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        results['image_gt_shape'] = (self.im_gt_h, self.im_gt_w)
        
        # adjust cam intrinsics
        sx = self.im_w/ori_im_w
        sy = self.im_h/ori_im_h
        K = results['K'].copy()
        
        K[0, 0] *= sx   # fx
        K[1, 1] *= sy   # fy
        K[0, 2] *= sx   # cx
        K[1, 2] *= sy   # cy
        
        results['K'] = K
        
        return results

@TRANSFORMS.register_module()
class PackLaneData(BaseTransform):
    """
    Pack inputs for the model.
    
    Returns a dict:
    - 'inputs': image as a (C, H, W) tensor
    - 'data_samples': a BaseDataElement with:
        - image_gt  (PixelData): front-view segmentation target (image_gt)
        - bev_gt    (PixelData): BEV targets (bev_gt, bev_y_offset_map, bev_z_map)
        - metainfo: meta_keys, plus camera params (K, ground2cam)
    """
    
    default_meta_keys = [
        'raw_file', 'cam_height', 'cam_pitch', 'sample_idx',
        'img_path', 'ori_shape', 'img_shape', 'image_gt_shape',
        'bev_gt_shape', 'ground2cam', 'K', 'gt_lanes' 
    ]
    
    def __init__(self, meta_keys=default_meta_keys):
        self.meta_keys = meta_keys
    
    def transform(self, results):
        
        packed_results = {}
        
        # ground-truth targets
        img = results['img']
        
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            img = to_tensor(img)
        else:
            img = to_tensor(img).permute(2, 0, 1).contiguous()
        
        packed_results['inputs'] = img
        
        data_sample = BaseDataElement()
        
        # FV target
        if 'image_gt_segment' in results:
            image_gt_segment = PixelData(
                sem_seg=to_tensor(results['image_gt_segment'][None, ...].copy())
            )
        if 'image_gt_instance' in results:
            image_gt_instance = PixelData(
                instance_seg=to_tensor(results['image_gt_instance'][None, ...].copy())
            )
        
        
        data_sample.image_gt_segment = image_gt_segment
        data_sample.image_gt_instance = image_gt_instance
        
        # BEV targets
        bev_fields = {}
        if 'bev_gt_segment' in results:
            bev_fields['sem_seg'] = to_tensor(results['bev_gt_segment'][None, ...].copy())
        if 'bev_gt_instance' in results:
            bev_fields['instance_seg'] = to_tensor(results['bev_gt_instance'][None, ...].copy())
        if 'bev_y_offset_map' in results:
            bev_fields['y_offset_map'] = to_tensor(results['bev_y_offset_map'][None, ...].copy())
        if 'bev_z_map' in results:
            bev_fields['z_map'] = to_tensor(results['bev_z_map'][None, ...].copy())
        
        if bev_fields:
            data_sample.bev_gt = PixelData(**bev_fields)
        
        # meta
        meta = {}
        for key in self.meta_keys:
            if key in results:
                if key in ('K', 'ground2cam'):
                    mat = to_tensor(results[key].copy())
                    meta[key] = mat
                else:
                    meta[key] = results[key]
        
        data_sample.set_metainfo(meta)
        
        packed_results['data_samples'] = data_sample
        
        return packed_results