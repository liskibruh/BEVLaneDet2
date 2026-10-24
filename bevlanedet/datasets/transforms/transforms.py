import numpy as np
import torch
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
        results['cam2ground'] = cam2ground
        results['ground2ipm'] = ground2ipm
        results['H_vc'] = H
        
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
    
import matplotlib.pyplot as plt
import numpy as np


def visualize_voxel_projections(
    warped_img,
    uv_warped,
    uv_feature,
    voxel_centers=None,
    feat_W=256,
    feat_H=144,
):
    """
    warped_img: warped image in HWC format
    uv_warped: projected pixels in warped-image coordinates, shape (2, N)
    uv_feature: corresponding feature coordinates, shape (2, M)
    voxel_centers: (3, N) voxel centers in [X, Y, Z] order
    """

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: warped-image coordinates
    axes[0].imshow(warped_img)
    axes[0].scatter(
        uv_warped[0],
        uv_warped[1],
        s=1,
        alpha=0.35,
    )
    axes[0].set_xlim(0, warped_img.shape[1])
    axes[0].set_ylim(warped_img.shape[0], 0)
    axes[0].set_title("Voxel projections on warped image")
    axes[0].set_xlabel("u (pixels)")
    axes[0].set_ylabel("v (pixels)")

    # Panel 2: feature-map coordinates
    axes[1].scatter(
        uv_feature[0],
        uv_feature[1],
        s=2,
        alpha=0.5,
    )
    axes[1].set_xlim(0, feat_W)
    axes[1].set_ylim(feat_H, 0)
    axes[1].set_aspect("equal")
    axes[1].set_title("Voxel projections on feature map")
    axes[1].set_xlabel("u (feature pixels)")
    axes[1].set_ylabel("v (feature pixels)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(r"C:\Users\oo_wa\Downloads\python_scripts\BEVLaneDet2\tools\sample_img.png")
    plt.show()
    
@TRANSFORMS.register_module()
class Voxel_Generator(BaseTransform):
    def __init__(
        self,
        base_height=1.1,
        roi_x=(3, 103),       #  X: forward
        roi_y=(-12, 12),      #  Y: lateral
        z_range=7,            #  Z: vertical half-range
        grid_res=(0.5, 0.5, 0.5),  # (dx, dy, dz)
        input_shape=(576, 1024),
        feat_downscale=16
    ):
        self.base_height = base_height
        self.z_range = z_range
        self.feat_downscale = feat_downscale
        self.input_shape = input_shape

        self.roi_x = torch.tensor(roi_x, dtype=torch.float32)
        self.roi_y = torch.tensor(roi_y, dtype=torch.float32)
        self.grid_res = torch.tensor(grid_res, dtype=torch.float32)

        self.num_grids_x = int(
            (self.roi_x[1] - self.roi_x[0]) / self.grid_res[0]
        )
        self.num_grids_y = int(
            (self.roi_y[1] - self.roi_y[0]) / self.grid_res[1]
        )
        self.num_grids_z = int(
            (2 * self.z_range) / self.grid_res[2]
        )

        self._build_centers()

    def _build_centers(self):
        x_centers = (
            torch.arange(self.num_grids_x) * self.grid_res[0]
            + self.roi_x[0]
            + self.grid_res[0] / 2
        )
        y_centers = (
            torch.arange(self.num_grids_y) * self.grid_res[1]
            + self.roi_y[0]
            + self.grid_res[1] / 2
        )
        z_centers = (
            torch.arange(self.num_grids_z) * self.grid_res[2]
            + self.base_height
            - self.z_range
            + self.grid_res[2] / 2
        )

        # BEV centers: (X, Y)
        xx, yy = torch.meshgrid(x_centers, y_centers, indexing="ij")
        self.map_centers = torch.stack((xx, yy), dim=-1).reshape(-1, 2)

        # 3D voxel centers: (X, Y, Z)
        xx, yy, zz = torch.meshgrid(
            x_centers, y_centers, z_centers, indexing="ij"
        )
        self.voxel_centers = torch.stack(
            (xx, yy, zz), dim=-1
        ).reshape(-1, 3).T.contiguous().numpy()

    def transform(self, results: dict):
        cam_intrinsic = np.asarray(results["K"], dtype=np.float64)
        H_vc = np.asarray(results["H_vc"], dtype=np.float64)
        num_voxels = self.voxel_centers.shape[1]
        
        ground2cam = results["ground2cam"]
        
        voxel_centers = self.voxel_centers[[1, 0, 2], :]     # (3, N)
        ones = np.ones((1, voxel_centers.shape[1]), dtype=voxel_centers.dtype)
        
        centers_homo = np.concatenate(
            [voxel_centers, ones], axis=0
        )
        
        voxel_cam = ground2cam @ centers_homo
        
        uv_indices_full = np.zeros((2, num_voxels), dtype=np.int64)
        voxel_valid_mask = np.zeros(num_voxels, dtype=bool)
        voxel_ids = np.flatnonzero(voxel_cam[2, :] > 1e-3)
        
        # valid_depth = voxel_cam[2, :] > 1e-3
        
        # for visual
        # vis_voxel_centers_valid = ((voxel_centers[2, :] > 1e-3) & (voxel_centers[2, :] <= 1.6))
        # vis_voxel_centers = voxel_centers[:, vis_voxel_centers_valid]
        # vis_voxel_centers = voxel_centers.copy()
        # vis_ones = np.ones((1, vis_voxel_centers.shape[1]), dtype=vis_voxel_centers.dtype)
        # vis_centers_homo = np.concatenate(
        #     [vis_voxel_centers, vis_ones], axis=0
        # )
        # voxel_cam_vis = ground2cam @ vis_centers_homo
        # vis_valid = ((voxel_cam_vis[2, :] > 1e-3) & (voxel_cam_vis[2,: ] <=5.00))
        # voxel_cam_vis = voxel_cam_vis[:3, vis_valid]
        # voxel_uvz_vis = cam_intrinsic @ voxel_cam_vis[:3, :]
        # voxel_uv_vis = (
        #     voxel_uvz_vis[:2, :] / voxel_uvz_vis[2:3, :]
        # )
        # voxel_uv_homo_vis = np.vstack([
        #     voxel_uv_vis,
        #     np.ones((1, voxel_uv_vis.shape[1])),
        # ])
        # vis_uvz_warped_homo = H_vc @ voxel_uv_homo_vis
        # vis_valid_warped_homo = np.abs(vis_uvz_warped_homo[2, :] > 1e-8)
        # vis_uv_warped = (
        #     vis_uvz_warped_homo[:2, vis_valid_warped_homo]
        #     / vis_uvz_warped_homo[2:3, vis_valid_warped_homo]
        # )
        
        voxel_cam_valid = voxel_cam[:3, voxel_ids]
        voxel_uvz = cam_intrinsic @ voxel_cam_valid
        voxel_uv = (
            voxel_uvz[:2, :] / voxel_uvz[2:3, :]
        )
        voxel_uv_homo = np.vstack([
            voxel_uv,
            np.ones((1, voxel_uv.shape[1])),
        ])
                
        uv_warped_homo = H_vc @ voxel_uv_homo
        valid_warped_homo = np.abs(uv_warped_homo[2, :]) > 1e-8
        voxel_ids = voxel_ids[valid_warped_homo]
        uv_warped = (
            uv_warped_homo[:2, valid_warped_homo]
            / uv_warped_homo[2:3, valid_warped_homo]
        )

        featH = self.input_shape[0]/self.feat_downscale
        featW = self.input_shape[1]/self.feat_downscale
        
        # feat_H, feat_W = 144, 256
        scale_x = featW/results["img"].shape[1]
        scale_y = featH/results["img"].shape[0] 
        
        # for visual
        # vis_uv_feature = vis_uv_warped * np.array([
        #     [scale_x],
        #     [scale_y],
        # ])
        
        uv_feature = uv_warped * np.array([
            [scale_x],
            [scale_y],
        ])
        
        valid_bounds = (
            (uv_feature[0]>=0) & (uv_feature[0]<featW) &
            (uv_feature[1]>=0) & (uv_feature[1]<featH)
        )
        
        # for visual
        # valid_bounds_vis = (
        #     (vis_uv_feature[0]>=0) & (vis_uv_feature[0]<featW) &
        #     (vis_uv_feature[1]>=0) & (vis_uv_feature[1]<featH)
        # )
        
        # vis_uv_feature = vis_uv_feature[:, valid_bounds_vis]
        # visualize_voxel_projections(warped_img=results["img"], uv_warped=vis_uv_warped, uv_feature=vis_uv_feature)
        # exit()
        
        voxel_ids = voxel_ids[valid_bounds]
        uv_feature = uv_feature[:, valid_bounds]
        
        uv_indices_full[:, voxel_ids] = np.floor(
            uv_feature,
        ).astype(np.int64)
        
        voxel_valid_mask[voxel_ids] = True
                
        results["voxels_info"] = dict(
            voxel_uv_idx=to_tensor(uv_indices_full),
            voxel_valid_mask=to_tensor(voxel_valid_mask),
            roi_x=self.roi_x,
            roi_y=self.roi_y,
            z_range=self.z_range,
            num_grids_x=self.num_grids_x,
            num_grids_y=self.num_grids_y,
            num_grids_z=self.num_grids_z,
            grid_res=self.grid_res,
            base_height=self.base_height,
        )

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


@TRANSFORMS.register_module()
class PackLaneDataVox(BaseTransform):
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
        
        packed_results['inputs'] = {'img': img}
        if results.get('voxels_info', None) is not None:
            voxels_info = results['voxels_info']
            packed_results['inputs']['voxel_proj_index'] = voxels_info['voxel_uv_idx']
            packed_results['inputs']['voxel_valid_mask'] = voxels_info['voxel_valid_mask']
        
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