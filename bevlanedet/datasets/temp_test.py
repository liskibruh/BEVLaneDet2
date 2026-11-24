# from mmcv.transforms import LoadImageFromFile
# from bevlanedet.datasets.openlane_lanes import OpenLaneDataset
# from bevlanedet.datasets.transforms.transforms import GenerateOpenLaneTargets

# if __name__ == "__main__":
    
#     vc_config = dict(
#         use_virtual_cam=True,
#         vc_intrinsic=[
#             [
#                 [2081.5212033927246, 0.0, 934.7111248349433],
#                 [0.0, 2081.5212033927246, 646.3389987785433],
#                 [0.0, 0.0, 1.0]
#             ]
#         ],
#         vc_extrinsic=[
#             [-0.002122161262459438, 0.010697496358766389, 0.9999405282331697, 1.5441039498273286],
#             [-0.9999378331046326, -0.010968621415360667, -0.0020048117763292747, -0.023774034344867204],
#             [0.010946522625388108, -0.9998826195688676, 0.01072010851209982, 2.1157397903843567],
#             [0.0, 0.0, 0.0, 1.0]
#         ],
#         vc_im_shape=(1920, 1280)
# )
    
#     dataset = OpenLaneDataset(
#         data_root= r"C:\Users\oo_wa\Downloads\python_scripts\BEVLaneDet2\data\OpenLane",
#         ann_dir = "lane3d_300/training/",
#         lazy_init = False,
#         pipeline = [
#             LoadImageFromFile(),
#             GenerateOpenLaneTargets(
#                 x_range=(3, 103),
#                 y_range=(-12, 12),
#                 meter_per_pixel=0.5,
#                 vc_config=vc_config
#             )
            
#         ]
#     )
    
#     print(dataset.__getitem__(200))