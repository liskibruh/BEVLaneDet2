import numpy as np
import os
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    data_pth = "/data8t_1/owais_tahir/BEVLaneDet2/data/Apollo_Sim_3D_Lane_Release/images"
    
    channel_sum = np.zeros(3, dtype=float)
    channel_sum_sq = np.zeros(3, dtype=float)
    pixel_count = 0
    
    for root, dirs, files in os.walk(data_pth):
        for fname in tqdm(files, total=len(files), desc=f"comulating stats of images in dir {root}"):
            if fname.endswith('.jpg'):
                im_pth = os.path.join(root, fname)
                im = cv2.imread(im_pth)
                im = im.astype(float)
                
                channel_sum += im.sum(axis=(0,1))
                channel_sum_sq += (im ** 2).sum(axis=(0,1))
                pixel_count += im.shape[0] * im.shape[1]
    
    mean = channel_sum / pixel_count
    variance = (channel_sum_sq / pixel_count) - mean**2
    std = np.sqrt(variance)
    
    print("="*64)
    print(f"mean: {mean}")
    print(f"std: {std}")
    print("="*64)