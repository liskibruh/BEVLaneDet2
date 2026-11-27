import os
import shutil
import random
from tqdm import tqdm

NUM_SAMPLES = 25                                # how many samples to select from each segment
SHUFFLE = True                                  # True: select samples randomly
                                                # False: select the first NUM_SAMPLES samples after ascend. sort
DATA_ROOT = "data/OpenLane"                     # path ta input data
OUT_DATA_DIR = "data/OpenLane/random_subset"    # where to store the sampled data
os.makedirs(OUT_DATA_DIR, exist_ok=True)  

def sample_from_segments(segments, num=25):
    
    selected_samples = {}
    
    for segment in segments:
                
        segment_annos = sorted(
            f for f in os.listdir(segment)
            if f.endswith('.json')
            and os.path.isfile(os.path.join(segment, f))
        )
        
        seed = 42
        segment_seed = f"{seed}_{os.path.basename(segment)}"
        segment_rng = random.Random(segment_seed)
        
        if SHUFFLE:
            segment_rng.shuffle(segment_annos)
        
        selected_samples[segment] = segment_annos[:num]
        
    return selected_samples

def save_to_new_set(samples_dict, split):
    
    for segment, samples in samples_dict.items():
        for sample in tqdm(samples, total=len(samples), desc=f"[{split.upper()} SPLIT]: random sampling from {os.path.basename(segment)}"):
            anno_src = os.path.join(segment, sample)
            
            anno_dist_dir = os.path.join(OUT_DATA_DIR, split, os.path.basename(segment))
            os.makedirs(anno_dist_dir, exist_ok=True)
            anno_dst = os.path.join(anno_dist_dir, sample)
            
            try:
                shutil.copy2(anno_src, anno_dst)
            except(OSError, shutil.Error) as e:
                print(f"Failed to copy {anno_src} to {anno_dst}")
                raise
        

if __name__ == "__main__":
    annos_dir = os.path.join(DATA_ROOT, "lane3d_300")
       
    for split in ["training", "validation"]:
        
        split_dir = os.path.join(annos_dir, split)
        
        segments = sorted(
            os.path.join(split_dir, segment)
            for segment in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, segment))
        ) 
                
        selected_samples = sample_from_segments(
            segments, 
            num=NUM_SAMPLES
        )
        
        save_to_new_set(
            selected_samples, 
            split
        )