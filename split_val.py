import os
from glob import glob

test_img_paths = []
test_mask_paths = []
test_data_path = ["/Users/brown/code/polyp_segmentation/data/TrainDataset"]
for i in test_data_path:
    test_img_paths.extend(glob(os.path.join(i, "images", "*")))
    test_mask_paths.extend(glob(os.path.join(i, "masks", "*")))
import numpy as np
import random

ind = list(range(len(test_img_paths)))
random.shuffle(ind)
val_img_path = np.array(test_img_paths)[ind[:100]]
val_masks_path = np.array(test_mask_paths)[ind[:100]]
import os

for i in range(len(val_img_path)):
    cmd = f"cp {val_img_path[i]} data/ValDataset/images"
    os.system(cmd)
    cmd = f"rm {val_img_path[i]}"
    os.system(cmd)
    cmd = f"cp {val_masks_path[i]} data/ValDataset/masks"
    os.system(cmd)
    cmd = f"rm {val_masks_path[i]}"
    os.system(cmd)
    print(cmd)