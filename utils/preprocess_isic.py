import os
import glob
import tqdm

Dataset_add = "data/ISIC2018_Task1/traindataset/images/"
all_img_path = glob.glob(Dataset_add + "*")
outdir = "data/ISIC2018_BCDU"
os.makedirs(os.path.join(outdir, "traindataset", "images"), exist_ok=True)
os.makedirs(os.path.join(outdir, "traindataset", "masks"), exist_ok=True)
os.makedirs(os.path.join(outdir, "valdataset", "masks"), exist_ok=True)
os.makedirs(os.path.join(outdir, "valdataset", "images"), exist_ok=True)
os.makedirs(os.path.join(outdir, "testdataset", "images"), exist_ok=True)
os.makedirs(os.path.join(outdir, "testdataset", "masks"), exist_ok=True)

train_img_path = all_img_path[0:1815]
val_img_path = all_img_path[1815 : 1815 + 259]
test_img_path = all_img_path[1815 + 259 : 2594]

for path in tqdm.tqdm(train_img_path):
    cmd = f'cp {path} {os.path.join(outdir,"traindataset","images")}'
    os.system(cmd)
    mask_file = os.path.splitext(os.path.basename(path))[0] + "_segmentation.png"
    mask_path = os.path.join("data/ISIC2018_Task1/traindataset/masks", mask_file)
    if os.path.exists(mask_path):
        cmd = f'cp {mask_path} {os.path.join(outdir,"traindataset","masks")}'
        os.system(cmd)
    else:
        print("not exist {}, check again".format(mask_path))

for path in tqdm.tqdm(val_img_path):
    cmd = f'cp {path} {os.path.join(outdir,"valdataset","images")}'
    os.system(cmd)
    mask_file = os.path.splitext(os.path.basename(path))[0] + "_segmentation.png"
    mask_path = os.path.join("data/ISIC2018_Task1/traindataset/masks", mask_file)
    if os.path.exists(mask_path):
        cmd = f'cp {mask_path} {os.path.join(outdir,"valdataset","masks")}'
        os.system(cmd)
    else:
        print("not exist {}, check again".format(mask_path))

for path in tqdm.tqdm(test_img_path):
    cmd = f'cp {path} {os.path.join(outdir,"testdataset","images")}'
    os.system(cmd)
    mask_file = os.path.splitext(os.path.basename(path))[0] + "_segmentation.png"
    mask_path = os.path.join("data/ISIC2018_Task1/traindataset/masks", mask_file)
    if os.path.exists(mask_path):
        cmd = f'cp {mask_path} {os.path.join(outdir,"testdataset","masks")}'
        os.system(cmd)
    else:
        print("not exist {}, check again".format(mask_path))