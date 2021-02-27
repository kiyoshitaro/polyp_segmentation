from skimage.io import imread

import glob
train_mask = glob.glob("/Users/brown/code/PraNet/TrainDataset/mask/*")
test_mask = glob.glob("/Users/brown/code/PraNet/TestDataset/masks/*")
for i in test_mask:
    if(255 not in set(imread(i).reshape((-1,)))):
        print(i)
        print(set(imread(i).reshape((-1,))))

import os
for id in range(5):
    fold = [os.path.basename(i) for i in glob.glob(f'Kvasir_fold_new/fold_{id}/masks/*')]
    os.system(f'mkdir Kvasir_fold_new/fold_{id}/_masks/')
    for i in fold:
        cmd = f'cp all_masks/{i} Kvasir_fold_new/fold_{id}/_masks/'
        os.system(cmd)


def check_type_image(path):
    img = imread(path,as_gray=True)
    print("shape",img.shape)
    print("value",set(img.reshape((-1,))))
    