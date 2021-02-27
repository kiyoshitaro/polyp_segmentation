import torch
import os
import torch.nn.functional as F
import cv2
from glob import glob
import tqdm
from skimage.io import imread
import numpy as np
from keras import backend as K
import skimage

from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations import transforms
def jaccard_m(y_true, y_pred):
  intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
  union = np.sum(y_true)+np.sum(y_pred)-intersection
  return intersection/(union+K.epsilon())
dices = []
ious = []
precisions = []
recalls = []

class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image_ = imread(img_path)
        mask = imread(mask_path)
        # image = cv2.resize(image_, (352, 352))
        if self.aug:
            augmented = self.transform(image=image_, mask=mask)
            image = augmented['image']
            # mask = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        
        mask = mask.astype('float32')
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask), os.path.basename(img_path), np.asarray(image_)

def save_img(path, img, lib = "cv2", overwrite = True):
    if(not overwrite and os.path.exists(path)):
        pass
    else:
        print(path)
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if(lib == "skimage"):
            skimage.io.imsave(path, img)
        elif(lib == "cv2"):
            cv2.imwrite(path,img)


def test(v,model,folds,visualize = False):
    gts = []
    prs = []

    for id in list(folds.keys()):
        test_fold = 'fold' + str(id)
        _data_name = 'Kvasir'
        data_path = 'Kvasir_fold_new/' + 'fold_' + str(id)
        save_path = 'results/PraNet_kfold/'
        model_path = './snapshots/PraNetv'+str(v)+'_Res2Net_kfold/' + 'PraNetDG-' + test_fold +'-'+str(folds[id])+'.pth'
        print(model_path)
        try:
          model.load_state_dict(torch.load(model_path)["model_state_dict"])
        except RuntimeError:
          model.load_state_dict(torch.load(model_path))

        model.cuda()
        model.eval()

        os.makedirs(save_path + test_fold, exist_ok=True)

        X_test = glob('{}/images/*'.format(data_path))
        X_test.sort()
        y_test = glob('{}/masks/*'.format(data_path))
        y_test.sort()
        test_transform = Compose([
        transforms.Resize(352,352),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

        test_dataset = Dataset(X_test, y_test, aug=True, transform = test_transform)
        test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
        print('TESTING ' + test_fold)

        tp_all = 0
        fp_all = 0
        fn_all = 0  
        mean_iou = 0

        for i, pack in tqdm.tqdm(enumerate(test_loader, start=1)):
            image, gt, filename, img = pack
            name = os.path.splitext(filename[0])[0]
            ext = os.path.splitext(filename[0])[1]

            # if(os.path.exists(os.path.join(save_path,test_fold,"v" + str(v),name+"_prv" + str(v) + ext))):
            #     continue
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)
            res2 = 0
            image = image.cuda()
            if(v == 0 or v == 1 or v ==2 or v ==3):
                res5, res4, res3, res2 = model(image)
            elif(v ==4 or v==5 or v==13 or v=="GALD" or v==14 or v==15 or v==16 or v==17 or v==18):
                res_head_out, res5, res4, res3, res2 = model(image)
            elif(v==6 or v==7):
                res_gald_head_out, res_dual_head_out, res5, res4, res3, res2 = model(image)
            elif(v==8):
                res4_head_out, res3_head_out, res2_head_out, res5, res4, res3, res2 = model(image)
            elif(v==12):
                res2 = model(image)
            else:
                print("Not have this version")
                break
            # res = res_head_out
            res = res2
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            if (visualize):
                save_img(os.path.join(save_path,test_fold,"v" + str(v),name+"_prv" + str(v) + ext), res.round()*255, "cv2",False)
                save_img(os.path.join(save_path,test_fold,"soft_v" + str(v),name+"_soft_prv" + str(v) + ext), res*255, "cv2",False)
                # mask_img = np.asarray(img[0]) + cv2.cvtColor(res.round()*60, cv2.COLOR_GRAY2BGR)
                mask_img = np.asarray(img[0]) + np.array((np.zeros_like(res.round()*60) , res.round()*60, np.zeros_like(res.round()*60) )).transpose((1,2,0)) + np.array((gt*60 , np.zeros_like(gt*60), np.zeros_like(gt*60) )).transpose((1,2, 0))
                mask_img = mask_img[:,:,::-1]
                save_img(os.path.join(save_path,test_fold,"mask_v" + str(v),name+"mask_prv" + str(v) + ext), mask_img ,"cv2",False)
                # skimage.io.imsave("aaa.png", np.asarray(img[0]) + cv2.cvtColor(res.round()*60, cv2.COLOR_GRAY2BGR))
                # import sys
                # sys.exit()

                # cv2.imwrite(save_path + test_fold + '/' + name + '_gt' + ext, gt*255)
                # print("cp {} {}".format('Kvasir_fold_new/' + 'fold_' + str(id)+'/images/' + name + ext, save_path + test_fold))
                # os.system("cp {} {}".format('/content/Kvasir_fold_new/' + 'fold_' + str(id)+'/images/' + name + ext, save_path + test_fold ))


            pr = res.round()
            prs.append(pr)
            gts.append(gt)
            tp = np.sum(gt * pr)
            fp = np.sum(pr) - tp
            fn = np.sum(gt) - tp
            tp_all += tp
            fp_all += fp
            fn_all += fn
            if i%100 == 0:
                print(i)

        precision_all = tp_all / ( tp_all + fp_all + K.epsilon())
        recall_all = tp_all / ( tp_all + fn_all + K.epsilon())
        dice_all = 2*precision_all*recall_all/(precision_all+recall_all)
        iou_all = recall_all*precision_all/(recall_all+precision_all-recall_all*precision_all)
        ious.append(iou_all)
        precisions.append(precision_all)
        recalls.append(recall_all)
        dices.append(dice_all)
        print("{:.3f} {:.3f} {:.3f} {:.3f}".format(iou_all, precision_all, recall_all, dice_all))

    # print(f"IoU: mean={round(np.mean(ious), 6)}, std={round(np.std(ious), 6)}, var={round(np.var(ious), 6)}")
    # print(f"dice: mean={round(np.mean(dices), 6)}, std={round(np.std(dices), 6)}, var={round(np.var(dices), 6)}")
    # print(f"precision: mean={round(np.mean(precisions), 6)}, std={round(np.std(precisions), 6)}, var={round(np.var(precisions), 6)}")
    # print(f"recall: mean={round(np.mean(recalls), 6)}, std={round(np.std(recalls), 6)}, var={round(np.var(recalls), 6)}")

    # print(f'iou = {iou_all}, precision = {precision_all}, recall = {recall_all}, dice = {dice_all}')
    return gts, prs


if __name__ == "__main__":

    import os
    import cv2
    import numpy as np
    from keras import backend as K

    from lib.PraNet_Res2Net import PraNetv16, PraNetGALD
    v = 18
    model = PraNetGALD()
    # from test import test
    folds = {
        # 0:98,
        # 1:89,
        # 2:68,
        # 3:95,
        # 4:95
        }
    folds = {
        # 0:98,
        # 1:89,
        # 2:92,
        # 3:92,
        4:89
        }
    for o in range(21,34):
        folds = {0:3*o-1}
        gts, prs = test(v, model, folds = folds, visualize = False)

    # gts, prs = test(v, model, folds = folds, visualize = False)


