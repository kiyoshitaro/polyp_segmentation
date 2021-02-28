

from dataloader import  get_loader
from dataloader.augment import Augmenter
import tqdm
import torch

def main():


    X_test, y_test, model_path = CONFIG(...)
    # DATASET
    test_transform = Augmenter(prob=0,
                    blur_prob=0,
                    jitter_prob=0,
                    rotate_prob=0,
                    flip_prob=0,
                    )
    test_loader = get_loader(X_test, y_test, 1, img_size, test_transform, shuffle=False, pin_memory=True, drop_last=True)
    test_size = len(test_loader)


    # MODEL 
    import network.models as models
    model = models.__dict__[arch]()
    try:
        model.load_state_dict(torch.load(model_path)["model_state_dict"])
    except RuntimeError:
        model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    print('TESTING ' + test_fold)

    # from util.metrics import  
    from util.visualize import save_img
    gts = []
    prs = []
    tp_all = 0
    fp_all = 0
    fn_all = 0  

    mean_precision= 0
    mean_recall= 0
    mean_iou= 0
    mean_dice= 0

    for i, pack in tqdm.tqdm(enumerate(test_loader, start=1)):
        image, gt, filename, img = pack
        name = os.path.splitext(filename[0])[0]
        ext = os.path.splitext(filename[0])[1]
        gt = gt[0][0]
        gt = np.asarray(gt, np.float32)
        res2 = 0
        image = image.cuda()

        res5, res4, res3, res2 = model(image)


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


        pr = res.round()

        prs.append(pr)
        gts.append(gt)
        tp = np.sum(gt * pr)
        fp = np.sum(pr) - tp
        fn = np.sum(gt) - tp
        tp_all += tp
        fp_all += fp
        fn_all += fn

        mean_precision += precision_m(gt, pr)
        mean_recall += recall_m(gt, pr)
        mean_iou += jaccard_m(gt, pr)
        mean_dice += dice_m(gt, pr)


    mean_precision /= len(test_loader)
    mean_recall /= len(test_loader)
    mean_iou /= len(test_loader)
    mean_dice /= len(test_loader)        
    print("scores ver1: {:.3f} {:.3f} {:.3f} {:.3f}".format(mean_iou, mean_precision, mean_recall, mean_dice))



    precision_all = tp_all / ( tp_all + fp_all + K.epsilon())
    recall_all = tp_all / ( tp_all + fn_all + K.epsilon())
    dice_all = 2*precision_all*recall_all/(precision_all+recall_all)
    iou_all = recall_all*precision_all/(recall_all+precision_all-recall_all*precision_all)
    print("scores ver2: {:.3f} {:.3f} {:.3f} {:.3f}".format(iou_all, precision_all, recall_all, dice_all))

    return gts, prs

if __name__ == "__main__":
    main()