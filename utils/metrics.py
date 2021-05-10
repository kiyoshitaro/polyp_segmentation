import numpy as np

# from keras import backend as K

mean_precision = 0
mean_recall = 0
mean_iou = 0
mean_dice = 0


def recall_m(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 1e-07)
    return recall


def precision_m(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 1e-07)
    return precision


def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + 1e-07))


def jaccard_m(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-07)


def dice_score(o, t, eps=1e-8):
    num = 2 * (o * t).sum() + eps  #
    den = o.sum() + t.sum() + eps  # eps
    # print(o.sum(),t.sum(),num,den)
    print(
        "All_voxels:240*240*155 | numerator:{} | denominator:{} | pred_voxels:{} | GT_voxels:{}".format(
            int(num), int(den), o.sum(), int(t.sum())
        )
    )
    return num / den


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0
    t = target > 0  # ce
    ret += (dice_score(o, t),)
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    ret += (dice_score(o, t),)
    # active
    o = output == 3
    t = target == 3
    ret += (dice_score(o, t),)

    return ret


def get_scores_v1(gts, prs, log):
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for gt, pr in zip(gts, prs):
        mean_precision += precision_m(gt, pr)
        mean_recall += recall_m(gt, pr)
        mean_iou += jaccard_m(gt, pr)
        mean_dice += dice_m(gt, pr)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)

    log.info(
        f"scores ver1: miou={mean_iou}, dice={mean_dice}, precision={mean_precision}, recall={mean_recall}"
    )
    return (mean_iou, mean_dice, mean_precision, mean_recall)


def get_scores_v2(gts, prs, log):
    tp_all = 0
    fp_all = 0
    fn_all = 0
    for gt, pr in zip(gts, prs):
        tp = np.sum(gt * pr)
        fp = np.sum(pr) - tp
        fn = np.sum(gt) - tp
        tp_all += tp
        fp_all += fp
        fn_all += fn

    precision_all = tp_all / (tp_all + fp_all + 1e-07)
    recall_all = tp_all / (tp_all + fn_all + 1e-07)
    dice_all = 2 * precision_all * recall_all / (precision_all + recall_all)
    iou_all = (
        recall_all
        * precision_all
        / (recall_all + precision_all - recall_all * precision_all)
    )

    log.info(
        f"scores ver2: miou={iou_all}, dice={dice_all}, precision={precision_all}, recall={recall_all}"
    )

    return (iou_all, dice_all, precision_all, recall_all)
