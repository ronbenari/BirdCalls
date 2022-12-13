import numpy as np
from scipy import stats
from sklearn import metrics
import torch

### Modified for birds project
ron_pavel_birds = True

# Ron: Don't show divide by zero errors
np.seterr(divide='ignore')

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target, label_mask=None, no_target_stats=False):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))
    if label_mask is None:
        birds_acc = (target == (output > 0.5).int()).float().mean()
        gt_positive_indexes = np.where(target == 1)[0]
        pred_positive_indexes = np.where(output > 0.5)[0]
    else:
        label_mask = label_mask.to(output.device)
        birds_acc = ((target * label_mask) == ((output * label_mask) > 0.5).int()).float().mean()
        gt_positive_indexes = np.where((target*label_mask) == 1)[0]
        pred_positive_indexes = np.where((output*label_mask) > 0.5)[0]
    birds_precision = target[pred_positive_indexes].sum() / len(pred_positive_indexes)
    birds_recall = (output[gt_positive_indexes] > 0.5).sum() / len(gt_positive_indexes)
    print(f'target[pred_positive_indexes].sum()={target[pred_positive_indexes].sum()}')
    print(f'len(pred_positive_indexes)={len(pred_positive_indexes)}')
    print(f'(output[gt_positive_indexes] > 0.5).sum()={(output[gt_positive_indexes] > 0.5).sum()}')
    print(f'len(gt_positive_indexes)={len(gt_positive_indexes)}')


    # Class-wise statistics
    raised_exception_list = []
    for k in range(classes_num):
        # print(f'*** @calculate_stats k={k}')

        # Average precision
        # Ron: Added check that target is not all zeros
        if target[:, k].sum() > 0:
            avg_precision = metrics.average_precision_score(target[:, k], output[:, k], average=None)
        else:
            avg_precision = 0

        # AUC
        # Ron: Added check that target is not all zeros and no_target_stats option
        if (target[:, k].sum() > 0) and not no_target_stats:
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)
        else:
            auc = 0

        # Precisions, recalls
        # Ron: Added check that target is not all zeros and no_target_stats option
        n_samples = len(target[:, k])
        if (target[:, k].sum() > 0) and not no_target_stats:
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(target[:, k], output[:, k])
        else:
            (precisions, recalls, thresholds) = (np.zeros(n_samples), np.zeros(n_samples), np.zeros(1))

        # FPR, TPR
        # Ron: Added check that target is not all zeros and no_target_stats option
        if (target[:, k].sum() > 0) and not no_target_stats:
            (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])
        else:
            (fpr, tpr, thresholds) = (np.zeros(3), np.zeros(3), np.zeros(3))

        save_every_steps = 1000     # Sample statistics to reduce size
        res_dict = {
                'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc,
                'birds_acc': birds_acc,
                'birds_precision': birds_precision,
                'birds_recall': birds_recall
                }
        stats.append(res_dict)
    # print(f'AUC error in {raised_exception_list}')

    return stats

