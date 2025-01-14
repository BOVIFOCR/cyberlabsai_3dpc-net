import os, sys
import torch
from torchvision import transforms
from PIL import ImageDraw

def add_visualization_to_tensorboard(cfg, epoch, img_batch, preds, targets, score, writer):
    """ Do the inverse transformation
    x = z*sigma + mean
      = (z + mean/sigma) * sigma
      = (z - (-mean/sigma)) / (1/sigma),
    Ref: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/6
    """
    mean = [-cfg['dataset']['mean'][i] / cfg['dataset']['sigma'][i] for i in range(len(cfg['dataset']['mean']))]
    sigma = [1 / cfg['dataset']['sigma'][i] for i in range(len(cfg['dataset']['sigma']))]
    img_transform = transforms.Compose([transforms.Normalize(mean, sigma),transforms.ToPILImage()])

    ts_transform = transforms.ToTensor()

    for idx in range(img_batch.shape[0]):
        vis_img = img_transform(img_batch[idx].cpu())
        ImageDraw.Draw(vis_img).text((0,0), 'pred: {} vs gt: {}'.format(int(preds[idx]), int(targets[idx])), (255,0,255))
        ImageDraw.Draw(vis_img).text((20,20), 'score {}'.format(score[idx]), (255,0,255))
        tb_img = ts_transform(vis_img)
        writer.add_image('Prediction visualization/{}'.format(idx), tb_img, epoch)


def predict(point_Cloud_label, threshold=0.5):
    with torch.no_grad():
        
        score = torch.mean(point_Cloud_label, dim=(1, 2)) # Make sure the mean is trough each axis independently
        # score = torch.mean(point_Cloud_label, dim=(2))
        # score = score[:, 2]
        # score = torch.mean(score, 1)
        preds = (score > threshold) # .type(torch.FloatTensor)

        return preds, score
#==========================================================================
def calc_accuracy(preds, targets):
    """
    Compare preds and targets to calculate accuracy
    Args
        - preds: batched predictions
        - targets: batched targets
    Return
        a single accuracy number
    """
    with torch.no_grad():
        equals = torch.mean(preds.eq(targets).type(torch.FloatTensor))
        return equals.item()


def calc_tp_tn_fp_fn(preds, targets):
    with torch.no_grad():
        # equals = torch.mean(preds.eq(targets).type(torch.FloatTensor))
        # return equals.item()

        # predict_issame = np.less(dist, threshold)
        tp = torch.sum(torch.logical_and(preds, targets))
        fp = torch.sum(torch.logical_and(preds, torch.logical_not(targets)))
        tn = torch.sum(torch.logical_and(torch.logical_not(preds), torch.logical_not(targets)))
        fn = torch.sum(torch.logical_and(torch.logical_not(preds), targets))
        # print('tp:', tp, '    fp:', fp, '    tn:', tn, '    fn:', fn)
        # sys.exit(0)
        return tp, fp, tn, fn


# https://drive.google.com/file/d/1krZTtIHS5fEOTDzCOaN69QtQt2nK2D_9/view
# https://chalearnlap.cvc.uab.cat/challenge/33/track/33/metrics/
# https://github.com/ZitongYu/CDCN/blob/master/CVPR2020_paper_codes/utils.py
def calc_fas_metrics(tp, fp, tn, fn):
    with torch.no_grad():
        metrics = {}
        metrics['apcer'] = fp / (tn + fp)
        metrics['bpcer'] = fn / (tp + fn)
        metrics['acer'] = (metrics['apcer'] + metrics['bpcer']) / 2.0
        return metrics
