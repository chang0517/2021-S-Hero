import cv2
import numpy as np
import colorlover as cl
import numpy as np
import torch
import wandb

from pytorch_lightning import Callback



# Custom Callback
class ImagePredictionLogger(Callback):
    def __init__(self, val_sample_image, val_sample_mask):
        super().__init__()
        self.val_imgs = val_sample_image 
        self.val_mask = val_sample_mask
        self.category_names = ['Backgroud', 'scratch']

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_mask.to(device=pl_module.device)
        # Get model prediction
        val_preds = pl_module(val_imgs)

        val_preds = val_preds.sigmoid() >= 0.3
        val_preds = val_preds.int()[:,0]

        log_list = list()
        for val_img, val_pred, val_label in zip(val_imgs, val_preds, val_labels):
            show_log = self.wandb_seg_image(val_img.detach().cpu().numpy(), val_pred.detach().cpu().numpy(), val_label.detach().cpu().numpy())
            log_list.append(show_log)

        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":log_list
            }, commit=False) # , commit=False

    def wandb_seg_image(self, image, pred_mask, true_mask):

        return wandb.Image(image, masks={
        "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
        "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})


    def labels(self):
        l = {}
        for i, label in enumerate(self.category_names):
            l[i] = label
        return l



def load_n_get_patch(img_tensor):
    '''
    img_tensor : pytorch type tensor
    '''
    # Create patches
    kc, kh, kw = 3, 100, 100  # kernel size
    dc, dh, dw = 3, 100, 100  # stride
    patches = img_tensor.unfold(0, kw, dw).unfold(1, kh, kw)
    
    # for channel last
    patches_img = patches.contiguous().view(-1, 3, kh, kw).permute(0, 2, 3, 1)
    
    return patches_img

def dice_channel_torch(probability, truth, threshold=0.5):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                if channel_num == 1:
                    channel_dice = dice_single_channel(probability[i], truth[i], threshold)
                else:
                    channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)
                mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel


def dice_single_channel(probability, truth, threshold, eps = 1E-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def mask_to_contours(image, mask_layer, color):
    """ converts a mask to contours using OpenCV and draws it on the image
    """

    # https://docs.opencv.org/4.1.0/d4/d73/tutorial_py_contours_begin.html
    # _, image_binary = cv2.threshold(mask_layer,  45, 255, cv2.THRESH_TOZERO)
    contours, hierarchy = cv2.findContours(mask_layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, color, 2)
        
    return image

def visualise_mask(image, mask):
    """ open an image and draws clear masks, so we don't lose sight of the 
        interesting features hiding underneath 
    """
    # see: https://plot.ly/ipython-notebooks/color-scales/
    colors = cl.scales['3']['qual']['Set3']
    labels = np.array(range(1,4))
    palette = dict(zip(labels, np.array(cl.to_numeric(colors))))

    # reading in the image

    # going through the 4 layers in the last dimension 
    # of our mask with shape (256, 1600, 4)
    for index in range(mask.shape[-1]):
        
        # indeces are [0, 1, 2, 3], corresponding classes are [1, 2, 3, 4]
        label = index + 1
        
        # add the contours, layer per layer 
        image = mask_to_contours(image, mask[:,:,index], color=palette[label])   
        
    return image

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1).contiguous()
        p_sum = p.sum(-1).contiguous()
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

#         dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
#         dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
#         dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


# RLE encoding decoding
def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    Source: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask2rle(img):
    '''
    Efficient implementation of mask2rle, from @paulorzp
    --
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    Source: https://www.kaggle.com/xhlulu/efficient-mask2rle
    '''
    pixels = img.T.flatten()
    pixels = np.pad(pixels, ((1, 1), ))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)