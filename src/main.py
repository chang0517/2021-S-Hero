import pandas as pd
import numpy as np
import os
from pycocotools.coco import COCO

from torch.utils.data import DataLoader
import random
import torch
import torch.nn.functional as F
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint#, StochasticWeightAveraging
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from typing import Optional

from sklearn.model_selection import train_test_split#, StratifiedKFold
from utils import dice_channel_torch, metric, ImagePredictionLogger
from loss import DiceBCELoss, DiceLoss, TverskyLoss, ComboLoss, FocalLoss
from segmentation_models_pytorch import losses
from lr_scheduler import CosineAnnealingWarmUpRestarts

from bmdataset import CustomDataset
from timm_unet import CustomUnet
from trans import get_transforms

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LitClassifier(pl.LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(
        self,
        scale_list = [0.25, 0.5], # 0.125, 
        backbone: Optional[CustomUnet] = None,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        if backbone is None:
            backbone = CustomUnet()
        self.backbone = backbone
        # self.criterion = losses.DiceLoss(mode='multiclass')
        # self.criterion = TverskyLoss()#.to(self.device)
        self.criterion = ComboLoss(nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10])), FocalLoss())#.to(self.device)
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]))

        # self.criterion = DiceBCELoss()#.to(self.device)
        self.split_patch = True
    def forward(self, batch):
        output = self.backbone.model(batch)
            
        return output

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        if self.split_patch:
            x = x.view(-1, 1, 256, 256)
            y = y.view(-1, 256, 256)
        output = self.backbone(x)
        # labels = F.one_hot(y.long(), num_classes=2).float().permute(0, 3, 1, 2)
        # labels = y.long().contiguous() # .float().unsqueeze(1)

        loss = self.criterion(output, y)

        try:
            dice_score = dice_channel_torch(y.detach().cpu(), output.detach().cpu().sigmoid())

            self.log("dice_score", dice_score, on_step= True, prog_bar=True, logger=True)
            self.log("Train Loss", loss, on_step= True,prog_bar=True, logger=True)
        
        except:
            pass

        return {"loss": loss, "predictions": output.detach().cpu(), "labels": y.detach().cpu()}

    def training_epoch_end(self, outputs):

        preds = []
        labels = []
        
        for output in outputs:
            
            preds += output['predictions']#.detach().cpu().sigmoid()
            labels += output['labels']#.detach().cpu()

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        # dice_score, dice_neg, dice_pos, num_neg, num_pos = metric(preds.sigmoid(), labels)

        # self.log("dice_score", dice_score, prog_bar=True, logger=True)
        # self.log("dice_neg", dice_neg, prog_bar=True, logger=True)
        # self.log("dice_pos", dice_pos, prog_bar=True, logger=True)
        # self.log("num_neg", num_neg, prog_bar=False, logger=True)
        # self.log("num_neg", num_neg, prog_bar=False, logger=True)
        # self.log("num_pos", num_pos, prog_bar=False, logger=True)

        dice_score = dice_channel_torch(labels, preds)
        self.log("mean_dice_score", dice_score, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.split_patch:
            x = x.view(-1, 1, 256, 256)
            y = y.view(-1, 256, 256)
        output = self.backbone(x)
        # labels = F.one_hot(y.long(), num_classes=2).float().permute(0, 3, 1, 2)#.contiguous()
        # labels = y.float().unsqueeze(1)
        # labels = y.long().contiguous() # .float().unsqueeze(1)

        loss = self.criterion(output, y)
        
        self.log('val_loss', loss, on_step= True, prog_bar=True, logger=True)
        return {"predictions": output.detach().cpu(), "labels": y.detach().cpu()}

    def validation_epoch_end(self, outputs):

        preds = []
        labels = []
        for output in outputs:
            preds += output['predictions']
            labels += output['labels']

        preds = torch.stack(preds)
        labels = torch.stack(labels)

        # onehot = F.one_hot(labels.long(), num_classes=3).permute(0, 3, 1, 2)# .contiguous()
        val_dice_score = dice_channel_torch(labels.detach().cpu(), preds.detach().cpu().sigmoid())
        self.log("val_dice_score", val_dice_score, prog_bar=True, logger=True)
        
        # val_dice_score, val_dice_neg, val_dice_pos, val_num_neg, val_num_pos = metric(preds.sigmoid(), labels)
        # self.log("val_dice_score", val_dice_score, prog_bar=True, logger=True)
        # self.log("val_dice_neg", val_dice_neg, prog_bar=True, logger=True)
        # self.log("val_dice_pos", val_dice_pos, prog_bar=True, logger=True)
        # self.log("val_num_neg", val_num_neg, prog_bar=False, logger=True)
        # self.log("val_num_neg", val_num_neg, prog_bar=False, logger=True)
        # self.log("val_num_pos", val_num_pos, prog_bar=False, logger=True)


    def test_step(self, batch, batch_idx):
        out = self.backbone(batch)
        return out.sigmoid()

    def configure_optimizers(self):

        param_optimizer = list(self.backbone.named_parameters()) # self.model.named_parameters()
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-6,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.hparams.learning_rate)
        scheduler_cosie = CosineAnnealingLR(optimizer, T_max= 10, eta_min=1e-6, last_epoch=-1)
        # scheduler_cosie = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.01,  T_up=2, gamma=0.5)
        # scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cosie)
        return dict(optimizer=optimizer, lr_scheduler=scheduler_cosie) # , lr_scheduler=scheduler_warmup lr_scheduler=scheduler[optimizer], [scheduler]

class MyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 2,
    ):
        super().__init__()

        cc = COCO('../data/seg_data/crop_bm_v2.json') # crop_bm-1
        img_ids = cc.getImgIds()
        train_img_ids, valid_img_ids = train_test_split(img_ids, test_size=0.2, shuffle=True, random_state=52)

        trn_dataset = CustomDataset(cc, train_img_ids, transform=get_transforms(data='train'), split_patch=True) # , feat_df=TRAIN_FEAT_DF
        val_dataset = CustomDataset(cc, valid_img_ids, transform=get_transforms(data='valid'), split_patch=True) # , feat_df=TRAIN_FEAT_DF

        self.train_dset = trn_dataset
        self.valid_dset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_dset, batch_size=self.batch_size, shuffle=False, num_workers=8)

def cli_main():
    logger = WandbLogger(name=f'weight BCE with focal', project='BM_seg_unet')
    classifier =  LitClassifier()
    mc = ModelCheckpoint('model', monitor='val_dice_score', mode='max', filename='{epoch}-{val_dice_score:.4f}_')
    # swa = StochasticWeightAveraging(swa_epoch_start=2, annealing_epochs=2)
    mydatamodule = MyDataModule()
    val_img, val_mask = next(iter(mydatamodule.val_dataloader()))
        
    
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=30,
        # stochastic_weight_avg=True,
        callbacks=[
            mc,
            ImagePredictionLogger(val_img[0], val_mask[0])
        ],
        logger=logger
        )
    trainer.fit(classifier, datamodule=mydatamodule)

if __name__ == '__main__':

    seed_everything()
    cli_main()

