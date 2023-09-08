import segmentation_models_pytorch as smp
import torch
from torch.nn import functional as F
from torch import nn
from utils import dice_channel_torch
from loss import DiceBCELoss

class CustomUnet(nn.Module):
    def __init__(self, backbone='timm-efficientnet-b0', pad=False):
        super(CustomUnet, self).__init__()
        num_class = 1
        in_channels = 1
        # self.model = smp.UnetPlusPlus(backbone, encoder_weights="imagenet", classes=num_class, in_channels=in_channels)
        self.model = smp.DeepLabV3Plus(encoder_name=backbone, encoder_weights="imagenet", classes=num_class, in_channels=in_channels)


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # if self.zero_pad:
        #     x = self.zero_pad(x)
        features = self.model(x)
        # cosine = torch.matmul(F.normalize(features).permute(0, 2, 3, 1), F.normalize(self.weight))
        # return cosine.permute(0, 3, 1, 2)

        return features

if __name__ == '__main__':
    model = CustomUnet(pad=True)
    sample = torch.randn(16, 1, 80, 320)
    output = model(sample)
    print(output.shape)

    # labels = torch.randint(0, 1, (32, 3, 256, 256)).float()
    # labels = torch.rand((32, 3, 256, 256)) > 0.5
    # labels = labels.float()
    # labels = F.one_hot(y.long(), num_classes=3).float()#.permute(0, 3, 1, 2)
    # loss_fn = DiceBCELoss()
    # loss = loss_fn(output, labels)
    # print('loss : ', loss)

    # dice_score = dice_channel_torch(output.sigmoid(), labels, threshold=0.5)
    # print('dice score : ', dice_score)