import numpy as np
import cv2
import os
from torch.functional import split
from torch.utils.data import Dataset, DataLoader, dataloader
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

from trans import get_transforms
import matplotlib.pyplot as plt

category_names = ['Backgroud', 'scratch', 'word_error']
def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

def view_patches(patches_img):
    fig, axs = plt.subplots(4,4, figsize=(15, 15), facecolor='w', edgecolor='k')
    # fig.subplots_adjust(hspace = .5, wspace=.001)

    axs = axs.ravel()

    for i in range(len(patches_img)):

        axs[i].imshow(patches_img[i].permute(1,2,0))
    #     axs[i].set_title(str(i))
        axs[i].axes.xaxis.set_visible(False)
        axs[i].axes.yaxis.set_visible(False)
    fig.savefig('test_pat.jpg')

class CustomDataset(Dataset):
    """COCO format"""
    def __init__(self, coco, img_ids=None, mode='train', transform=None, split_patch=False):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = coco
        
        if img_ids is None:
            self.img_ids = list(self.coco.getImgIds())
        else:
            self.img_ids = list(img_ids)
        
        self.split_patch = split_patch
        self.dataset_path = '../data/seg_data/all'
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=self.img_ids[index])
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']), cv2.IMREAD_GRAYSCALE)#.astype(np.float32)
        # images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                # if className == 'word_error':
                #     print(index)
                #     print(className)
                pixel_value = category_names.index(className)
                if pixel_value == 2:
                    pixel_value = 1
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)
                        
            
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"] # -> C, H, W
                masks = transformed["mask"] # -> H, W
            if self.split_patch:
                images = self.do_split_img_fatch(images)
                masks = self.do_split_mask_fatch(masks)
            images = images / 255.
            return images, masks
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
                test = thisdict = {
                                      "brand": 1,
                                      "model": 0,
                                      "year": 1
                                    }
                image_infos = str(image_infos)
                print(image_infos)
            return images, test

    def do_split_img_fatch(self, obj):
        kc, kh, kw = 1, 256, 256  # kernel size
        dc, dh, dw = 1, 256, 256  # stride
        obj = obj.permute(1,2,0)
        obj = obj.unfold(0, kw, dw).unfold(1, kh, kw)
        # print('split patch : ', obj.contiguous().view(-1, 3, 256, 256).shape)
        return obj.contiguous().view(-1, 1, 256, 256)
    
    def do_split_mask_fatch(self, obj):
        kc, kh, kw = 1, 256, 256  # kernel size
        dc, dh, dw = 1, 256, 256  # stride
        obj = obj.unfold(0, kw, dw).unfold(1, kh, kw)
        return obj.contiguous().view(-1, 256, 256)

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.img_ids)



if __name__ == '__main__':
    dataset_path = '../data/seg_data/all'

    dset = CustomDataset('../data/seg_data/crop_bm-1.json', transform=get_transforms(data='train'), split_patch=True)
    dloader = DataLoader(dset, batch_size=1)
    x, y = next(iter(dloader))
    print(x.shape)
    print(y.shape)