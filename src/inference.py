import numpy as np
from main import LitClassifier
import torch
import cv2


def do_center_crop(img, cropx, cropy):
    y,x = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[starty:starty+cropy, startx:startx+cropx]

def split_image2patch(img, patch_gird=4):
    h, w = img.shape
    p_h, p_w = int(h / patch_gird), int(w / patch_gird)
    h_slice_img = np.split(img, patch_gird, axis=0)
    h_slice_img = np.array(h_slice_img)
    
    slice_img = np.split(h_slice_img, patch_gird, axis=2)
    slice_img = np.array(slice_img)
    slice_img = slice_img.reshape(-1, p_h, p_w)
    return slice_img

def stack_patch2img(imgs):

    hs_img_list = list()
    for i in range(0, 16, 4):
        hs_img = np.hstack(imgs[i : i+ 4])
        hs_img_list.append(hs_img)
    img = np.vstack(hs_img_list)
    return img


def load_model_pl(path):
    model = LitClassifier()
    model.load_from_checkpoint(path)
    return model


def inference(path, image_array):
    '''
    path : trained model path
    image_array : image array
    '''
    # doing center crop
    img = do_center_crop(image_array, 1900, 1900).astype(np.float32)
    img = cv2.resize(img, (1024, 1024))
    img /= 255

    # split 
    patches = split_image2patch(img, patch_gird=4)
    # cv2.imwrite('../temp_img/patch_0.jpg', patches[8] * 255)
    patches = torch.from_numpy(patches)
    patches = patches.unsqueeze(dim=1)

    model = load_model_pl(path)
    model = model.cuda().eval()

    with torch.no_grad():
        output_patches = model(patches.cuda())

    return output_patches.detach().cpu()

def make_dicision(output_patches, th=0.5):
    
    # output_patches shape : 16, 3, 256, 256
    tf_mask = output_patches >= 0.5
    

    return None


if __name__ == '__main__':

    model_path = 'model/epoch=41-val_dice_score=0.9286_.ckpt'
    img_path = '../data/selected_data/15_33_07/02.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    output = inference(model_path, img)
    
    print(output.shape)
    # print(output)
    # sample = np.random.randn(16, 32, 32, 3)
    # one_img = stack_patch2img(sample)
    # print(one_img.shape)