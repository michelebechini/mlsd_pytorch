import json
import numpy as np
import cv2
from mlsd_pytorch.data.utils import gen_TP_mask2, gen_SOL_map, gen_junction_and_line_mask
from albumentations import (
    RandomBrightnessContrast,
    OneOf,
    HueSaturationValue,
    Compose,
    Normalize,
    Flip,
    Rotate,
    Affine,
    RandomScale,
    Downscale,
    ColorJitter,
    GaussianBlur,
    GaussNoise,
    Resize,
    LongestMaxSize,
    PadIfNeeded,
    KeypointParams
)
import matplotlib.pyplot as plt


json_path = './data/wireframe_raw/train.json'
img_path = '/Users/michelebechini/GitHub/mlsd_pytorch/data/wireframe_raw/images/'

with open(json_path, 'r') as f:
    variable = json.load(f)

# get the first image json
#img1_data = variable[10]
for i in range(len(variable)):
    if variable[i]['filename'] == '00405957.png':
        img1_data = variable[i]

print('Processing image ' + img1_data['filename'])
keyps = []
for i in range(len(img1_data['lines'])):
    keyps.append([img1_data['lines'][i][0], img1_data['lines'][i][1]])
    keyps.append([img1_data['lines'][i][2], img1_data['lines'][i][3]])


img = cv2.imread(img_path + img1_data['filename'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

aug1 = Compose([
            Flip(p=0.5),  # random flip vertically and/or horizontally the image
            #Rotate(limit=90, interpolation=3, border_mode=1, p=0.5), # rotate with 90° limits with Area interpolation and border reflect
            Affine(shear=[-45, 45], interpolation=3, fit_output=True, mode=1, p=0.5), # shear with area interpolation and border reflect
            RandomScale(scale_limit=0.5, interpolation=3, always_apply=True),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0., p=0.5),
            #GaussianBlur(sigma_limit=75, always_apply=True), # already applied to all images in _aug_test
            #GaussNoise(var_limit=(10, 50), mean=0, always_apply=True) # already applied to all images in _aug_test
        ], p=1, keypoint_params=KeypointParams(format='xy', remove_invisible=False)) # add the keyword for keypoints

# OneOf([
#                 RandomScale(scale_limit=0.5, interpolation=3, p=0.5), # resize the image to smaller (0.5) or bigger (1.5) size
#                 Downscale(scale_min=0.25, scale_max=0.75, interpolation=3, p=0.5)
#                 ], p=0.5),
aug2 = Compose(
            [
                LongestMaxSize(max_size=512, interpolation=1, always_apply=True),
                PadIfNeeded(min_height=512, min_width=512, border_mode=1, always_apply=True),
                GaussianBlur(blur_limit=0, sigma_limit=(1, 1), always_apply=True),
                GaussNoise(var_limit=(0.0022, 0.0022), mean=0, always_apply=True),
                # Normalization is needed if you pretrain on imagenet-like data, otherwise not
                #Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True)
            ],
            p=1.0, keypoint_params=KeypointParams(format='xy', remove_invisible=False))

# show original image
plt.imshow(img)
plt.title('Original Image')
for i in range(0, len(keyps)-1, 2):
    plt.plot([keyps[i][0], keyps[i+1][0]], [keyps[i][1], keyps[i+1][1]], 'go--', linewidth=2)
plt.show()

img_orig = img.copy()
keyps_orig = keyps.copy()

transf1 = aug1(image=img, keypoints=keyps)

img1 = transf1['image']
keyps1 = transf1['keypoints']

transf2 = aug2(image=img1, keypoints=keyps1)

img_mod = transf2['image']
kp_mod = transf2['keypoints']

# show mod image
plt.imshow(img1)
plt.title('MID MOD Image')
for i in range(0, len(keyps1)-1, 2):
    plt.plot([keyps1[i][0], keyps1[i+1][0]], [keyps1[i][1], keyps1[i+1][1]], 'go--', linewidth=2)
plt.show()

# show mod image
plt.imshow(img_mod)
plt.title('FINAL MOD Image')
for i in range(0, len(kp_mod)-1, 2):
    plt.plot([kp_mod[i][0], kp_mod[i+1][0]], [kp_mod[i][1], kp_mod[i+1][1]], 'go--', linewidth=2)
plt.show()

norm_lines = []
for i in range(0, len(kp_mod)-1, 2):
    norm_lines.append([kp_mod[i][0]/2, kp_mod[i][1]/2, kp_mod[i+1][0]/2, kp_mod[i+1][1]/2])

tp_mask = gen_TP_mask2(norm_lines,  h = 256, w = 256, with_ext=False)
centermap = tp_mask[0, :, :]

plt.imshow(centermap)
plt.title('Centermap')
plt.show()

sol_mask, ext_lines = gen_SOL_map(norm_lines,  h =256, w =256, min_len =0.125, with_ext= False)

sol_centers = sol_mask[0, :, :]

plt.imshow(sol_centers)
plt.title('SOL Centermap')
plt.show()


j_map, len_map = gen_junction_and_line_mask(norm_lines, h = 256, w = 256)

plt.imshow(j_map[0])
plt.title('Junction map')
plt.show()

plt.imshow(len_map[0])
plt.title('Length map')
plt.show()
