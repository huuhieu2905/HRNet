import cv2
from utils_inference import get_lmks_by_img, get_model_by_name
import numpy as np


model = get_model_by_name(hrnet_model=False, device='cuda')
path = 'data/300w/images/afw/815038_1.jpg'
get_name = path.split('/')[-1]
print(get_name)
img = cv2.imread(path)
lmks = get_lmks_by_img(model, img)
for lmk in lmks:
    cv2.circle(img, (int(lmk[0]), int(lmk[1])), 2, (255,255,0), 2, cv2.LINE_AA)
    
cv2.imwrite(f'test.jpg', img)

