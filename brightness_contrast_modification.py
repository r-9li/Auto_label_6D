import cv2
import os
from tqdm import trange
import numpy as np

dataset_path = '/media/r/T7 Shield/lm_2/test/000001/rgb'
image_list = os.listdir(dataset_path)
contrast = 1.5
brightness = 30
for i in trange(len(image_list)):
    image_path = os.path.join(dataset_path, image_list[i])
    img = cv2.imread(image_path).astype(np.float32)
    bri_mean = np.mean(img)
    img = contrast * (img - bri_mean) + brightness + bri_mean
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(image_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
