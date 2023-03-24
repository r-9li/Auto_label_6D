import cv2
import os
from tqdm import trange
import numpy as np

dataset_path = '/media/r/T7 Shield/Auto_label_6D-master/Dataset/train'
folder_list = ['000001', '000004', '000006', '000009', '000011', '000014', '000016', '000019', '000021', '000024',
               '000026', '000029', '000031', '000034', '000036', '000039', '000041', '000044', '000046', '000049',
               '000051', '000054', '000056', '000059', '000061', '000064', '000066', '000069', '000071', '000074',
               '000076', '000079']
contrast = 1.2
brightness = 36
for folder in folder_list:
    print(folder)
    folder_path = os.path.join(dataset_path, folder, 'rgb')
    image_list = os.listdir(folder_path)

    for i in trange(len(image_list)):
        image_path = os.path.join(folder_path, image_list[i])
        img = cv2.imread(image_path).astype(np.float32)
        bri_mean = np.mean(img)
        img = contrast * (img - bri_mean) + brightness + bri_mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(image_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
