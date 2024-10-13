import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


PALETTE = [[10, 10, 10], [230, 5, 5],[4, 200, 3], [204, 5, 255], [5, 128, 148]] #dataset.METAINFO['palette']
CLASSES = ('background','rigid_plastic', 'cardboard', 'metal', 'soft_plastic') # dataset.METAINFO['classes']

path = './datasets/zerowaste-f-final/splits_final_deblurred/test/sem_seg'
path_img = './datasets/zerowaste-f-final/splits_final_deblurred/test/data'

lst = sorted(os.listdir(path))

for iname in lst:
    print(iname)
    img = cv2.cvtColor(cv2.imread(os.path.join(path_img, iname)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(path, iname))

    mm = mask[:,:,0]

    res = img.copy()

    for ii in range(img.shape[0]):
        for jj in range(img.shape[1]):
            if mask[ii,jj,0] == 0:
                res[ii, jj] = img[ii, jj] * 0.5 + np.array(PALETTE[0]) * 0.5

            elif mask[ii,jj,0] == 1:
                res[ii, jj] = img[ii, jj] * 0.5 + np.array(PALETTE[1]) * 0.5

            elif mask[ii,jj,0] == 2:
                res[ii, jj] = img[ii, jj] * 0.5 + np.array(PALETTE[2]) * 0.5

            elif mask[ii,jj,0] == 3:
                res[ii, jj] = img[ii, jj] * 0.5 + np.array(PALETTE[3]) * 0.5
    
            elif mask[ii,jj,0] == 4:
                res[ii, jj] = img[ii, jj] * 0.5 + np.array(PALETTE[4]) * 0.5

    res = cv2.cvtColor(np.uint8(res), cv2.COLOR_RGB2BGR)
    cv2.imwrite('./gt_zerowaste/'+iname, res)

