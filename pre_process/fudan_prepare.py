import json
import os

import cv2
import h5py
import numpy as np
import scipy
import scipy.ndimage


def gaussian_filter_density(gt, pts):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = len(pts)
    if gt_count == 0:
        return density
    # print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[min(int(round(pt[1])), gt.shape[0] - 1), min(int(round(pt[0])), gt.shape[1] - 1)] = 1.
        if gt_count >= 1:
            # print(round(pt[1]))
            # sigma = (round(pt[1])/50+1)
            sigma = 15
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    # print('done.')
    return density


if __name__ == "__main__":
    print("haha")
    for root, dirs, files in os.walk("/home/pp/FDST_dataset/test_data/", topdown=False):
        for name in dirs:
            for i in range(1, 151):
                path = os.path.join(root, name, "%03d.jpg" % (i))
                with open(path.replace('jpg', 'json'), 'r') as load_f:
                    load_dict = json.load(load_f)
                regions = load_dict.popitem()[1]['regions']
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                pts = []
                for i in regions:
                    x = i['shape_attributes']['x']
                    y = i['shape_attributes']['y']
                    width = i['shape_attributes']['width']
                    height = i['shape_attributes']['height']
                    pts.append((x + width / 2, y + height / 2))
                density = gaussian_filter_density(img, pts)
                with h5py.File(path.replace('jpg', 'h5'), 'w') as hf:
                    hf['density'] = density
                print(path)
