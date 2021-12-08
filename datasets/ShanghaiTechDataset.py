import glob
import os.path as osp

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ShanghaiTechDataset(Dataset):

    def __init__(self, mode="train", **kwargs):
        self.root = "./shtu_dataset/original/part_B_final/train_data/" if mode == "train" else \
            "./shtu_dataset/original/part_B_final/test_data/"
        self.paths = glob.glob(self.root + "images/*.jpg")
        self.transform = kwargs['transform']
        self.length = len(self.paths)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        path = self.paths[item]
        img, den, den2 = self.load_data(path)
        if self.transform is not None:
            img = self.transform(img)
        mask = den > 0
        exp = lambda x: -(x % 16) if -(x % 16) != 0 else None
        h = den.shape[0]
        w = den.shape[1]
        img = img[:, :exp(h), :exp(w)]
        den = den[:exp(h), :exp(w)]
        den2 = den2[:exp(h), :exp(w)]

        return img, den * 1000, den2

    def load_data(self, img_path):

        got_img = False

        gt_path = img_path.replace('.jpg', '_sigma15.h5').replace('images', 'ground_truth')
        gt_path2 = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
        if not osp.exists(img_path) or not osp.exists(gt_path):
            raise IOError("{} does not exist".format(img_path))

        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                gt_file = h5py.File(gt_path)
                gt_file2 = h5py.File(gt_path2)
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass

        den = np.asarray(gt_file['density'])
        den2 = np.asarray(gt_file2['density'])
        h = den.shape[0]
        w = den.shape[1]
        h_trans = h // 8
        w_trans = w // 8
        # den = cv2.resize(den, (0, 0),fx=0.125, fy=0.125, interpolation=cv2.INTER_LINEAR) * 64
        # den = cv2.resize(den, (w_trans, h_trans), interpolation=cv2.INTER_LINEAR) * 64

        return img, den, den2
