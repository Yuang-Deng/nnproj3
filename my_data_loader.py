import pickle

import torch.utils.data as data
from PIL import Image
import os


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list):
        self.root = data_root
        f = open(os.path.join(data_root, data_list), 'rb')
        data = pickle.load(f)
        f.close()
        self.dataset = []
        self.labelset = []
        for item in list(data.keys()):
            for i in range(len(data[item]['data'])):
                self.dataset.append(data[item]['data'][i])
                self.labelset.append(data[item]['label'][i]+1)
        self.n_data=len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labelset[item]

    def __len__(self):
        return self.n_data
