import torch 
from PIL import Image
from tqdm import tqdm
from torch.utils.data import  Dataset
from torchvision import datasets
import os 
import numpy as np

class  Dataprep(Dataset):

    def __init__(self, root, mean =0, var = 0.1, sigma =25, data = 'stl-10'):
        super().__init__()

        ## Root Directory 
        self.root = root 

        self.mean = 0
        self.var = 0.1
        self.sigma = 25

        if data == 'stl-10':
            self.data = datasets.STL10(self.root, split = 'train', download = True)
        
        ## Downscaling Factor
        self.ds_factor = 4

    def __len__(self):

        return len(self.data)
    
    def gauss_noise(self,image):
        row, col, ch = image.shape
        
        gauss = np.random.normal(self.mean,self.sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss

        return noisy

    def __getitem__(self,idx):

        img, _ = self.data[idx]
        h,w= img.size
        img_n = img.resize((int(h/self.ds_factor),int(w/self.ds_factor)))

        ## Gaussian Noise 
        img_n = self.gauss_noise(np.array(img_n))
        img_n = torch.tensor(img_n, dtype=torch.float32).view(-1, int(h/self.ds_factor), int(w/self.ds_factor))
        img = torch.tensor(np.array(img), dtype=torch.float32).view(-1,h,w)
        
        return {'hr':img,
                        'lr':img_n}

