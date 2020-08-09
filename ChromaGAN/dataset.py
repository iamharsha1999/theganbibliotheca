import torch 
import cv2
from tqdm import tqdm
from torch.utils.data import  Dataset
import os 


class TinyImagenet(Dataset):

    def __init__(self, rootdir):
        super().__init__()

        self.root_dir = rootdir 
        self.size = (224,224)
        self.img_list = []
        for folder in os.listdir(self.root_dir):
            if folder == 'train':
                for subfolder in  os.listdir(self.root_dir + '/' + folder ):
                    for image in os.listdir(self.root_dir + '/' + folder + '/' + subfolder + '/images'):
                        self.img_list.append(self.root_dir + '/' + folder + '/' + subfolder + '/images/' + image)
            elif folder == 'test' or folder == 'val':
                for image in os.listdir(self.root_dir + '/' + folder + '/images'):
                    self.img_list.append(self.root_dir + '/' + folder + '/images/' + image)

        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):

        img = cv2.imread(self.img_list[idx])

        # Resize and covert the image into CIE Lab Color Space
        img = cv2.cvtColor(cv2.resize(img, self.size), cv2.COLOR_BGR2LAB)
        # Split the images into L and AB channels
        l = img[:,:,0]
        ab = img[:,:,1:3]

        return {
            'l': torch.tensor(l, dtype=torch.float32).view(-1,224,224),
            'ab':torch.tensor(ab, dtype=torch.float32).view(-1,224,224)
        }


        


