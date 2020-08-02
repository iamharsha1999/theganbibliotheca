import torch 
import torch.nn as nn
import torch.functional as F 

class generator(nn.Module):

    def __init__(self):
        super().__init__()

class discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # 224 x 224 x
            nn.Conv2d(3, 64, kernel_size=4, padding = 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, padding = 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, padding = 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, padding = 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, padding =1, bias=False)
            )

    def forward(self,x,y):

        x = torch.cat(x,y)

        x = self.model(x)

        return x 



        
