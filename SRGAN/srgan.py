import torch 
import torch.nn as nn 

class ResidualBlock(nn.Module):
    
    def __init__(self,input_channels, output_channels):
        super().__init__()

        self.block = nn.Sequential(

                                    nn.Conv2d(input_channels, output_channels, 3, stride = 1, padding=1),
                                    nn.BatchNorm2d(output_channels),
                                    nn.PReLU(),
                                    nn.Conv2d(output_channels, output_channels, 3, stride = 1, padding =1),
                                    nn.BatchNorm2d(output_channels),
                                    nn.PReLU()
        )

    def forward(self, x):
        x = self.block(x) + x 
        return x  


class Generator(nn.Module):

    def __init__(self,in_channels):
        super().__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels,64, 9, stride =1, padding = 4),
            nn.PReLU()
        )

        self.r1 = ResidualBlock(64,64) 
        self.r2 = ResidualBlock(64,64)
        self.r3 = ResidualBlock(64,64)
        self.r4 = ResidualBlock(64,64)
        self.r5 = ResidualBlock(64,64) 

        self.l2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding =1),
                nn.BatchNorm2d(64)
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(64,256, 3, stride =1, padding = 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64,256, 3, stride =1, padding = 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, in_channels, 9, stride =1, padding =4)
        )
    
    def forward(self, x):

        x = self.l1(x)

        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)

        x = x + self.l2(x)
        x = self.l3(x)

        return x 

class Discriminator(nn.Module):

    def __init__(self, in_channels,height, width):
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(in_channels, 64, 3, stride =1, padding = 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,64,3,stride = 2, padding =1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,3,stride = 1, padding =1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,128,3,stride = 2, padding =1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,256,3,stride = 1, padding =1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256,256,3,stride = 2, padding =1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256,512,3,stride = 1, padding =1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512,512,3,stride = 2, padding =1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512,1,1)
        )

    def forward(self,x):

        return self.model(x)

