import torch 
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
import pretrainedmodels

class Trainer():

    def __init__(self, gen , dis,data, device = 'cuda'):

        self.device = device 
        self.generator = gen 
        self.discriminator = dis
        self.vgg_model = pretrainedmodels.__dict__['vgg16'](pretrained = 'imagenet').to(device)

        ## Device Initialization
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Device Name:', torch.cuda.get_device_name(0))
        else:
            self.device = torch.device('cpu')

        self.epochs = 5
        self.batch_size = 10
        self.gen_optimizer = Adam(gen.parameters(), lr = 2e-5, betas=(0.5, 0.999))
        self.dis_optimizer = Adam(dis.parameters(), lr = 2e-5, betas=(0.5, 0.999))
        
        self.klloss = nn.KLDivLoss()
        self.mseloss = nn.MSELoss()    
        self.dataloader = ## need to fill


    def wgan_loss(self,fake, real ):
        
        return -(torch.mean(real) - torch.mean(fake))
    
    def gradient_penalty(self):

    def train(self):

        for epoch in range(self.epochs):

            epoch_loss = {'gen_loss':[], 'dis_loss':[]}
            print('[Epoch: {} / {}]'.format(epoch+1, self.epochs))

            for batch in tqdm(self.dataloader):

                real_l,real_ab = batch['l'].to(self.device),batch['ab'].to(self.device)
                
                ## Retrive the Batch Size
                bs = images.size(0)

                ## VGG class prediction for real grayscale image
                output_vgg =  self.vgg_model

                ## Labels for Real Images and Fake Images
                targetr = torch.ones(bs, device = device)
                targetf = torch.zeros(bs, device = device)
                
                ## Produce fake images with generator
                img_class,fake_ab = self.generator(real_l)

                ## Clear the accumulated gradients
                generator.zero_grad()

                col_err = self.mseloss(fake_ab, real_ab)
                col_err.backward()
                class_loss = self.klloss(img_class, output_vgg)
                class_loss.backward()        
                
                ## Fake images for generator
                dis_out = discriminator(fake_ab, real_l)
                fake_loss = self.wgan_loss(dis_out, targetf)

                loss = col_err + 0.003 * class_loss + fake_loss
                loss.backward()

                ## Update generator parameters
                self.gen_optimizer.step()

                ## Clear the accumulated gradients
                discriminator.zero_grad()
                
                ## Fake images for discriminator
                dis_fake_out = self.discriminator(fake_ab.detach(), real_l)
                                
                ## Real Images for discriminator
                dis_real_out = self.discriminator(real_ab, real_l)
                
                dis_loss = self.wgan_loss(dis_fake_out, dis_real_out) 3# + gradient_penalty
                dis_loss.backward()

                ## Update discriminator parameters
                self.dis_optimizer.step()

           







        
