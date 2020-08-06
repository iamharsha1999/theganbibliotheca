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

    def interpolated_images(self):
    
    def train_gen(self, l, vgg_output, real_ab):

        ## Generate images and class output
        img_class, gen_out = self.generator(l)

        ## Fake images for discriminator
        dis_out = self.discriminator(gen_out)

        dis_loss = -dis_out.mean()
        kl_loss = self.klloss(img_class, vgg_output)
        ce_loss = self.mseloss(gen_out, real_ab)

        loss = ce_loss + 0.003*kl_loss + dis_loss


        ## Update the weights
        self.generator.zero_grad()
        loss.backward()
        self.gen_optimizer.step()

        return loss 

    def train_disc(self, real_ab, real_l):

        ## Generated Data
        _, fake_ab = self.generator(real_l)

        ## Predictions from discriminator for real and fake chrominance channels
        real_dis_out = self.discriminator(real_ab, real_l)
        fake_dis_out = self.discriminator(fake_ab.detach(), real_l)

        ## Compute the gradient penalty
        gp = self.gradient_penalty(real_ab, fake_ab)

        ##Update the weights
        self.discriminator.zero_grad()
        loss = self.wgan_loss(fake_dis_out, real_dis_out) + gp 
        loss.backward()
        self.dis_optimizer.step()

        return loss 

    def train(self):

        self.gen_loss = []
        self.dis_loss = []

        for epoch in range(self.epochs):

            epoch_loss = {'gen_loss':[], 'dis_loss':[]}
            print('[Epoch: {} / {}]'.format(epoch+1, self.epochs))
            
            eg_loss = 0
            ed_loss = 0 
            
            for batch in tqdm(self.dataloader):

                real_l,real_ab = batch['l'].to(self.device),batch['ab'].to(self.device)
                
                ## Retrive the Batch Size
                bs = images.size(0)

                ## VGG class prediction for real grayscale image
                output_vgg =  self.vgg_model(real_l)

                ## Update the generator
                gen_loss = self.train_generator(real_l, output_vgg, real_ab)
                eg_loss.append(gen_loss)

                ## Update the discrminator
                dis_loss = self.train_disc(real_ab, real_l)
                ed_loss.append(dis_loss)

            self.gen_loss.append(torch.mean(torch.FloatTensor(eg_loss)))
            self.dis_loss.append(torch.mean(torch.FloatTensor(ed_loss)))

            ## Print Epoch Information
            print('[ Generator Loss: {} | Discriminator Loss: {} ] '.format(gen_loss, dis_loss))