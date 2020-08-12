import torch 
from torch.autograd import grad 
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
import pretrainedmodels
import cv2

class Trainer():

    def __init__(self, gen , dis,data, fixed_gray_images,device = 'cuda'):

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
        
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.epochs = 5
        self.batch_size = 10
        self.gen_optimizer = Adam(gen.parameters(), lr = 2e-5, betas=(0.5, 0.999))
        self.dis_optimizer = Adam(dis.parameters(), lr = 2e-5, betas=(0.5, 0.999))
        
        self.klloss = nn.KLDivLoss()
        self.mseloss = nn.MSELoss()    
        self.dataloader = data 

        self.gp_weight = 10

        self.fixed_noise = fixed_gray_images
                
    def wgan_loss(self,fake, real ):
        
        return -(torch.mean(real) - torch.mean(fake))
    
    def gradient_penalty(self, real_ab, fake_ab, real_l):

        interpolated = self.interpolated_images(real_ab, fake_ab)
        dis_interpolated = self.discriminator(interpolated,real_l)

        gradients = grad(outputs = dis_interpolated, inputs=interpolated, grad_outputs=torch.ones(dis_interpolated.size()).to(self.device),
                                         create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(real_ab.size()[0], -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        loss = self.gp_weight * ((gradients_norm - 1) ** 2).mean()

        return loss 
        
    def interpolated_images(self, x, y):

        alpha = torch.randn(x.size()[0],1,1,1).expand_as(x).to(self.device)
        interpolated = alpha * x + (1-alpha) * y

        return interpolated

    
    def train_gen(self, l, vgg_output, real_ab):

        ## Generate images and class output
        img_class, gen_out = self.generator(l)
      
        ## Fake images for discriminator
        dis_out = self.discriminator(gen_out, l)

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
        gp = self.gradient_penalty(real_ab, fake_ab, real_l)

        ##Update the weights
        self.discriminator.zero_grad()
        loss = self.wgan_loss(fake_dis_out, real_dis_out) + gp 
        loss.backward()
        self.dis_optimizer.step()

        return loss
    
    def plot_images(self,real_l, no_of_images, epoch_no):

        _,pred_ab = self.generator(real_l)
        img = torch.cat((real_l,pred_ab), dim =1)
        img = img.to('cpu').numpy()
        
        for i in len(no_of_images):
            img[i] = cv2.cvtColor(img[i], cv2.COLOR_Lab2LBGR)
            plt.imshow(np.transpose(img[i], (1,2,0)), interpolation = 'none')
            plt.savefig('Image_Epoch:{}_{}.png'.format(epoch_no+1,i+1))
            
    def train(self):

        self.gen_loss = []
        self.dis_loss = []

        for epoch in range(self.epochs):

            epoch_loss = {'gen_loss':[], 'dis_loss':[]}
            print('[Epoch: {} / {}]'.format(epoch+1, self.epochs))
            
            eg_loss = []
            ed_loss = []
            
            for batch in tqdm(self.dataloader):

                real_l,real_ab = batch['l'].to(self.device),batch['ab'].to(self.device)
                
                ## Retrive the Batch Size
                bs = real_l.size(0)

                ## VGG class prediction for real grayscale image
                output_vgg =  self.vgg_model(real_l.repeat(1,3,1,1))

                ## Update the generator
                gen_loss = self.train_gen(real_l, output_vgg, real_ab)
                eg_loss.append(gen_loss)

                ## Update the discrminator
                dis_loss = self.train_disc(real_ab, real_l)
                ed_loss.append(dis_loss)

            self.gen_loss.append(torch.mean(torch.FloatTensor(eg_loss)))
            self.dis_loss.append(torch.mean(torch.FloatTensor(ed_loss)))

            ## Print Epoch Information
            print('[ Generator Loss: {} | Discriminator Loss: {} ] '.format(gen_loss, dis_loss))

            ## Plot predicted images to visualize images
            self.plot_images(self.fixed_noise, self.fixed_noise.size()[0], epoch + 1)