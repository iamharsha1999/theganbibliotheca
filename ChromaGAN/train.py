import torch 
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn

## Initilize the parameters
optm = Adam(lr = 2e-5, betas = (0.5,0.999))

def wgan_loss(fake, real = 0, model = 'dis'):
    ## For discrminator
    if model = 'dis':
        return -(torch.mean(real) - torch.mean(fake))
    else 
        return  -torch.mean(fake)

def Train(device, generator,  discriminator, dataloader, epochs = 5, batch_size = 10):

    """
    Color Error Loss = MSE
    Class Vector Loss = KL Loss
    Adversarial Loss = WGAN Loss

    L(G) = MSE + KL Loss + WGAN Loss
    """

    ## Build the VGG Model
    vgg_model = pretrainedmodels.__dict__['vgg16'](pretrained = 'imagenet').to(device)

    optmg = Adam(generator.parameters(), lr = 2e-5, betas = (0.5,0.99)) 
    optmd = Adam(discriminator.prameters(), lr = 2e-5, betas = (0.5, 0.99))

    klloss = nn.KLDivLoss()
    mse = nn.MSELoss()
    

    for epoch in range(epochs):

        epoch_loss = {'gen_loss':[], 'dis_loss':[]}
        print('[Epoch: {} / {}]'.format(epoch+1, EPOCHS))

        for batch in tqdm(dataloader):

            real_l,real_ab = batch['l'].to(device),batch['ab'].to(device)
            
            ## Retrive the Batch Size
            bs = images.size(0)

            ## VGG class prediction for real grayscale image
            output_vgg = vgg_model()

            ## Labels for Real Images and Fake Images
            targetr = torch.ones(bs, device = device)
            targetf = torch.zeros(bs, device = device)
            
            ## Produce fake images with generator
            img_class,fake_ab = generator(real_l)

            ## Clear the accumulated gradients
            generator.zero_grad()

            col_err = mse(fake_ab, real_ab)
            col_err.backward()
            class_loss = klloss(img_class, output_vgg)
            class_loss.backward()        
            
            ## Fake images for generator
            dis_out = discriminator(fake_ab, real_l)
            fake_loss = wgan_loss(dis_out, targetf)

            loss = col_err + 0.003 * class_loss + fake_loss
            loss.backward()
            optmg.step()

            ## Clear the accumulated gradients
            discriminator.zero_grad()

            ## Real images for discriminator
            dis_out = discriminator(fake_ab.detach(), real_l)
            fake_loss = wgan_loss(dis_out, targetf)
            
            ## Fake images for discriminator
            dis_out = discriminator(fake_ab.detach(), real_l)
            fake_loss = wgan_loss(dis_out, targetf)
            fake_loss.backward()
            
            ## Real Images for discriminator
            dis_out = discriminator(real_ab, real_l)
            real_loss = wgan_loss(dis_out, targetr)
            real_loss.backward()

            optmd.step()






            

            ##Zero down the gradient for generator
            generator.zero_grad()

            ## Produce fake outputs
            img_class, fake_ab = generator(real_l)

            ## Discriminator output for fake images
            dis_out = discriminator(fake_ab, real_l)

            ## WGAN loss for fake images
            fake_wloss = wgan_loss(dis_out, targetr)

           







        
