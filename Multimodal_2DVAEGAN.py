from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.optim as optim
#from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import math
#import torchio as tio
import albumentations as A
from albumentations.pytorch import ToTensorV2

from loss import loss_function

import torchvision.models as models

parser = argparse.ArgumentParser(description='Pytorch DCGAN - Medical Data')
parser.add_argument('--workers',default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--num_epochs', default=10, type=int, metavar='epochs',
                    help='number of total epochs to run')
parser.add_argument('--ngpu', default=2, type=int, metavar='ngpu',
                    help='number of GPUs to run')
parser.add_argument('--channel', default=1, type=int, metavar='channel',
                    help='channel image ')
parser.add_argument('--features', default=32, type=int, metavar='features',
                    help='dimension of the feature map [Discriminator|Generator] ')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='batch_size', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--latent_z', default=1024, type=int, metavar='latent_z',
                    help='number of latent dimension')
parser.add_argument('--beta1', default=0.05, type=int, metavar='beta1',
                    help='beta for Adam Optimizers')
parser.add_argument('-save', '--save', default=100, type=int, metavar='save',
                    help='save every epoch')
parser.add_argument('-plot_inter', '--plot_inter', default=50, type=int, metavar='save',
                    help='for plot')
parser.add_argument('--beta', default=10, type=int, metavar='beta',
                    help='beta value for loss function _ recons')
parser.add_argument('--gamma', default=1, type=int, metavar='gamma',
                    help='beta value for loss function _ kl')



# create folder for saving training and test results
global args
args = parser.parse_args()


root_save = '2DVAEGAN/'

if not os.path.isdir(str(args.features)):
    os.makedirs(root_save+str(args.features))

if not os.path.isdir(str(args.features)+'/'+str(args.num_epochs)):
    os.makedirs(root_save+str(args.features)+'/'+str(args.num_epochs))


# random seed for reproducibility
#seed = 42
#random.seed(seed)
#torch.manual_seed(seed)

k = 2500

# Device
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

# Data loader
data_dir_train = './BraTS2021.npz'

def Data(dir):
    data_load = np.load(dir)
    train_data = np.array(data_load['data'])
    
    return train_data

# Get training and test data
train_data = Data(data_dir_train)


class MyDataset(Dataset):
    def __init__(self,data,transform=None):
        self.data = data
        self.transform = transform
        self.transform1 = transform

    def normalize(self,data):
        img_min = np.min(data)

        data_ = (data - img_min) / (np.max(data) - img_min)  

        return data_

    def __getitem__(self, index):
        imgs = self.data[index] 
        x0 = self.normalize(imgs[0])
        x1 = self.normalize(imgs[1])
        #x0 = imgs[0]/2313.0
        #x1 = imgs[1]/4.0
        
        x0 = x0.astype('float32').reshape((240, 240, 1))
        x1 = x1.astype('float32').reshape((240, 240, 1))
        
        if self.transform is not None:
          return self.transform(image=x0)["image"],self.transform(image=x1)["image"]
      
    def __len__(self):
        return len(self.data)


# Apply some data transformation
transform_A = A.Compose([
                A.CenterCrop(height=180, width=180),
                A.Resize(height=128, width=128),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
                    
])


transform_mask = A.Compose([
            A.CenterCrop(height=180, width=180),
            A.Resize(height=128, width=128),
            ToTensorV2(),
           
        ])


class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor.eval()
        self.criterion = nn.L1Loss()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        features_x = self.feature_extractor(x)
        with torch.no_grad():
            features_y = self.feature_extractor(y).detach()
        loss = 0
        for feat_x, feat_y in zip(features_x, features_y):
            loss += self.criterion(feat_x, feat_y)
        return loss

# Load a pre-trained feature extractor (e.g. VGG-16)
vgg16 = models.vgg16(pretrained=True)

vgg16.eval()
# Freeze all model parameters
for param in vgg16.parameters():
    param.requires_grad_(False)


# train data
print("Size of the data:",len(train_data))
dataset_train = MyDataset(train_data, transform=transform_A)
train_x = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True)


# test data
dataset_test = MyDataset(train_data, transform=transform_mask)
test_x = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=False)


# Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!= -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class View(nn.Module):
    def __init__(self,size):
        super(View,self).__init__()
        self.size = size
    
    def forward(self, tensor):
        return tensor.view(self.size)


# reconstruction loss
def reconstruction_loss(x, x_recon):
    recon_loss = nn.MSELoss(size_average=None, reduction="mean")

    return recon_loss(x_recon, x)


# Encoder
class Encoder(nn.Module):
    def __init__(self,channel,features):
        super(Encoder, self).__init__()
        self.channel = channel
        self.features = features

        padding = 0
        padding1 = 1

        self.conv1 = nn.Conv2d(self.channel, self.features*8,4,2,padding1, bias=False)
        self.conv2 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1, bias=False)
        self.conv3 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1, bias=False)
        self.conv4 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1, bias=False)
        #self.conv5 = nn.Conv2d(self.features, args.features,4,2,padding1, bias=False)

        self.conv5_ = nn.Conv2d(args.features*2*8, args.features*8,4,2,padding1, bias=False)
        #self.conv5 = nn.Conv2d(args.features*8, args.features*8,4,2,padding1, bias=False)
        self.conv6 = nn.Conv2d(args.features*8, args.latent_z,4,2,padding1, bias=False) #3,1

        self.batchNorm1 = nn.BatchNorm2d(self.features*8)
        #self.batchNorm2 = nn.BatchNorm2d(self.features*16)
        #self.batchNorm3 = nn.BatchNorm2d(args.features*8)
        #self.batchNorm4 = nn.BatchNorm2d(args.features*8)
        #self.batchNorm4 = nn.BatchNorm2d(args.features*8)

        self.batchNorm5 = nn.BatchNorm2d(args.latent_z)

        self.view = View((-1, args.latent_z*2*2))

        self.linear1 = nn.Linear(args.latent_z*2*2,args.latent_z)
        self.linear2 = nn.Linear(args.latent_z,args.latent_z*2*2)

        self.batchNorm6 = nn.BatchNorm1d(args.latent_z)

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,x1):
        #x = torch.concat((x,x1),dim=1)
        x = self.relu(self.batchNorm1(self.conv1(x)))   #8
        x = self.relu(self.batchNorm1(self.conv2(x)))   #16
        x = self.relu(self.batchNorm1(self.conv3(x)))   #32
        x = self.relu(self.batchNorm1(self.conv4(x)))   #64
        #x = self.relu(self.batchNorm3(self.conv5(x)))  #128

        x1 = self.relu(self.batchNorm1(self.conv1(x1))) #8
        x1 = self.relu(self.batchNorm1(self.conv2(x1))) #16
        x1 = self.relu(self.batchNorm1(self.conv3(x1))) #32
        x1 = self.relu(self.batchNorm1(self.conv4(x1))) #64
        #x1 = self.relu(self.batchNorm3(self.conv5(x1))) #128

        x_= self.relu(self.batchNorm1(self.conv5_(torch.concat((x,x1),dim=1))))
        #x_ = self.relu(self.batchNorm1(self.conv5(x))))   #64
        
        x = self.relu(self.batchNorm5(self.conv6(x_)))   #256
        x = self.view(x)    

        x = self.relu(self.linear1(x))
        x = self.linear2(x)

        return x



# Generator Code
class Generator(nn.Module):
    def __init__(self,channel,latent_z,features):
        super(Generator, self).__init__()
        self.latent_z = latent_z
        self.features = features
        self.channel = channel

        padding = 0
        padding1 = 1

        self.linear = nn.Linear(self.latent_z,self.latent_z*1*1, bias=False)

        self.view = View((-1, self.latent_z,1,1))
        
        self.convT1 = nn.ConvTranspose2d(self.latent_z, self.features*8,4,1,padding,bias=False)     #4
        self.convT2 = nn.ConvTranspose2d(self.features*8, self.features*8,4,2,padding1,bias=False)  #8
        self.convT3 = nn.ConvTranspose2d(self.features*8, self.features*8,4,2,padding1,bias=False)  #16
        self.convT4 = nn.ConvTranspose2d(self.features*8, self.features*8,4,2,padding1,bias=False)  #32
        self.convT5 = nn.ConvTranspose2d(self.features*8, self.features*8,4,2,padding1,bias=False)  #64
        #self.convT6 = nn.ConvTranspose2d(self.features*2, self.features,4,2,padding1,bias=False)    #128
        self.convT7 = nn.ConvTranspose2d(self.features*8, self.channel,4,2,padding1,bias=False)     #256

        self.batchNorm1 = nn.BatchNorm2d(self.features*8)
        #self.batchNorm2 = nn.BatchNorm2d(self.features*16)
        #self.batchNorm3 = nn.BatchNorm2d(self.features*8)
        #self.batchNorm4 = nn.BatchNorm2d(self.features*8)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

      
    def forward(self,x):
        x = self.linear(x)
        x = self.view(x)
        x = self.relu(self.batchNorm1(self.convT1(x)))
        x = self.relu(self.batchNorm1(self.convT2(x)))
        x = self.relu(self.batchNorm1(self.convT3(x)))
        x = self.relu(self.batchNorm1(self.convT4(x)))
        x = self.relu(self.batchNorm1(self.convT5(x)))
        #x = self.relu(self.batchNorm3(self.convT6(x)))
        x = self.tanh(self.convT7(x)) ##########

        return x


# Generator Code
class VAE_Generator(nn.Module):
    def __init__(self,channel,latent_z,features):
        super(VAE_Generator, self).__init__()
        self.latent_z = latent_z
        self.features = features
        self.channel = channel
        self.enc = Encoder(args.channel, args.features)
        self.view = View((-1, args.latent_z*2*2))
        self.decoder = Generator(args.channel,args.latent_z,args.features)

        self.fc_mu = nn.Linear(args.latent_z*2*2,args.latent_z)
        self.fc_logvar = nn.Linear(args.latent_z*2*2,args.latent_z)

        
    def encode(self,x,x1):
        x = self.enc(x,x1)
        x = self.view(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
        

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z
    
    def add_noise(self,z, noise_multiplier=0.01):
        noise = torch.randn_like(z) * noise_multiplier

        return z + noise

    def decode(self,x):
        x = self.decoder(x)
        
        return x

      
    def forward(self,x,x1):
        mu, logvar = self.encode(x,x1)
        z = self.reparameterize(mu, logvar)
        #z = self.add_noise(z)
        x_recon = self.decode(z)

        return x_recon, mu, logvar, z


# Create the generator and initialize # Multiple GPU
generator = VAE_Generator(args.channel,args.latent_z,args.features).to(device)
generator = nn.DataParallel(generator)
generator.apply(weights_init)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channel,features):
        super(Discriminator, self).__init__()
        self.channel = channel
        self.features = features

        padding = 0
        padding1 = 1

        self.conv1 = nn.Conv2d(self.channel, self.features*8,4,2,padding1,bias=False)       #4
        self.conv2 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1,bias=False)    #8
        self.conv3 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1,bias=False)    #16
        self.conv4 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1,bias=False)    #32
        self.conv5 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1,bias=False)   #64
        #self.conv6 = nn.Conv2d(self.features*4, self.features*4,4,2,padding1,bias=False)  #128
        self.conv7 = nn.Conv2d(self.features*8, 1,2,4,padding,bias=False)                  #256

        self.batchNorm1 = nn.BatchNorm2d(self.features*8)
        #self.batchNorm2 = nn.BatchNorm2d(self.features*8)  
        #self.batchNorm3 = nn.BatchNorm2d(self.features*8)

        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2)
       

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.batchNorm1(self.conv2(x)))
        x = self.leakyrelu(self.batchNorm1(self.conv3(x)))
        x = self.leakyrelu(self.batchNorm1(self.conv4(x)))
        x = self.leakyrelu(self.batchNorm1(self.conv5(x)))
        #x = self.leakyrelu(self.batchNorm3(self.conv6(x)))
        #x = self.conv7(x)
        #x = x.view(x.size()[0],-1)
        x = self.sigmoid(self.conv7(x))

        return x


# create the Discriminator
discriminator = Discriminator(args.channel,args.features).to(device)
discriminator = nn.DataParallel(discriminator)
discriminator.apply(weights_init)


generator_optimizer = optim.Adam(generator.parameters(),lr=args.lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(),lr=args.lr)

#scheduler = StepLR(generator_optimizer, step_size=200, gamma=0.1)

l2_loss = nn.MSELoss().to(device)
perceptual_loss = PerceptualLoss(vgg16).to(device)
criterion = nn.BCELoss().to(device)

generator.train()
discriminator.train()


# Training Loop
G_losses = []
D_losses = []

losses = []
losses_ = []

rec_noise = []
noise_noise = []

# for later plot
output_train = []
output_test  = []

fixed_noise = torch.randn(args.batch_size,args.latent_z, device=device)

# Start training
for epoch in range(args.num_epochs):
    for i, data in enumerate(train_x, 0):
        # Move data to device
        real_images = data[0].to(device)
        real_images1 = data[1].to(device)

        # Update discriminator
        discriminator.zero_grad()

        # Train with real images
        real_labels = torch.ones(real_images.size(0),).to(device)
        real_outputs = discriminator(real_images).view(-1)

        #print(real_labels.shape, real_outputs.shape)
        discriminator_real_loss = criterion(real_outputs, real_labels)

        # Train with fake images
        z = torch.randn(real_images.size(0), args.latent_z).to(device)

        x_recon, mu, logvar, z_prime = generator(real_images,real_images1)
        #cos_sim = nn.CosineSimilarity(dim=1,eps=1e-6).to(device)
        #cos_loss = cos_sim(z_prime.view(args.latent_z,-1),z.view(args.latent_z,-1))

        
        fake_images = generator.module.decode(z)

        tumor_loss = l2_loss(fake_images*real_images1,real_images*real_images1)
        #print(tumor_loss)

        fake_labels = torch.zeros(real_images.size(0),).to(device)
        recons_outputs = discriminator(x_recon.detach()).view(-1)
        discriminator_recons_loss = criterion(recons_outputs, fake_labels)

        fake_outputs = discriminator(fake_images.detach()).view(-1)
        discriminator_fake_loss = criterion(fake_outputs, fake_labels)

        # Compute total discriminator loss and update parameters
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss + discriminator_recons_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Update generator
        generator.zero_grad()

        # Train with fake images again
        fake_outputs = discriminator(fake_images).view(-1)
        generator_loss = criterion(fake_outputs, real_labels) + 10*tumor_loss
       

        # Compute VAE loss
        #x_recon, mu, logvar, z_prime = generator(real_images,real_images1)
        vae_recon_loss = l2_loss(x_recon, real_images)
        #vae_recon_loss = perceptual_loss(real_images, x_recon)
        vae_kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = args.beta*vae_recon_loss + args.gamma*vae_kl_loss

        # Compute perceptual loss
        perceptual_loss_real = perceptual_loss(real_images, x_recon)
        perceptual_loss_fake = perceptual_loss(real_images, fake_images)
        
        perceptual_loss_total = perceptual_loss_real + perceptual_loss_fake

        # Compute total generator loss and update parameters
        #generator_loss += 0.01 * perceptual_loss_total
        generator_loss += vae_loss + perceptual_loss_total  #cos_loss.mean().item()
        generator_loss.backward()
        generator_optimizer.step()

        #scheduler.step()

        # Save losses for plotting later
        G_losses.append(generator_loss.item())
        D_losses.append(discriminator_loss.item())

        if math.isnan(generator_loss) or math.isnan(discriminator_loss):
            print("The find nan for one of these loss generator_loss or discriminator_loss: ",generator_loss.item(),discriminator_loss.item())
            break


        # Output training stats
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{args.num_epochs}] Batch [{i}/{len(train_x)}] Discriminator Loss: {discriminator_loss.item():.6f} Generator Loss: {generator_loss.item():.6f}")

    if (epoch % k == 0 and epoch > 0):
        output_train.append([epoch,real_images.cpu().detach().numpy(),fake_images.cpu().detach().numpy(),x_recon.cpu().detach().numpy()])

        ######################## compute loss values ###################
        loss_ssim, loss_psrn, loss_mse = loss_function(real_images.cpu().detach().numpy(),fake_images.cpu().detach().numpy())
        losses.append([epoch, loss_ssim, loss_psrn, loss_mse])

    # check the performance of the generator on fixed_noise
    #if epoch % k == 0 and epoch > 0:
        with torch.no_grad():
            fake = generator.module.decode(fixed_noise)
            output_test.append([epoch,real_images.cpu().detach().numpy(),fake.cpu().detach().numpy()])
            #loss_ssim_, loss_psrn_, loss_mse_ = loss_function(real_images.cpu().detach().numpy(),fake.cpu().detach().numpy())
            #losses_.append([epoch, loss_ssim_, loss_psrn_, loss_mse_])

    # save results to plot later
    if (epoch % k == 0 and epoch > 0):
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/train_out'+'_'+str(epoch)+'.npz', data = output_train, num_epochs = args.num_epochs, allow_pickle=True)
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/test_out'+'_'+str(epoch)+'.npz', data = output_test, num_epochs = args.num_epochs, allow_pickle=True)
        ####################### save losses ############################
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/G_D_losses.npz', G_losses=G_losses, D_losses=D_losses,allow_pickle=True)

        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/loss_function_train.npz', losses=losses, allow_pickle=True)
        #np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/loss_function_test.npz', losses=losses_, allow_pickle=True)

        
        plt.plot(G_losses, label="G-loss",color='r')
        plt.plot(D_losses, label="D-loss",color='b')
        plt.legend()
        plt.title("Generator and Discriminator Loss")
        plt.grid()
        plt.savefig(root_save+str(args.features)+'/'+str(args.num_epochs)+'/lossGD.png')
        plt.clf()

############# THE END  ############## 

















