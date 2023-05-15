#
from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
from torch.autograd import Variable
from torch import autograd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
#import torchio as tio
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
#import torchvision.models as models

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import Compose, RandFlip, Resize, ScaleIntensity,ToTensor, CenterSpatialCrop

from loss import loss_function


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
parser.add_argument('--batch_size', default=5, type=int,
                    metavar='batch_size', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--latent_z', default=256, type=int, metavar='latent_z',
                    help='number of latent dimension')
parser.add_argument('--beta1', default=0.5, type=int, metavar='beta1',
                    help='beta for Adam Optimizers')
parser.add_argument('-save', '--save', default=100, type=int, metavar='save',
                    help='save every epoch')
parser.add_argument('-plot_inter', '--plot_inter', default=50, type=int, metavar='save',
                    help='for plot')
parser.add_argument('--beta', default=10, type=int, metavar='beta',
                    help='beta value for loss function ')
parser.add_argument('--gamma', default=20, type=int, metavar='gamma',
                    help='beta value for loss function')


# create folder for saving training and test results
global args
args = parser.parse_args()

root_save = '/I3M_IO_CALCULS_2/allInOne/3DGAN/'

if not os.path.isdir(str(args.features)):
	os.mkdir(root_save+str(args.features))

if not os.path.isdir(str(args.features)+'/'+str(args.num_epochs)):
	os.mkdir(root_save+str(args.features)+'/'+str(args.num_epochs))


# random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)

k = 2000

# Device
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

# Data loader
#data_dir_train = '/I3M_IO_CALCULS_2/3D_240bigans/BraTS2021_3D100.npz'
#data_dir_train = './BraTS2021_3D111.npz'
data_dir_train = '/I3M_IO_CALCULS_2/allInOne/Data/BraTS2021_3D30.npz'

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
    
    def normalize(self,data):
        img_min = np.min(data)

        return (data - img_min) / (np.max(data) - img_min)

    def __getitem__(self, index):
        imgs = self.data[index] 
        x0 = imgs[0] 
        x1 = imgs[1] 
    
                
        if x0.max() != 0 and x1.max() != 0:
            x0 = x0.astype('float32').reshape((1, 240, 240, 155))
            x1 = x1.astype('float32').reshape((1, 240, 240, 155))

            if self.transform is not None:
              return self.transform(x0)#,self.transform(x1)
      
    def __len__(self):
        return len(self.data)

transform_A = Compose([
                        CenterSpatialCrop((180, 180, 155)),
                        Resize((128,128,128)),
                        ScaleIntensity(),
                        RandFlip(prob=0.5, spatial_axis=None)
                        ])
                        
transform_mask = Compose([
                        CenterSpatialCrop((180, 180, 155)),
                        Resize((128,128,128)),
                        ])

# train data
print("Size of the data:",len(train_data))
dataset_train = MyDataset(train_data, transform=transform_A)
train_x = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True)


# test data
dataset_test = MyDataset(train_data, transform=transform_A)
test_x = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=False)

# Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    #if classname.find('Conv')!= -1:
    #	nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
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

# kl divergence
def kl_divergence(mu, logvar):
    latent_kl = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()).mean(dim=0)
    total_kld = latent_kl.sum()

    return total_kld


def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

def calc_gradient_penalty(netD, real_data, fake_data,LAMBDA=10):    
    alpha = torch.rand(real_data.size(0),1,1,1,1)
    alpha = alpha.expand(real_data.size())
    
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty

# Generator Code
class Generator(nn.Module):
    def __init__(self,channel,latent_z,features):
        super(Generator, self).__init__()
        self.latent_z = latent_z
        self.features = features
        self.channel = channel

        padding = (0,0,0)
        padding1 = (1,1,1)

        self.linear = nn.Linear(self.latent_z,self.latent_z*1*1*1, bias=False)

        self.view = View((-1, self.latent_z,1,1,1))
        
        self.convT1 = nn.ConvTranspose3d(self.latent_z, self.features*5,4,1,0,bias=False)     #4
        self.convT2 = nn.ConvTranspose3d(self.features*5, self.features*4,4,2,1,bias=False)  #8
        self.convT3 = nn.ConvTranspose3d(self.features*4, self.features*3,4,2,1,bias=False)  #16
        self.convT4 = nn.ConvTranspose3d(self.features*3, self.features*2,4,2,1,bias=False)  #32
        self.convT5 = nn.ConvTranspose3d(self.features*2, self.features,4,2,1,bias=False)  #64
        #self.convT6 = nn.ConvTranspose3d(self.features, self.features,4,2,1,bias=False)    #128
        self.convT7 = nn.ConvTranspose3d(self.features, self.channel,4,2,1,bias=False)       #256

        self.batchNorm1 = nn.BatchNorm3d(self.features*5, momentum=0.9)
        self.batchNorm2 = nn.BatchNorm3d(self.features*4, momentum=0.9)
        self.batchNorm3 = nn.BatchNorm3d(self.features*3, momentum=0.9)
        self.batchNorm4 = nn.BatchNorm3d(self.features*2, momentum=0.9)
        self.batchNorm5 = nn.BatchNorm3d(self.features, momentum=0.9)

        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
      
    def forward(self, x):
        x = self.linear(x)
        x = self.view(x)

        x = self.relu(self.batchNorm1(self.convT1(x)))
        x = self.relu(self.batchNorm2(self.convT2(x)))
        x = self.relu(self.batchNorm3(self.convT3(x)))
        x = self.relu(self.batchNorm4(self.convT4(x)))
        x = self.relu(self.batchNorm5(self.convT5(x)))
        #x = self.relu(self.batchNorm3(self.convT6(x)))
        x = self.tanh(self.convT7(x))

        return x


# Create the generator and initialize # Multiple GPU
netG = Generator(args.channel,args.latent_z,args.features).to(device)
netG = nn.DataParallel(netG)

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.02.
netG.apply(weights_init)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channel,features):
        super(Discriminator, self).__init__()
        self.channel = channel
        self.features = features

        padding = (0,0,0)
        padding1 = (1,1,1)

        self.conv1 = nn.Conv3d(self.channel, self.features,4,2,padding1,bias=False)     #4
        self.conv2 = nn.Conv3d(self.features, self.features*2,4,2,padding1,bias=False)    #8
        self.conv3 = nn.Conv3d(self.features*2, self.features*3,4,2,padding1,bias=False)    #16
        self.conv4 = nn.Conv3d(self.features*3, self.features*4,4,2,padding1,bias=False)    #32
        self.conv5 = nn.Conv3d(self.features*4, self.features*5,4,2,padding1,bias=False)    #64
        #self.conv6 = nn.Conv3d(self.features, self.features,4,2,padding1,bias=False)   #128
        self.conv7 = nn.Conv3d(self.features*5, 1,2,4,padding,bias=False)                 #256

        self.batchNorm1 = nn.BatchNorm3d(self.features, momentum=0.9)
        self.batchNorm2 = nn.BatchNorm3d(self.features*2, momentum=0.9)  
        self.batchNorm3 = nn.BatchNorm3d(self.features*3, momentum=0.9)
        self.batchNorm4 = nn.BatchNorm3d(self.features*4, momentum=0.9)  
        self.batchNorm5 = nn.BatchNorm3d(self.features*5, momentum=0.9)

        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.batchNorm2(self.conv2(x)))
        x = self.leakyrelu(self.batchNorm3(self.conv3(x)))
        x = self.leakyrelu(self.batchNorm4(self.conv4(x)))
        x = self.leakyrelu(self.batchNorm5(self.conv5(x)))
        #x = self.leakyrelu(self.batchNorm1(self.conv6(x)))
        x = self.sigmoid(self.conv7(x))

        return x
        

# create the Discriminator
netD = Discriminator(args.channel,args.features).to(device)
netD = nn.DataParallel(netD)
netD.apply(weights_init)

# Initialization BCELoss function
criterion_D =  nn.BCELoss().to(device) 
criterion_G =  nn.BCELoss().to(device)
criterion_l1 =  nn.L1Loss().to(device)

# create a batch of latent vectors that
# we will use to visualize the progression of the generator
def fixed_noise():
    return  torch.randn(args.batch_size,args.latent_z, device=device)
#fixed_noise = torch.randn(args.batch_size,args.latent_z, device=device)

#def noise_(b_size):
#    return torch.randn(b_size, args.latent_z, device=device)

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

# Optimizers for both Generator and Discriminator
optimD = optim.Adam(netD.parameters(), lr=args.lr)
optimG = optim.Adam(netG.parameters(), lr=args.lr)

# Training Loop
G_losses = []
D_losses = []
#W_losses = []

losses = []

# for later plot
output_train = []
output_test  = []

netG.train()
netD.train()


print("Sarting training loop ... ")
# for each epoch
for epoch in range(args.num_epochs):
    ######### train discriminator #########
    for i, data in enumerate(train_x, 0):
        netD.zero_grad()
        # Train with all-real batch
        real_cpu = data.to(device, non_blocking=True, dtype=torch.float32) 
        #real_cpu1 = data[1].to(device, non_blocking=True, dtype=torch.float32)

        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float32, device=device)

        real_pred = netD(real_cpu).view(-1)
        errD_real = criterion_D(real_pred, label)
        errD_real.backward()
        errD1 = real_pred.mean().item()


        noise = torch.randn(b_size, args.latent_z, device=device)
        fake_images = netG(noise)
        label.fill_(fake_label)
        fake_pred = netD(fake_images.detach()).view(-1)
        errD_fake = criterion_D(fake_pred, label)
        errD_fake.backward()
        #d_fake_loss = fake_pred.mean().item()
        errD = errD_real + errD_fake 
        optimD.step()

        ############## update Generator network ################
        netG.zero_grad()
        label.fill_(real_label)
        fake_pred = netD(fake_images).view(-1)
        errG = criterion_G(fake_pred, label)
        errG.backward()
        optimG.step()
            
        # output training stats
        if i%500 == 0:
          print(f'[Epoch/num_epochs]: {epoch}/{args.num_epochs},| errD:{errD.item()} | errG:{errG.item()}')


        # Save losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
    
    if (epoch % k == 0 and epoch > 0):
        output_train.append([epoch,real_cpu.cpu().detach().numpy(),fake_images.cpu().detach().numpy()])

        ######################## compute loss values ###################
        loss_ssim, loss_psrn, loss_mse = loss_function(real_cpu.cpu().detach().numpy(),fake_images.cpu().detach().numpy())
        losses.append([epoch, loss_ssim, loss_psrn, loss_mse])

    # check the performance of the generator on fixed_noise
    if epoch % k == 0 and epoch > 0:
      with torch.no_grad():
        fake = netG(fixed_noise()).detach().cpu()
        output_test.append([epoch,real_cpu.cpu().detach().numpy(),fake.cpu().detach().numpy()])

    # save results to plot later
    if (epoch % k == 0 and epoch > 0):
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/train_out'+'_'+str(epoch)+'.npz', data = output_train, num_epochs = args.num_epochs, plot_inter =args.plot_inter, allow_pickle=True)
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/test_out'+'_'+str(epoch)+'.npz', data = output_test, num_epochs = args.num_epochs, plot_inter =args.plot_inter, allow_pickle=True)
        ####################### save losses ############################
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/G_D_losses.npz', G_losses=G_losses, D_losses=D_losses, allow_pickle=True)
        #np.savez(str(args.features)+'/'+str(args.num_epochs)+'/G_D_losses.npz', G_losses=G_losses, D_losses=D_losses, W_losses=W_losses, allow_pickle=True)

        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/loss_function.npz', losses=losses, allow_pickle=True)

        plt.plot(G_losses, label="G-loss",color='r')
        plt.plot(D_losses, label="D-loss",color='b')
        plt.legend()
        plt.title("Generator and Discriminator Loss")
        plt.grid()
        plt.savefig(root_save+str(args.features)+'/'+str(args.num_epochs)+'/lossGD.png')
        plt.clf()

        #plt.plot(W_losses, label="EWasserstein_D-loss",color='g')
        #plt.legend()
        #plt.title("Wasserstein_D Loss")
        #plt.grid()
        #plt.savefig(str(args.features)+'/'+str(args.num_epochs)+'/lossE.png')
        #plt.clf()

############# THE END  ##############     

