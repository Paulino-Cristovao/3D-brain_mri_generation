import numpy as np
from scipy.signal import fftconvolve
from scipy import signal
from scipy import ndimage
import math
from scipy.ndimage import gaussian_filter
import math


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def SSIM(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)

def MSSSIM(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on
    Signals, Systems and Computers, Nov. 2003
    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    #im1 = img1.astype(np.float64)
    #im2 = img2.astype(np.float64)

    mssim = np.array([])
    mcs = np.array([])

    for l in range(level):
        ssim_map, cs_map = SSIM(im1, im2)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter,
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter,
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level - 1] ** weight[0:level - 1]) *
            (mssim[level - 1] ** weight[level - 1]))


def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    #print('MSE Loss: ',mse)

    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def loss_function(x,y):
    loss_ssim = SSIM(x, y)
    #print('SSIM Loss: ',loss_ssim)
    loss_psrn = PSNR(x, y)
    #print('PSNR Loss: ',loss_psrn)
    loss_mse = np.mean((x - y) ** 2)
    #print('MSE Loss: ',loss_mse)

    return loss_ssim, loss_psrn, loss_mse






