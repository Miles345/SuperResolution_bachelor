##
# @file Gaussian_downsample.py
# @brief Gaussian kernel for downsampling GT frames to LR 

## Imports
import scipy.ndimage.filters as fi
import numpy as np
import torch.nn.functional as F
import torch 

def gaussian_downsample(x, scale=4):
    """!Downsamping with Gaussian kernel used in the DUF official code
    @param x  (Tensor, [C, T, H, W]): frames to be downsampled.
    @param scale  (int): downsampling factor: 2 | 3 | 4
    @return Returns the downscaled image as tensor
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    if scale == 2:
        h = gkern(13, 0.8)  # 13 and 0.8 for x2
    elif scale == 3:
        h = gkern(13, 1.2)  # 13 and 1.2 for x3
    elif scale == 4:
        h = gkern(13, 1.6)  # 13 and 1.6 for x4
    else:
        raise ValueError(f'Invalid upscaling factor {scale}, must be 2, 3 or 4')

    C, T, H, W = x.size()
    x = x.contiguous().view(-1, 1, H, W) # depth convolution (channel-wise convolution)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0

    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)

    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')
    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(C, T, x.size(2), x.size(3))
    return x


