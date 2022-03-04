##
# @file psnr_ssim.py
# @brief Utility method for calculatiing peak-signal-to-noise-ratio and the structural similarity index

## Imports
import math
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import pickle
from tqdm import tqdm
import random


def psnr(img1, img2):
    """! Simple approach for calculating the psnr between two rgb images
    @param img1  Ground truth image
    @param img2  Image to be measured
    @return  returns the psnr 
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim_own(original, upscaled):
    """! Calculates the SSIM index between two images. Sets all needed parameters to calculate ssim with skimage.metrics.structural_similarity
    @param original  ground truth 
    @param upscaled  upscaled test frame
    @return  returns the SSIM index
    """
    score, diff = ssim(original, upscaled, channel_axis=3, multichannel=True, full=True)
    return score

def measure():
    """! main function to measure the PSNR and SSIM of the different models"""
    fd_original = "/mnt/8tbd/Superres_datafolder/Superres_testImgs"
    fd_upscaled_paper = "/mnt/8tbd/Superres_datafolder/Superres_upscaled/paper_5L/"
    fd_upscaled_pretrained = "/mnt/8tbd/Superres_datafolder/Superres_upscaled/pretrained_5L/"
    fd_upscaled_not_pretrained = "/mnt/8tbd/Superres_datafolder/Superres_upscaled/not_pretrained_v3/"

    imgs_upscaled_paper = sorted(os.listdir(fd_upscaled_paper))[:100]
    imgs_upscaled_pretrained = sorted(os.listdir(fd_upscaled_pretrained))[:100]
    imgs_upscaled_not_pretrained = sorted(os.listdir(fd_upscaled_not_pretrained))[:100]
    imgs_original = sorted(os.listdir(fd_original))[:100]
    val_list = list()

    count_pics = 50#len(os.listdir(fd_original))
    count_samples = 1000

    for _ in tqdm(range(count_samples)):
        innerlist = list()
        idx = random.randint(0, count_pics)
        original = cv2.imread(os.path.join(fd_original, imgs_original[idx]))

        upscaled = cv2.imread(os.path.join(fd_upscaled_paper, imgs_upscaled_paper[idx]), 1)
        innerlist.append(ssim_own(original, upscaled))

        upscaled = cv2.imread(os.path.join(fd_upscaled_pretrained, imgs_upscaled_pretrained[idx]), 1)
        innerlist.append(ssim_own(original, upscaled))

        upscaled = cv2.imread(os.path.join(fd_upscaled_not_pretrained, imgs_upscaled_not_pretrained[idx]), 1)
        innerlist.append(ssim_own(original, upscaled))


        val_list.append(innerlist)

    with open("/home/lugo/git/SuperResolution-RRN/bachelorthesis/measurements/ssim_v3.pickle", "wb") as handle:
        pickle.dump(val_list, handle)
    outsum1 = list()
    outsum2 = list()
    outsum3 = list()
    for i in val_list:
        outsum1.append(i[0])
        outsum2.append(i[1])
        outsum3.append(i[2])
    print("paper:")
    print(sum(outsum1)/len(outsum1))
    print("pretrained")
    print(sum(outsum2)/len(outsum2))
    print("not_pretrained")
    print(sum(outsum3)/len(outsum3))


img_original = "/mnt/8tbd/Superres_datafolder/Superres_testImgs/210918_1_flashmob_001443.png"
img_upscaled = "/mnt/8tbd/Superres_datafolder/Superres_upscaled/not_pretrained_v3/testimg_000000.png"
img_paper = "/mnt/8tbd/Superres_datafolder/Superres_upscaled/paper_5L/testimg_000000.png"

original = cv2.imread(img_original)
upscaled = cv2.imread(img_upscaled)
paper = cv2.imread(img_paper)

print(f"upscaled : {psnr(upscaled, original)}")
print(f"paper: {psnr(paper, original)}")


measure()
#print(Brisque.score(paper))