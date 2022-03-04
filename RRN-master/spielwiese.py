########################################################################
# FILE TO TRY OUT DIFFERENT THINGS. NOT PART OF ANY IMPORTANT FUNCTIONS#
########################################################################

import numpy as np
import cv2
from time import time
import pickle
import skimage
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os


# %%

def plot_ssim():
    # Plot SSIM in comparison to upscaled image

    srcfolder_upscaled = "/home/lugo/git/SuperResolution-RRN/RRN-master/out/eggsanimals_hlrs_03/"
    srcfolder_original = "/home/lugo/git/SuperResolution-RRN/RRN-master/testImgs/eggsanimals.mp4/"
    upscaled_files = [srcfolder_upscaled + i for i in sorted(os.listdir(srcfolder_upscaled))]
    original_files = [srcfolder_original + i for i in sorted(os.listdir(srcfolder_original))]

    if len(upscaled_files) != len(original_files):
        raise Exception("Two Folders don't contain the same amount of files")
    first = True
    for i in range(len(upscaled_files)):
        img_original = cv2.imread(original_files[i])
        img_upscaled = cv2.imread(upscaled_files[i])
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_upscaled = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2RGB)
        score, diff = ssim(img_original, img_upscaled, multichannel=True, full=True)
        if first == True:
            fig, axis = plt.subplots(1,2, figsize=(15,15)) 
            plt.subplots_adjust(top=0.88)
            plt.tight_layout()
            axis[1].title.set_text('Low Res Image')
            axis[0].title.set_text('SSIM')
            img1 = axis[0].imshow(diff)
            img2 = axis[1].imshow(img_original)
            fig.suptitle(f"Frame: {i}")
            plt.gcf()
            plt.show(block=False)
            first = False
        else:
            fig.suptitle(f"Frame: {i}")
            axis[1].title.set_text('Low Res Image')
            axis[0].title.set_text('SSIM')
            img1.set_data(diff)
            img2.set_data(img_upscaled)
            plt.draw()
            plt.pause(.01)


#%%
#print loss 

def print_loss():
    import pickle
    import matplotlib.pyplot as plt

    with open('/home/lugo/git/SuperResolution-RRN/result/weight/bachelor_not_pretrained/v2/testing_loss/testing_loss_batch_12_no_norm.pickle', 'rb') as handle:
        content = pickle.load(handle)
    with open('/home/lugo/git/SuperResolution-RRN/result/weight/bachelor_not_pretrained/v2/testing_loss/testing_loss_batch_4_no_norm.pickle', 'rb') as handle:
        content1 = pickle.load(handle)
    with open('/home/lugo/git/SuperResolution-RRN/result/weight/bachelor_not_pretrained/v2/testing_loss/testing_loss_batch_12_no_norm_multinode.pickle', 'rb') as handle:
        content2 = pickle.load(handle)
    with open('/home/lugo/git/SuperResolution-RRN/result/log/loss_bachelor_not_pretrained_5L_v3.pickle', 'rb') as handle:
        content3 = pickle.load(handle)

    def chunks(l, n):
        n = max(1, n)
        return (l[i:i+n] for i in range(0, len(l), n))

    newlist = list()

    q = chunks(content3,100)
    for i in q:
        newlist.append(sum(i)//len(i))
    
    fig, axs = plt.subplots(4)
    axs[0].plot(range(len(content[:1750])), content[:1750])
    axs[0].set_title('Batch size 12 old')
    axs[1].plot(range(len(content1[:1750])), content1[:1750])
    axs[1].set_title('Batch size 4 old')
    axs[2].plot(range(len(content2[:1750])), content2[:1750])
    axs[2].set_title('Batch size 12')
    axs[3].plot(range(len(content3)), content3)
    axs[3].set_title('Batch size 4')
    #plt.plot(range(len(newlist)), newlist)
    #plt.xlabel("Epoch")
    #plt.ylabel("Loss")
    #plt.title("Loss of pretrained model")
    #plt.savefig("/home/lugo/git/SuperResolution-RRN/bachelorthesis/loss_pretrained.png")
    plt.show()



# %%
print_loss()


