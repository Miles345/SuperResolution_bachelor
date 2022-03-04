##
# @file load_train.py
# @brief DataLoader for training. It loads frames from storage and generates the low resolution versions. 

## Imports
from Gaussian_downsample import gaussian_downsample
import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
import random
import torch
import os

def load_img(index, image_filenames):            
    """! Modified image loader that crops out a random part of the image. Only a certain size of frame can be used as we otherwise run into memory problems
    @param index  Index of frame to be loaded
    @param image_filenames  List of frames in the training folder
    @return  returns seven high resolution images after they are cropped 
    """
    HR = []
    ## Cropping factor of all 7 frames
    r = 5
    GT_temp = Image.open(image_filenames[index]).convert('RGB')
    crop_size = (GT_temp.size[0]//r, GT_temp.size[1]//r)          # Get image to calculate cropped framesize
    first_run = True
    if index + 7 > len(image_filenames):    # Check if all image indexes are < max
        index = random.randint(0, len(image_filenames)-7)      # Get index that is for sure possible
    for img_num in range(7):
        if first_run == True:                       # On the first run get position and size of cropwindow
            max_x = GT_temp.size[1] - crop_size[1]         
            max_y = GT_temp.size[0] - crop_size[0]
            x = np.random.randint(20, max_x-20)
            y = np.random.randint(20, max_y-20)
            first_run = False
        else:
            GT_temp = Image.open(image_filenames[index+img_num]).convert('RGB')     # Load image if not first run
        crop = GT_temp.crop((y, x, y + crop_size[0],x + crop_size[1]))                     # Crop here
        HR.append(crop)
    return HR

def train_process(GH, flip_h=True, rot=True): 
    """! if data_augmentation is set loaded iamges are getting flipped and rotated at random
    @param GH  frames to be rotated
    @param flip_h  If set frames get flipped
    @param rot  If set frames get rotated
    @return  Returns the augmented frames
    """
    if random.random() < 0.5 and flip_h: 
        GH = [ImageOps.flip(LR) for LR in GH]
    if rot:
        if random.random() < 0.5:
            GH = [ImageOps.mirror(LR) for LR in GH]
    return GH

class DataloadFromFolder(data.Dataset): # load train dataset
    """! Main DataLoader class for loading training data
    """
    def __init__(self, image_dir, scale, data_augmentation, transform):
        """!
        @param image_dir  Directory where raining data is located
        @param scale  Upscaling factor of the network
        @param data_augmentation  If set data gets augmented as seen in function "train_process"
        @param transform  construct that transforms the numpy array to a pytorch tensor
        """
        super(DataloadFromFolder, self).__init__()
        alist = os.listdir(image_dir)
        alist.sort()
        self.image_filenames = [os.path.join(image_dir, x) for x in alist] 
        self.scale = scale
        self.transform = transform # To_tensor
        self.data_augmentation = data_augmentation # flip and rotate
    def __getitem__(self, index):
        """! Each time the DataLoader is used this function is called.
        @param index  the index of the first frame of the batch
        @return  returns the LR and GT frames
        """
        GT = load_img(index, self.image_filenames)          # Modified image loader
        if self.data_augmentation:
            GT = train_process(GT) # input: list (contain PIL), target: PIL
        GT = [np.asarray(HR) for HR in GT]  # PIL -> numpy # input: list (contatin numpy: [H,W,C])
        GT = np.asarray(GT) # numpy, [T,H,W,C]
        T,H,W,C = GT.shape
        t, h, w, c = GT.shape
        GT = GT.transpose(1,2,3,0).reshape(h, w, -1) # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']
        GT = GT.view(c,t,h,w) # Tensor, [C,T,H,W]
        LR = gaussian_downsample(GT, self.scale)
        LR = torch.cat((LR[:,1:2,:,:], LR), dim=1)
        return LR, GT

    def __len__(self):
        """! DataLoader always needs the count of the data it should process
        @return  Returns the count of files in the frame directory
        """
        return len(self.image_filenames) 

