##
# @file load_test.py
# @brief Heavily modified DataLoader. This DataLoader is used to load test data for the SuperRes_v1 method

## Imports
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import os

def modcrop(img,scale):
    """! Crops an image to a size thats multiplicable with scaling factor
    @param img  Frame to be cropped
    @param scale  super resolution factor 
    @return  Returns the cropped image
    """
    (iw, ih) = img.size
    ih = ih - (ih % scale)
    iw = iw - (iw % scale)
    img = img.crop((0,0,iw,ih))
    return img

class DataloadFromFolderTest(data.Dataset): # load test dataset
    """! DataLoader for testruns. It loads the frames like the training loader but doesn't downscale them"""
    def __init__(self, image_dir, scale, transform):
        """!
        @param image_dir  Directory where the frames are located
        @param scale  Scale of the super resolution factor
        @param transform  transforms the numpy array to a pytorch tensor
        """
        super(DataloadFromFolderTest, self).__init__()
        alist = os.listdir(image_dir)
        alist.sort()
        self.image_filenames = [os.path.join(image_dir, x) for x in alist] 
        self.L = len(alist)
        self.scale = scale
        self.transform = transform # To_tensor
    def __getFrame__(self, index):
        """! Get frame is called by __getItem__ out of structural reasons. It handles the loading of the frames and formats them to a torch tensor
        @param index  Index of the frame to be loaded
        @return  Returns the low resolution frame
        """
        target = []
        try:
            GT_temp = modcrop(Image.open(self.image_filenames[index]).convert('RGB'), self.scale)
        except IndexError:
            GT_temp = modcrop(Image.open(self.image_filenames[index-1]).convert('RGB'), self.scale)
        target.append(GT_temp)
        target = [np.asarray(HR) for HR in target] 
        target = np.asarray(target)
        t, h, w, c = target.shape
        target = target.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        if self.transform:
            target = self.transform(target) # Tensor, [CT',H',W']
        LR = target.view(c,t,h,w)
        LR = torch.cat((LR[:,1:2,:,:], LR), dim=1)
        return LR
        
    def __getitem__(self, index):
        """! Executes each time the DataLoader is called
        @param index  Index of frame to be loaded
        """
        targetframe = self.__getFrame__(index-1)
        targetframe2 = self.__getFrame__(index)
        return torch.cat((targetframe, targetframe2), dim=1)
      
    def __len__(self):
        """! DataLoader always needs the count of the data it should process
        @return  Returns the count of files in the frame directory
        """
        return len(self.image_filenames)

