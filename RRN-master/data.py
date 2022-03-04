##
# @file data.py
# @brief This file links the main files with the data loaders for test and training scenarios. To also enables the automatic transformation to pytorch tensors

## Imports
from load_train import DataloadFromFolder
from load_test import DataloadFromFolderTest
from torchvision.transforms import Compose, ToTensor

def transform():
    """! Transforms the loaded data to pytorch tensors
    """
    return Compose([
             ToTensor(),
            ])

def get_training_set(data_dir, upscale_factor, data_augmentation):
    """! Returns the data loader object for training scenarios
    @param data_dir  Directory where the training data lies
    @param upscale_factor  Factor by which the frames should be upscaled. Important for generating the low resolution training data
    @param data_augmentation  If set the data gets augmented when loaded
    @return  Returns the configured DataLoader  
    """
    return DataloadFromFolder(data_dir, upscale_factor, data_augmentation, transform=transform())

def get_test_set(data_dir, upscale_factor):
    """! Returns the data loader object for testing scenarios automatically downsamples frames so the network can be tested on a video without needing to downscale them first
    @param data_dir  Directory where the training data lies
    @param upscale_factor  Factor by which the frames should be upscaled. Important for generating the low resolution testing data
    @param data_augmentation  If set the data gets augmented when loaded
    @return  Returns the configured DataLoader  
    """
    return DataloadFromFolderTest(data_dir, upscale_factor, transform=transform())
