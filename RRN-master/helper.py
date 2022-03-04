##
# @file helper.py
# @brief This file houses several funktions for standard image/video processing operations.
# It solely exists to support the development of this project and is not directly integrated in the training/production processes.


## Imports
from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle
import torch
import math
import cv2
import sys
import os

sys.path.append("/home/lugo/mmlab_testing/RRN-master/RRN")

def pngToVid():
    """! Used to convert a stream of images to a video, src is the folder of the frames to be processed. 
    In cv2.VideoWrite can the output and format of the output be specified. Watch out for memory problems in large folders"""
    src = "/mnt/8tbd/Superres_datafolder/Superres_testImgs"
    img_list = list()
    for picpath in tqdm(sorted(os.listdir(src))[:1000]):
        img = cv2.imread(os.path.join(src, picpath))
        height, width, layers = img.shape
        size = (width, height)
        img_list.append(img)
    out = cv2.VideoWriter('/home/lugo/git/SuperResolution-RRN/RRN-master/out/testing_bachelor_original.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()

def readVidToImgs():
    """! Splits videos in a specified folder in its single frames and saves them in a folder"""
    srcfolder = "/home/lugo/git/SuperResolution-RRN/RRN-master/trainMovies/"
    destfolder = "/mnt/8tbd/Superres_trainImgs/"
    for filename in tqdm(os.listdir(srcfolder)):
        abspath = os.path.join(srcfolder, filename)
        dest = os.path.join(destfolder, filename)
        print("Check if Path exists\n")
        if not os.path.exists(dest):
            os.makedirs(dest)
            print("Created path" + filename)
        vid = cv2.VideoCapture(abspath)
        print(f"video {filename} opened")
        idx = 0
        while(vid.isOpened()):
            ret, frame = vid.read()
            idx += 1
            if ret:
                cv2.imwrite(os.path.join(dest, filename[:-4] + '_' + str(format(idx, '06d')) + ".png"), frame)
            else:
                break
        vid.release()
        cv2.destroyAllWindows()  
        print(f"finnished video {filename}")

def crop_images():
    """! Used for testing the image cropping"""
    r = 5
    fd = os.listdir("/home/lugo/mmlab_testing/RRN-master/trainImages")
    image_filenames = [os.path.join("/home/lugo/mmlab_testing/RRN-master/trainImages", i) for i in fd if i.startswith('1_012')]
    first_run = True
    for index in range(len(image_filenames)):
        GT_temp = Image.open(image_filenames[index]).convert('RGB')
        if first_run == True:
            crop_size = (GT_temp.size[0]//r, GT_temp.size[1]//r)
            max_x = GT_temp.size[1] - crop_size[1]         
            max_y = GT_temp.size[0] - crop_size[0]
            x = np.random.randint(20, max_x-20)
            y = np.random.randint(20, max_y-20)
            first_run = False
        crop = GT_temp.crop((y, x, y + crop_size[0],x + crop_size[1]))

        crop.save(os.path.join("/home/lugo/mmlab_testing/RRN-master/testimages/test_hyper/", image_filenames[index].split("/")[-1]), "PNG")

def ds_image():
    """! Function to downscale the image and resample it with PIL.Image.BICUBIC"""

    df = "/home/lugo/git/SuperResolution-RRN/RRN-master/testing_downscale/210919_9_tango_beautiful_001800.png"
    file = Image.open(df).convert('RGB')

    resized = file.resize((file.size[0]//4, file.size[1]//4), resample= Image.BICUBIC)
    resized.save("/home/lugo/git/SuperResolution-RRN/RRN-master/testing_downscale/testfile.png")

def psnr():
    """! Testing the PSNR calculation"""
    srcfolder_upscaled = "./RRN-master/out/eggsanimals_hlrs_03/"
    srcfolder_original = "./RRN-master/testImgs/eggsanimals.mp4/"
    upscaled_files = [srcfolder_upscaled + i for i in sorted(os.listdir(srcfolder_upscaled))]
    original_files = [srcfolder_original + i for i in sorted(os.listdir(srcfolder_original))]
    psnr_list = np.empty(len(upscaled_files))
    if len(upscaled_files) != len(original_files):
        raise Exception("Two Folders don't contain the same amount of files")
    
    for i in tqdm(range(len(upscaled_files))):
        img_original = cv2.imread(original_files[i])
        img_upscaled = cv2.imread(upscaled_files[i])
        mse = np.mean( (img_original/255. - img_upscaled/255.) ** 2 )
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        psnr_list[i] = (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
    with open("./RRN-master/log/psnr/eggsanimals_hlrs_03.pickle", "wb") as handle:
        pickle.dump(psnr_list, handle)
    return np.mean(psnr_list)

# Threaded vidToImg

#srcfolder = "/home/lugo/git/SuperResolution-RRN/RRN-master/trainMovies/"
#filelist = os.listdir(srcfolder)
#pool_of_threads = Pool(multiprocessing.cpu_count() - 1)
#pool_of_threads.map(readVidToImgs, filelist)

#readVidToImgs()
#makeImageList()
#downscale_images()

#pngToVid()
#print(psnr())
#ds_image()
#own_downsample1()