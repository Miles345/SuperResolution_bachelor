##
# @file SuperRes_v1.py
# @brief Main production file with several impelementations for super resolution at colugo

## Imports
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from data import get_test_set
from pathlib import Path 
from utils import Logger
from tqdm import tqdm
from arch import RRN
import numpy as np
import datetime
import torch
import time
import cv2
import sys
import os

##
# These are all internal classes of colugo which are not publicly available and have to be replaced if this project should be used elsewhere
# They are mainly responsible as i/o-functions and have no direct impact on the core functionality of this class
#  
sys.path.append("/home/lugo/git/garden4")
import movie_man
import gl_module
import cam_man
import general as gs

git = gs.get_locale('dp_git')

class Timer():
    """! Simple class for keeping track of runtimes when GPUs are used
    """
    def __init__(self, name="Timer", printT=True):
        """! Initializes the timer and checks if GPUs are available"""
        self.name = name
        self.print = printT
        self.cuda = torch.cuda.is_available()
    def __enter__(self):
        """! Synchronizes the GPU and starts the timer"""
        if self.cuda:
          torch.cuda.synchronize()
        self.start = time.perf_counter()
    def __exit__(self):
        """! Synchronizes the GPU again end ends the timer
        @return Returns the runtime of all the operations betweeen enter and exit
        """
        if self.cuda:
          torch.cuda.synchronize()
        self.end = time.perf_counter()
        if self.print:
          print(f"{self.name} took {self.end-self.start} Seconds")
        else:
          return self.end-self.start
        
class SuperRes():   
    """! Main class for integration can be used in a multitude of ways to achive super resolution
    @param scale  Factor by which the frames are upscaled
    @param image_dir  If set the class expects a directory from which it can pull the frames
    @param image_out  If set the class will not display the upscaled frames. Instead it will write it to the provided direcotry
    """
    def __init__(self, scale, image_dir=None, image_out=None):
        """! Initialization of the SuperRes class
        """
        ## Factor by which to upscale
        self.scale = scale
        ## Dont really know if batch size is important outside of training but the DataLoader requires it
        self.batchsize = 5
        ## Threads used by the DataLoader to preload data from disk
        self.threads = 12
        ## Setting the seed for generating random numbers on the current GPU, not really improtant but doesn't cost anything so why not
        self.seed = 0
        ## For now it supports the use of one gpu. In further development you could split a frame and compute it on several gpus
        self.gpus = 1
        ## Needed to create the correct network, as not the whole network is saved, just the weights
        self.layer = 5
        ## If set the class expects a directory from which it can pull the frames
        self.image_dir = image_dir
        ## Path where the logs are saved
        self.save_test_log = os.path.join(git, "SuperResolution-RRN/RRN-master/log/test")

        ## Changes used model if the scale is set either to 4 or 2. Other upscaling factors are currently not supported
        if self.scale == 4:
            self.pretrain = "./result/weight/bachelor_not_pretrained_5L_v3/X4_5L_64_epoch_200.pth"
        elif self.scale == 2:
            self.pretrain = os.path.join(git, "SuperResolution-RRN/RRN-master/result/weight/hlrs_03/X2_5L_64_epoch_55.pth")
        else:
            raise Exception('This scale is not supported')
        ## Directory where upscaled frames are saved
        self.image_out = image_out 
        ## List of ID's of all GPUs used 
        self.gpus_list = range(self.gpus)
        ## Gets system time that is used for naming logfiles etc.
        self.systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        ## Variable to enumerate frames for the path where they are saved
        self.image_num = 0
        ## Count of neurons per layer
        self.n_c = 128
        ## Count of layers. Needed when the network is initialized as only the weights of the trained network are saved
        self.n_b = self.layer
        ## Initialize the network
        self.rrn = RRN(self.scale, self.n_c, self.n_b) 
        ## Bool used for the first frame that is processed
        self.init = True
        ## Set this if you want to display the high res video live on screen
        self.LIVE = False 
        ## Reporter to keep track of the frames per second while displaying
        self.reporter = lambda:0

        sys.stdout = Logger(os.path.join(self.save_test_log,'test_' + self.systime + '.txt'))
        if not torch.cuda.is_available():
            raise Exception('No Gpu found, please run with gpu')
        else:
            use_gpu = torch.cuda.is_available()
        if use_gpu:
            cudnn.benchmark = False
            torch.cuda.manual_seed(self.seed)
        self.pin_memory = True if use_gpu else False 

        if self.image_out != None:
            self.save_pics = True
        else: 
            self.save_pics = False
        if self.image_dir != None:
            Path(self.image_out).mkdir(parents=True, exist_ok=True)
            test_set = get_test_set(self.image_dir, self.scale)
            self.test_loader = DataLoader(dataset=test_set, num_workers=self.threads, batch_size=self.batchsize, shuffle=False, pin_memory=self.pin_memory, drop_last=False)

        self.rrn = torch.nn.DataParallel(self.rrn, device_ids=self.gpus_list)
        if os.path.isfile(self.pretrain):
            self.rrn.load_state_dict(torch.load(self.pretrain, map_location=lambda storage, loc: storage))
        else:
            raise Exception('pretrain model does not exist')
        if use_gpu:
            self.rrn = self.rrn.cuda(self.gpus_list[0])

    def getFrame(self, frame): 
        """! Transforms the numpy frames in the dimensions and format used by the network
        @param frame  frame to be converted
        @return  Returns the low resolution frame in a pytorch tensor
        """
        target = []    
        target = np.expand_dims(frame, axis=0)
        t, h, w, c = target.shape
        target = target.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        target = torch.from_numpy(target.transpose(2,0,1)/255).float()
        LR = target.view(c,t,h,w)
        LR = torch.cat((LR[:,1:2,:,:], LR), dim=1)
        return LR

    def __call__(self, frame=None):
        """! The main body of this class any SuperRes object can just be called with a frame and the frame will be upscaled by the scaling factor that was provided during initialization
        @param frame  low resolution input
        """
        self.rrn.eval()       
        if self.image_dir != None:  # If image dir is set, its expected that it is a testing scenario where frames in the image_dir will be downsampled before they are upsampled again for testing purposes
            for self.image_num, x_input in tqdm(enumerate(self.test_loader)):
                with torch.no_grad():
                    x_input = Variable(x_input).cuda(self.gpus_list[0])
                    if self.init: # If its the first frame processed it just repeats it one time as there is no "previous" low res frame
                        init_temp = torch.zeros_like(x_input[:,0:1,0,:,:])  
                        init_o = init_temp.repeat(1, self.scale*self.scale*3, 1, 1)
                        init_h = init_temp.repeat(1, self.n_c, 1, 1)
                        with torch.cuda.amp.autocast(): # Autocasts the variable for mixed precision training
                            h, prediction = self.rrn(x_input, init_h, init_o, self.init)
                        self.init = False
                    else:
                        with torch.cuda.amp.autocast():
                            h, prediction = self.rrn(x_input, h, prediction, self.init)
                    if self.save_pics == True:
                        vutils.save_image(prediction, os.path.join(self.image_out, "frame_{:05}.png".format(self.image_num)))

        else:
            with torch.no_grad():  
                self.x_input = self.getFrame(frame)
                self.x_input = Variable(self.x_input).cuda(self.gpus_list[0])               
                if self.init:
                    self.old_x_input = self.x_input
                    self.x_input = torch.cat((self.x_input, self.x_input), dim=1).unsqueeze(dim=0)
                    init_temp = torch.zeros_like(self.x_input[:,0:1,0,:,:])
                    init_o = init_temp.repeat(1, self.scale*self.scale*3, 1, 1)
                    init_h = init_temp.repeat(1, self.n_c, 1, 1)
                    with torch.cuda.amp.autocast():
                        self.h, self.prediction = self.rrn(self.x_input, init_h, init_o, self.init)
                    self.init = False
                else:   
                    self.new_x_input = torch.cat((self.x_input, self.old_x_input), dim=1).unsqueeze(dim=0)       
                    self.old_x_input = self.x_input              
                    with torch.cuda.amp.autocast():
                        self.h, self.prediction = self.rrn(self.new_x_input, self.h, self.prediction, self.init)
                self.out_prediction = self.prediction       # Copies the high resolution frames so they can be transformed in a format that can be displaied, original is used as input of the next step
                if self.save_pics == True:
                    vutils.save_image(self.out_prediction, os.path.join(self.image_out, "frame_{:05}.png".format(self.image_num))) # Saves the superres output if the variable is set 
                    self.image_num += 1
            self.out_prediction = torch.clamp(self.out_prediction, min=0, max=1)
            
            if self.LIVE == True:
                return self.out_prediction.squeeze().permute(1,2,0)
            else:
                return self.out_prediction.squeeze().permute(1,2,0).cpu().numpy()*255

    def srVid(self, video, out):
        """! This function can be used to upscale a pre recorded video. If you wan to change the framerate of the output video change the 30 in line 209. 
        @param video  Pre recorded LR video that you want to upscale
        @param out  Directory where to save the HR video to
        """
        self.returnTensor = True
        if os.path.isdir(out):
            out = os.path.join(out, video.split("/")[-1] + "_upscaled.avi")
        ms = movie_man.MovieSaver(out, 30, auto_finalize=True)
        cap = cv2.VideoCapture(video)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(framecount//16)):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sr_out = self.__call__(frame)
            ms.write_frame(sr_out)
    
    def live_frameiterator(self, cap):
        """! Iterator that is used by the liveUpscale function to provide the frames in a pre recorded video
        @param cap  cv2.VideoCapture object that is iterated over
        """
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 0
        for i in range(framecount//32):
            ret, frame = cap.read()
            print(i)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sr_out = self.__call__(frame)*255
            yield sr_out

    def liveUpscale(self, video):
        """! This class can be called if you want to upscale a video and want it to be shown live on screen
        @param video  Path to LR video you want to upscale
        """
        self.LIVE = True
        self.returnTensor = True
        cap = cv2.VideoCapture(video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))*self.scale
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))*self.scale
        gen = self.live_frameiterator(cap)
        gl_man = gl_module.GLman((height,width), gen)
        gl_man.start()  

    def cam_frameiterator(self, cap):
        """! Iterator for the "fromCam" function used in case an video stream from a webcan should be upscaled
        @param cap  cam_man.CamManager object - An Colugo internal class for handling video streams from webcams
        """
        while True:
            t0 = time.time()
            frame = cap.get_img()
            frame = self.__call__(frame)
            t1 = time.time()
            fps = 1 / (t1-t0)
            self.reporter.msg=str(int(fps))
            yield frame

    def fromCam(self):
        """! Can directly take input from a webcam and display the upscaled video stream
        """
        cam = cam_man.CamManager(-1)  
        gen = self.cam_frameiterator(cam) 
        gl_man = gl_module.GLman((960,1728), gen, reporter=self.reporter)
        gl_man.start() 

    def allonCUDA(self, frame=None):
        """! Used for integration in rtsf_demo_parallel. It keeps the frame all the time on the gpu, which saves processing time.
        The drawback of this class is that it used a pretty special format that you have to fit to your usecase.
        Use this function as template for further integration in an live environment
        @param frame  LR frame in an torch tensor that already resides on the GPU
        @return  Returns the high resolution frame in an torch tensor
        """
        with torch.no_grad():  
            self.x_input = torch.unsqueeze(frame, dim=0).permute(3,0,1,2)
            if self.init:
                self.old_x_input = self.x_input
                self.x_input = torch.cat((self.x_input, self.x_input), dim=1).unsqueeze(dim=0)
                init_temp = torch.zeros_like(self.x_input[:,0:1,0,:,:])
                init_o = init_temp.repeat(1, self.scale*self.scale*3, 1, 1)
                init_h = init_temp.repeat(1, self.n_c, 1, 1)
                with torch.cuda.amp.autocast():
                    self.h, self.prediction = self.rrn(self.x_input, init_h, init_o, self.init)
                self.init = False
            else:   
                self.new_x_input = torch.cat((self.x_input, self.old_x_input), dim=1).unsqueeze(dim=0)       
                self.old_x_input = self.x_input              
                with torch.cuda.amp.autocast():
                    self.h, self.prediction = self.rrn(self.new_x_input, self.h, self.prediction, self.init)
            self.out_prediction = self.prediction
        self.out_prediction = torch.clamp(self.out_prediction, min=0, max=1)
        return self.out_prediction.squeeze().permute(1,2,0)



######### EXAMPLE AREA #########

        
#%%
#if __name__=='__main__':
#    SuperRes(2)

#%% Run for video superres of folder

#image_dir=os.path.join(git, "SuperResolution-RRN/RRN-master/testImgs/eggsanimals.mp4_downscaled")
#image_out=os.path.join(git, "SuperResolution-RRN/RRN-master/out/testing_class")
#superres = SuperRes(4,image_dir=image_dir, image_out=image_out)

#superres()

# %% Run for frame by frame superres
#image_dir="/mnt/8tbd/Superres_datafolder/Superres_testImgs_lowres"
#fp_images = [os.path.join(image_dir, i) for i in sorted(os.listdir(image_dir))]

#superres = SuperRes(4)
#for i in tqdm(fp_images):
#    im = Image.open(i)
#    sr_arr = superres(frame=im)
#    sr_img = Image.fromarray(sr_arr.astype(np.uint8))
#    sr_img.save(os.path.join("/mnt/8tbd/Superres_datafolder/Superres_upscaled/not_pretrained_v3", os.path.basename(i)))


#superres = SuperRes(4)
#im = Image.open("/home/lugo/git/SuperResolution-RRN/RRN-master/testing_downscale/testfile.png")
#sr_arr = superres(frame=im)
#sr_img = Image.fromarray(sr_arr.astype(np.uint8))
#sr_img.save("/home/lugo/git/SuperResolution-RRN/RRN-master/testing_downscale/upscaled2.png")
    
# %% Run if you want to upscale a video
#superres = SuperRes(4)
#superres.srVid("./RRN-master/testMovies/eggsanimals.mp4", "./RRN-master/testMovies/eggs_times4_test.mp4")

# %% Run if you want to live upscale
#superres = SuperRes(2)
#superres.liveUpscale("./RRN-master/trainMovies/dubai-bw_artists.mp4")

# %% Run if you want to live upscale
#superres = SuperRes(4)
#superres.fromCam()
#superres.liveUpscale("./RRN-master/testMovies/eggsanimals.mp4")