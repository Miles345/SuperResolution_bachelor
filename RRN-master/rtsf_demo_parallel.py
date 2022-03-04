
#####################################################################################################################################
# THIS IS AN INTEGRATION EXAMPLE OF THE REAL TIME STYLE TRANSFER IMPLEMENTATION OF COLUGO THIS DOESNT DIRECTLY BELONG TO THIS THESIS#
#####################################################################################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:14:59 2021

@author: turbid
"""

import sys, os
import numpy as np
import cv2
sys.path.append(".")

dp_git = os.path.join(os.path.dirname(os.path.realpath(__file__)).split("git")[0]+"git")
sys.path.append(os.path.join(dp_git,'garden4'))
from cam_man import CamManager
import general as gs
from tiling import Tiling
from contextlib import contextmanager
from timeit import default_timer
from threading import Thread
import copy
import u_torch
from u_torchgl import OpenGLRenderer
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.join(dp_git, "SuperResolution-RRN/RRN-master"))
import SuperRes_v1 as SuperRes  
from cam_man import MultiIPWebcam

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
    print('it took: {}'.format(elapser()))

# also disable grad to save memory
import torch
torch.set_grad_enabled(False)

use_multi_cam = False

gpu_def = [0,0,0,0]
gpu_models = []


import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

print('loading models....')

# config32x32 = load_config("logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt")    

# config32x32 = load_config("logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt")

# config32x32 = load_config("logs/vqgan_gumbel_f8/configs/model.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/vqgan_gumbel_f8/checkpoints/last.ckpt", is_gumbel=True)    

# config32x32 = load_config("logs/2021-07-19T19-21-25_custom_vqgan/configs/2021-07-19T19-21-25-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/2021-07-19T19-21-25_custom_vqgan/checkpoints/last.ckpt")    

# config32x32 = load_config("logs/2021-07-20T08-20-54_custom_vqgan/configs/2021-07-20T08-20-54-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/2021-07-20T08-20-54_custom_vqgan/checkpoints/last.ckpt")    

# s_id = '2021-07-21T13-30-55'
# config32x32 = load_config("logs/"+s_id+"_custom_vqgan/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+s_id+"_custom_vqgan/checkpoints/last.ckpt")    

# s_id = '2021-07-23T10-51-24'
# config32x32 = load_config("logs/"+s_id+"_custom_vqgan/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+s_id+"_custom_vqgan/checkpoints/last.ckpt")    

# s_id = '2021-07-27T09-40-11'; k_id = s_id.split('_')[0] + '_custom_vqgan_imagenet_r1s0.57_63FG2'
# config32x32 = load_config("logs/"+k_id+"/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+k_id+"/checkpoints/last.ckpt")    

###
# s_id = '2021-07-28T11-29-59'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.57_63FG2'
# config32x32 = load_config("logs/"+k_id+"/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+k_id+"/checkpoints/last.ckpt")

# s_id = '2021-07-27T09-43-47'; k_id = s_id.split('_')[0] + '_custom_vqgan_imagenet_r1s0.5_DQ09A'
# config32x32 = load_config("logs/"+k_id+"/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+k_id+"/checkpoints/last.ckpt")    

# s_id = '2021-07-28T11-27-41'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.5_DQ09A'
# config32x32 = load_config("logs/"+k_id+"/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+k_id+"/checkpoints/last.ckpt")    
###
# s_id = '2021-08-02T16-41-47'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.5_DQ09A'
# config32x32 = load_config("logs/"+k_id+"/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+k_id+"/checkpoints/last.ckpt")    

# s_id = '2021-07-27T09-43-27'; k_id = s_id.split('_')[0] + '_custom_vqgan_imagenet_r1s0.5_7-Grand-Floral-Velvet-1013-Mulberry-7'
# config32x32 = load_config("logs/"+k_id+"/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+k_id+"/checkpoints/last.ckpt")    

# s_id = '2021-07-28T11-30-16'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.5_7-Grand-Floral-Velvet-1013-Mulberry-7'
# config32x32 = load_config("logs/"+k_id+"/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+k_id+"/checkpoints/last.ckpt")    

###
# s_id = '2021-08-02T16-41-55'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.5_7-Grand-Floral-Velvet-1013-Mulberry-7'
# config32x32 = load_config("logs/"+k_id+"/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+k_id+"/checkpoints/last.ckpt")    

# s_id = '2021-07-21T14-28-45'
# config32x32 = load_config("logs/"+s_id+"_custom_vqgan_large/configs/"+s_id+"-project.yaml", display=False)
# model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+s_id+"_custom_vqgan_large/checkpoints/last.ckpt")    

# Testing models:
# s_id = '2021-07-28T11-29-59'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.57_63FG2' # generic
#s_id = '2021-08-05T16-10-41'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.5_DQ09A' # Codebook weigting from 1 to 1.5
#s_id = '2021-08-05T13-21-32'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.5_DQ09A' # num res blocks from 2 to 5
# s_id = '2021-08-02T16-41-47'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.5_DQ09A' # Alex ver
# s_id = '2021-08-10T14-21-07'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.5_DQ09A' # halfed c to 64
# s_id = '2021-08-11T11-34-34'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.57_63FG2' # New Style and halfed c
# s_id = '2021-08-13T10-35-43'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_styletransfer_1' # Styletransfer_1
# s_id = '2021-08-13T10-57-33'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_styletransfer_2' # Styletransfer_2
# s_id = '2021-08-13T12-59-56'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.57_63FG2' # half of pics
# s_id = '2021-08-13T15-30-06'; k_id = s_id.split('_')[0] + '_custom_vgan_imagenet_r1s0.5_DQ09A_c32' # c = 32

#s_id = '2021-08-13T15-30-06'; k_id = s_id.split('_')[0] + '_custom_vgan_imagenet_r1s0.5_DQ09A_c32' # c = 32
#config32x32 = load_config("logs/"+k_id+"/configs/"+s_id+"-project.yaml", display=False)
#model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+k_id+"/checkpoints/last.ckpt")

s_id = '2021-08-10T14-21-07'; k_id = s_id.split('_')[0] + '_custom_vqgan_large_imagenet_r1s0.5_DQ09A'
config32x32 = load_config("logs/"+s_id+"_custom_vqgan_large_imagenet_r1s0.5_DQ09A/configs/"+s_id+"-project.yaml", display=False)
model32x32 = load_vqgan(config32x32, ckpt_path="logs/"+s_id+"_custom_vqgan_large_imagenet_r1s0.5_DQ09A/checkpoints/last.ckpt").cuda(0)

for g in gpu_def:
    DEVICE_model = torch.device("cuda:"+str(g))
    gpu_models.append(copy.deepcopy(model32x32).to(DEVICE_model))

print('loading models finished')

import os, sys
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def preprocess(img, target_image_size=256):
    s = min(img.size)
    
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)


#%%#
shape_hw=(1080+1080//2,1920+1920//2)
shape_hw=(1080,1920)
shape_hw = (448,800)
# shape_hw = (576,1024)

if use_multi_cam:
    mipw = MultiIPWebcam(['10.4.1.46', '10.4.1.50', '10.4.1.47','10.4.1.44'])
else:
    cam_man = CamManager(use_ipwebcam=False, shape_hw=shape_hw, ip_webcam_address='192.168.178.60:8080')

shape_output = (1080,1920)
#shape_output=(1024*2,1024*4)
# shape_hw = (360, 640)
# shape_hw = (480, 640)
# shape_hw = (448,800)
# surf = gs.pg_init(shape_output)

fps_limit = 60
reporter = lambda:0
reporter.msg = 'boom!'

ctrl_obj = lambda:0
ctrl_obj.tlast =  0
ctrl_obj.gl_man =  OpenGLRenderer(gpu_def[0], gpu_def[0], is_fullscreen=False, reporter=reporter, keycallback=None, mousecallback=None, fps_limit = fps_limit, benchmarker=None)
ctrl_obj.schleifing_threshold = 0.008*255
# ctrl_obj.schleifing_threshold = 0.02*255
ctrl_obj.schleifing_threshold = 0.05*255
# ctrl_obj.schleifing_threshold = 0.8*255
ctrl_obj.use_schleifing = True
ctrl_obj.last_schleiging_image = []

if use_multi_cam:
    frame = mipw.list_imgs[0]
else:
    frame = cam_man.get_img()

#%%#
superres_factor = 4

size=384
size=512
size=[64,128]
size=[128,256]
#size=[256,512]
#size=[270,480]
#size=[1080//4,1920//4]
#size=[256+128,512+256]
# size=[512,1024]
# size=[512+512//2,1024+1024//2]
# size=[1024,2048]

#size=768
# size = 1024
# size=1024+512

tiling_obj = Tiling((size[0]*2,size[1]*2), backend='torch', gpu=gpu_def[0])
tiling_obj.pm.set_verbose_level(3)
tiling_obj.set_tiles(fract_overlap=0.1, div_factor=16, nmb_y=2, nmb_x=2)
# tiling_obj.set_tiles(fract_overlap=0.05, div_factor=16, nmb_y=2, nmb_x=2)

if superres_factor > 1:
    superres = SuperRes.SuperRes(superres_factor)

def reconstruct_with_vqgan(x, model):
    z, _, [_, _, indices] = model.encode(x)
    xrec = model.decode(z)
  
    return xrec

def prep_image(frame, vq_size, gpu):
    img = frame.permute([0,3,1,2]) / 255
    img = (1 - 2 * 0.1) * img + 0.1
    
    return img

def prep_output(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    output = (x)
    output = torch.flip(output, dims=[0])
    return output

gpu_holder = []
for g in range(len(gpu_def)):
    gpu_holder.append(lambda:0)
    ctrl_obj.last_schleiging_image.append(None)
    
def do_vq_parallel(g):
    tile = gpu_holder[g].tile
    
    tile_torch = torch.from_numpy(tile.copy()).cuda(gpu_def[g]).float()
    
    # get temporal frame difference for schleifing
    if ctrl_obj.last_schleiging_image[g] is not None and ctrl_obj.use_schleifing:
        diff = torch.abs(tile_torch - ctrl_obj.last_schleiging_image[g])
        diff = diff.mean(2)
        diff = diff.unsqueeze(2).repeat(1,1,3)
        
        tile_torch[diff < ctrl_obj.schleifing_threshold] = ctrl_obj.last_schleiging_image[g][diff < ctrl_obj.schleifing_threshold]    
    
    ctrl_obj.last_schleiging_image[g] = tile_torch.clone()
    x = prep_image(tile_torch[None], tile.shape, g)
    with torch.no_grad():
        result = reconstruct_with_vqgan(preprocess_vqgan(x), gpu_models[g])[0].permute(1,2,0)
    return result
    
# prepare thread jobs
def rf1():
    gpu_holder[0].result = do_vq_parallel(0)

def rf2():
    gpu_holder[1].result = do_vq_parallel(1)

def rf3():
    gpu_holder[2].result = do_vq_parallel(2)

def rf4():
    gpu_holder[3].result = do_vq_parallel(3)
        
thread_jobs = [rf1,rf2,rf3,rf4]

def get_camera_image():
    use_multi_cam = False
    if use_multi_cam:
        mipw.refresh()
        idx_cam = 0
        img = mipw.list_imgs[0].astype(np.float32)*0.5 + mipw.list_imgs[3].astype(np.float32)*0.5
    else:
        img = cam_man.get_img().astype(np.float32)
    # img = mipw.list_imgs[idx_cam].astype(np.float32)
    
   
    if True:
        img = img/np.median(img)*100
    
    img = cv2.resize(img, (size[1]*2,size[0]*2))
    # img = cv2.resize(img, (ctrl_obj.shape_cam[1],ctrl_obj.shape_cam[0]))
    
    # img = 255.0*img / np.max(img)
    # img += cam_noise_img*ctrl_obj.cam_noise_coef
    # img *= ctrl_obj.img_intensity_mod
    
    # if use_double_cam_mode:
    #     img_usb_halal = ctrl_obj.cam_man_usb_halal.get_img().astype(np.float32)
    #     img[:,img.shape[0]:,:] = img_usb_halal[:,img.shape[0]:,:]
    
    img[img > 255] = 255
    
    return img



def render_func():
    
    reporter.msg = 'fps {}'.format(1/(time.time() - ctrl_obj.tlast))
    ctrl_obj.tlast =  time.time()
    
    frame = get_camera_image()
    # frame = cv2.resize(frame, (size[1]*2,size[0]*2))
        
    list_tiles = tiling_obj.slice_img_input(frame)
    
    for g, tile in enumerate(list_tiles):
        gpu_holder[g].tile = tile
    
    processes = []
    for g in range(len(gpu_def)):
        p = Thread(target=thread_jobs[g], args=())
        p.daemon = True
        p.start()
        processes.append(p)
        
    # wait till all thread workers ready
    for g in range(len(gpu_def)):
        processes[g].join() 
        
    # recombine tiles
    result_tiles = []
    for g in range(len(gpu_def)):
        result_tiles.append(gpu_holder[g].result.to('cuda:0'))
    
    output = tiling_obj.recombine_tiles(result_tiles)
    output = prep_output(output)
    print(output.shape)
    if superres_factor > 1:
        output = superres.allonCUDA(output)
    print(output.shape)
    output = u_torch.torch_resize(255*output, shape_output, mode='bilinear')
    
    return output.byte()

ctrl_obj.gl_man.initGL(shape_output, shader_func=render_func)
ctrl_obj.resolution_set_counter = 0
ctrl_obj.gl_man.start()




