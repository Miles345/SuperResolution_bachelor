##
# @file main.py
# @brief Original version of the main training script. Currently obsolete as multinode training is implemented. Can be used when a fallback is needed for single node training

## Imports
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import get_training_set 
import torch.optim as optim
from utils import Logger 
import torch.nn as nn
from arch import RRN
import numpy as np
import datetime
import random
import pickle
import torch
import time
import sys
import os

# Global Variables
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
## super resolution upscale factor
scale = 2    
## Training batch size  
batchsize = 4  
## Starting epoch for continuing training 
start_epoch = 1 
## Number of epochs to train for
nEpochs = 70   
## Learning Rate. Default=0.01 
lr = 0.0001
## Number of threads for the data loader to use     
threads = 12   
## Random seed to use
seed = 0    
## Record of all imagenames in dataset     
file_list = 'allImages.txt' 
## Directory of the training data
data_dir = './RRN-master/trainImages'
## 0 to use original frame size
patch_size = 64 
## If set training data gets modified while loading
data_augmentation = True 
## Count of network layers
layer = 5    
## Amount, by which the weights should be modified
stepsize = 60   
## Learning rate is decayed by a factor of 10 every half of total epochs   
gamma = 0.1     
## Location of logs
save_train_log = './result/log/'
## Weight decay (default: 5e-04)
weight_decay = 5e-04 
## Name of logs
log_name = 'rrn-10'
## gpu device ids for CUDA_VISIBLE_DEVICES
gpu_devices = '0,1,2,3,4,5,6,7' 
## Directory of the training data
data_dir = './RRN-master/trainImages'   

loss_dir = []

def main():
    """! Main function that gets called to initialize training"""
    torch.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    sys.stdout = Logger(os.path.join(save_train_log, 'train_'+ log_name + '.txt'))
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
       use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    pin_memory = True if use_gpu else False
    print('===> Loading Dataset')
    train_set = get_training_set(data_dir, scale, data_augmentation, file_list) 
    train_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchsize, shuffle=True, pin_memory=pin_memory, drop_last=True)
    print('===> DataLoading Finished')
    # Selecting network layer
    n_c = 128
    n_b = 5
    rrn = RRN(scale, n_c, n_b) # initial filter generate network 
    rrn = rrn.half()
    p = sum(p.numel() for p in rrn.parameters())*4/1048576.0
    print('Model Size: {:.2f}M'.format(p))
    print(rrn)
    print('===> {}L model has been initialized'.format(n_b))
    rrn = torch.nn.DataParallel(rrn)
    criterion = nn.L1Loss(reduction='sum')
    if use_gpu:
        rrn = rrn.cuda()
        criterion = criterion.cuda().half()
    optimizer = optim.Adam(rrn.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-4, weight_decay=weight_decay) # Had to change eps from 1e-8 to 1e-4 because of half precision
    if stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size = stepsize, gamma=gamma)
    for epoch in range(start_epoch, nEpochs+1):
        print(f"Epoch {epoch} started \n")
        train(train_loader, rrn, scale, criterion, optimizer, epoch, use_gpu, n_c) #feed data into network
        scheduler.step()
        checkpoint(rrn, epoch)

def train(train_loader, rrn, scale, criterion, optimizer, epoch, use_gpu, n_c):
    """! Main function that gets called for each training epoch

    @param train_loader  Passes the defined DataLoader into the training function
    @param rrn  Passes the network into the training function
    @param scale  passes the super resolution scale into the training function
    @param criterion  Passes the L1 loss function into the training function
    @param optimizer  Passes the Adam optimizer into the training function
    @param n_c  Passes the neuron count of each layer
    @param device  Passes the device that the training will occur on
    @param scaler  Passes the grad scaler so backpropagation will use the correct values
    """
    rrn.train()
    for iteration, data in enumerate(train_loader):
        x_input, target = data[0], data[1] # input and target are both tensor, input:[N,C,T,H,W] , target:[N,C,H,W]
        if use_gpu:
            x_input = Variable(x_input).cuda()
            target = Variable(target).cuda()
        t0 = time.time()
        optimizer.zero_grad()
        B, _, T, _ ,_ = x_input.shape
        out = []
        init = True
        for i in range(T-1):
            if init:
                init_temp = torch.zeros_like(x_input[:,0:1,0,:,:]).half()
                init_o = init_temp.repeat(1, scale*scale*3,1,1)
                init_h = init_temp.repeat(1, n_c, 1,1)
                h, prediction = rrn(x_input[:,:,i:i+2,:,:].half(), init_h, init_o, init)
                prediction = prediction.float()
                out.append(prediction)
                init = False
            else:
                h, prediction = rrn(x_input[:,:,i:i+2,:,:].half(), h, prediction.half(), init)
                prediction = prediction.half()
                out.append(prediction)

        prediction = torch.stack(out, dim=2)
        loss = criterion(prediction, target)/(B*T)
        loss_dir.append(loss.item())
        loss.backward()
        optimizer.step()
        t1 = time.time()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(train_loader), loss.item(), (t1 - t0)))

def checkpoint(rrn, epoch): 
    """! Used to save checkpoints after each epoch

    @param rrn  Passes the Network to save the weights
    @param epoch  Passes the Epoch so the saved weights can be correctly labeled
    """
    save_model_path = os.path.join('./result/weight', systime)
    isExists = os.path.exists(save_model_path)
    if not isExists:
        os.makedirs(save_model_path)
    model_name  = 'X'+str(scale)+'_{}L'.format(layer)+'_{}'.format(patch_size)+'_epoch_{}.pth'.format(epoch)
    torch.save(rrn.state_dict(), os.path.join(save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))
    with open("./log/loss_logs.pickle", "wb") as handle:
        pickle.dump(loss_dir, handle)
    
def set_random_seed(seed):
    """! Sets all random seeds to the global one to provide constant results
    @param seed  Passes the seed that will be set on all normally random variables
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    main()    
