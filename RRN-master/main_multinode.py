##
# @mainpage Evaluation of different methods for live video super resolution
# @section description_main Description
# This is the practical implementation of the bachelorthesis "Evaluation of different methods for live video super resolution " written by Maximilian Leibiger

##
# @file main_multinode.py
# @brief This is the multinode implementation of the training script

# Imports
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
from torch.autograd import Variable
from data import get_training_set 
import torch.optim as optim
from utils import Logger 
import torch.nn as nn
from arch import RRN
import numpy as np
import datetime
import argparse
import random
import pickle
import torch
import glob
import sys
import os

# Global Variables
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
## super resolution upscale factor
scale = 4   
## Training batch size    
batchsize = 4 
## Starting epoch for continuing training  
start_epoch = 1 
## Number of epochs to train for
nEpochs = 70   
## Learning Rate. Default=0.01 
lr = 0.0001   
## Number of threads for the data loader to use  
threads = 24  
## Random seed to use  
seed = 0        
## 0 to use original frame size
patch_size = 64 
## If set data gets augmented when loaded
data_augmentation = True 
## Count of network layers
layer = 5    
## Stepsize
stepsize = 60   
## Learning rate decay
gamma = 0.1     
## Folder for saving logs
save_train_log = './result/log/'
## Weight decay (default: 5e-04)
weight_decay = 5e-04 
## Name of logs
log_name = 'rrn-10'
## Folder of training data
data_dir = './RRN-master/trainImgs'   
## List of collected loss
loss_dir = []

def main():
    """! Main body of the method. This function gets called when the training is initialized"""
    try:
        torch.manual_seed(seed)
        ############# - new for multinode - #############
        ## Parser for automatic distribution of training
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
        parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
        argv = parser.parse_args()
        local_rank = argv.local_rank
        resume = argv.resume

        torch.distributed.init_process_group(backend="nccl")
        ## get the device thats locally asigned
        device = torch.device("cuda:{}".format(local_rank))

        ########### - end new for multinode - ###########

        sys.stdout = Logger(os.path.join(save_train_log, 'train_' + log_name + '.txt'))

        ## Selecting network layer
        n_c = 128
        n_b = layer
        rrn = RRN(scale, n_c, n_b) # initial filter generate network 
        scaler = torch.cuda.amp.GradScaler()
        ############# - new for multinode - #############
        rrn = rrn.to(device)
        rrn = torch.nn.parallel.DistributedDataParallel(rrn, device_ids=[local_rank], output_device=local_rank)

        # We only save the model who uses device "cuda:0"
        # To resume, the device for the saved model would also be "cuda:0"
        if resume == True:
            map_location = {"cuda:0": "cuda:{}".format(local_rank)}
            save_model_path = os.path.join(os.path.join('/lustre/nec/ws3/ws/xwwjoste-styleflow/git/SuperResolution-RRN/result/weight', systime), "*")
            models = glob.glob(save_model_path)
            latest_model = max(models, key=os.path.getctime)
            rrn.load_state_dict(torch.load(latest_model, map_location=map_location))
        print('===> Initialize DataLoader')
        train_set = get_training_set(data_dir, scale, data_augmentation)
        train_sampler = DistributedSampler(dataset=train_set)
        train_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchsize, drop_last=True, sampler=train_sampler)
        print('===> DataLoader initialized')
        ########### - end new for multinode - ###########

        p = sum(p.numel() for p in rrn.parameters())*4/1048576.0
        print('Model Size: {:.2f}M'.format(p))
        print(rrn)
        print('===> {}L model has been initialized'.format(n_b))
        criterion = nn.L1Loss(reduction='sum')
        optimizer = optim.Adam(rrn.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-4, weight_decay=weight_decay) # Had to change eps from 1e-8 to 1e-4 because of half precision
        if stepsize > 0:
            scheduler = lr_scheduler.StepLR(optimizer, step_size = stepsize, gamma=gamma)
        for epoch in range(start_epoch, nEpochs+1):
            print(f"Epoch {epoch} started \n")
            train(train_loader, rrn, scale, criterion, optimizer, n_c, device, scaler) # feed data into network
            scheduler.step()
            checkpoint(rrn, epoch)
    except Exception as e:
        print(e)
        sys.exit(1)

def train(train_loader, rrn, scale, criterion, optimizer, n_c, device, scaler):
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
    for data in train_loader:
        x_input, target = data[0], data[1] # input and target are both tensor, input:[N,C,T,H,W] , target:[N,C,H,W]
        x_input = Variable(x_input).to(device)
        target = Variable(target).to(device)
        optimizer.zero_grad()
        B, _, T, _ ,_ = x_input.shape
        out = []
        init = True
        for i in range(T-1):
            if init:
                init_temp = torch.zeros_like(x_input[:,0:1,0,:,:]) # If its the first frame processed it just repeats it one time as there is no "previous" low res frame
                init_o = init_temp.repeat(1, scale*scale*3,1,1)
                init_h = init_temp.repeat(1, n_c, 1,1)
                with torch.cuda.amp.autocast(): # Autocast for mixed precision training
                    h, prediction = rrn(x_input[:,:,i:i+2,:,:], init_h, init_o, init)
                out.append(prediction)
                init = False
            else:
                with torch.cuda.amp.autocast():
                    h, prediction = rrn(x_input[:,:,i:i+2,:,:], h, prediction, init)
                out.append(prediction)

        prediction = torch.stack(out, dim=2)
        with torch.cuda.amp.autocast():
            loss = criterion(prediction, target)/T
        loss_dir.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def checkpoint(rrn, epoch): 
    """! Used to save checkpoints after each epoch

    @param rrn  Passes the Network to save the weights
    @param epoch  Passes the Epoch so the saved weights can be correctly labeled
    """
    save_model_path = os.path.join('/lustre/nec/ws3/ws/xwwjoste-styleflow/git/SuperResolution-RRN/result/weight', systime)
    isExists = os.path.exists(save_model_path)
    if not isExists:
        os.makedirs(save_model_path)
    model_name  = 'X'+str(scale)+'_{}L'.format(layer)+'_{}'.format(patch_size)+'_epoch_{}.pth'.format(epoch)
    torch.save(rrn.state_dict(), os.path.join(save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))
    with open("/lustre/nec/ws3/ws/xwwjoste-styleflow/git/SuperResolution-RRN/result/log/loss_logs.pickle", "wb") as handle:
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

