"""
+ Author: Haoxiang Huang
+ Date: 01-Mar-2023
+ This is the main entry script for testing
"""

import sys, os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import torch
from spikingjelly.activation_based import functional, surrogate, neuron
import numpy as np
from Utils.test_utils import Test_SHD, Custom_Args
import model
import datetime
import pickle

_seed_ = 202208
torch.manual_seed(_seed_)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def main():
    model_net = 'FC'
    fire_opt = True
    snr_db = 10.0
    attention = 'STSC'
    channel = 'multipath'
    taps = 8
    quant = 'None'
    alpha = 0.0
    fire_rate = 0.0
    batch_size = 64
    # model_path = f'./test_logs/SHD_checkpoint_max_15_{taps}{channel}_{snr_db}_STSC_{model_net}_{alpha}_{fire_rate}.pth'
    model_path = f'./test_logs/SHD_checkpoint_max_15_{taps}{channel}_{snr_db}_STSC_{model_net}_{alpha}.pth'
    
    
    args = Custom_Args( net = model_net,
                        resume = model_path, 
                        channel = channel,
                        attention = attention,
                        snr_db = snr_db, 
                        taps = taps,
                        fire_opt = fire_opt,
                        batch_size = batch_size,
                        alpha = alpha,
                        quant = quant)

    if (args.taps<=0):
        raise ValueError("Channel Taps must large than 0")
    
    net = model.SHD_JSCC(args)
    
    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)
    
    print(net)
    net.to(args.device)

    _, fr , test_acc, test_acc_cum =  Test_SHD(args).SHD_test()
    
    filename = f'./test_logs/SHD_cum_{taps}{channel}_{snr_db}_STSC_{model_net}_{alpha}_{fire_rate}.pickle'
    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump(test_acc_cum, f)
    print(f'test_acc ={test_acc: .4f}, fire_rate ={fr: .4f} ')
    print(f'escape time = {(datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()

