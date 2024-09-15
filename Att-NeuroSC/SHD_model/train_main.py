"""
+ Author: Haoxiang Huang
+ Date: 20-Jan-2023
+ This is the main entry script for training
+ Run with STSC module: 
+ python ./train_main.py -dataset SHD -T 15 -dt 60 -device cuda:0 -batch_size 64 -epochs 100 -opt adam -lr 0.0001 -loss MSE -net SNN -attention STSC -channel multipath -alpha 0
"""

import sys, os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import torch
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from torch.utils.data import DataLoader
import time, argparse, datetime
import numpy as np
from Utils.option_utils import option
from Utils.data_utils import load_data
from Utils.train_utils import Train_SHD
import model

_seed_ = 202208
torch.manual_seed(_seed_)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def main():
    parser = option().initialize()
    args = parser.parse_args()
   

    if (args.taps<=0):
        raise ValueError("Channel Taps must large than 0")

    if args.channel == 'multipath' or 'multipathreal':
        suffix = f'{args.taps}{args.channel}_{args.snr_db}_{args.attention}_{args.net}_{args.alpha}'
    else:
        args.taps = 1
        suffix = f'{args.channel}_{args.snr_db}_{args.attention}_{args.net}_{args.alpha}'
    
    print(args)
    mode = 'a' if args.resume else 'w'

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir + '/txt')
        print(f'Mkdir {args.out_dir}.')

    with open(os.path.join('./logs/txt', f'{args.dataset}_logs_{args.T}_{suffix}.txt'), mode, encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))
        args_txt.write('\n\n')
    
    net = model.SHD_JSCC(args)
    
    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)
    
    print(net)
    net.to(args.device)

    train_data_loader, test_data_loader = load_data(args)  

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print("Loading Successfully!")
        print(f'start_epoch = {start_epoch}')

    Train_SHD(net, optimizer, lr_scheduler, args, train_data_loader, test_data_loader,start_epoch,scaler,max_test_acc)

if __name__ == '__main__':
    main()

