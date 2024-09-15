import argparse

class option():
     def __init__(self):
         pass
     def initialize(self):
        # with STSC module: 
        # python ./train_main.py -dataset SHD -T 15 -dt 60 -device cuda:0 -batch_size 64 -epochs 100 -opt adam -lr 0.0001 -loss MSE -net SNN -attention STSC -channel multipath -alpha 2e-7

        parser = argparse.ArgumentParser(description='Classify SHD')
        parser.add_argument("-dataset",type=str,default="SHD")
        parser.add_argument("-batch_size",type=int,default=256) 
        parser.add_argument("-T",type=int,default=15,help='simulating time-steps') 
        parser.add_argument("-dt",type=int,default=60,help='frame time-span') 
        parser.add_argument('-device', default='cuda:0', help='device')
        parser.add_argument('-epochs', default=200, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('-amp', default=True, type=bool, help='automatic mixed precision training')
        parser.add_argument('-cupy', default=True, type=bool, help='use cupy backend')
        parser.add_argument('-opt', default="adam", type=str, help='use which optimizer. SDG or Adam')
        parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
        parser.add_argument('-lr', default=0.0001, type=float, help='learning rate')
        parser.add_argument('-loss', default="MSE", type=str, help='loss function')
        parser.add_argument('-resume', type=str, default=None, help='resume from the checkpoint path')
        parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
        parser.add_argument('-net', type=str, default='SNN', help='use which network(SNN or ANN)')
        parser.add_argument('-attention', type=str, default='None', help='use which attention')
        parser.add_argument('-snr_db', default=10, type=float, help='Training Channel SNR (dB)')
        parser.add_argument('-channel', type=str, default='noiseless', help='use which channel')
        parser.add_argument('-taps', type=int, default=8, help='Channel taps (only for multipath channel)')
        parser.add_argument('-alpha', type=float, default= 2e-7, help='for fire_opt loss function')
        parser.add_argument('-quant', type=str, default='None', help='use which quantization')
        return parser