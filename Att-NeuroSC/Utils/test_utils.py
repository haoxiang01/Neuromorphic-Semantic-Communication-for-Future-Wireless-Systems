import sys, os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import torch
from spikingjelly.activation_based import functional, neuron
import datetime
import os
import time
from torch.cuda import amp
import torch.nn.functional as F
import numpy as np
from SHD_model import model
_seed_ = 202208
torch.manual_seed(_seed_)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from Utils import data_utils
np.random.seed(_seed_)
import matplotlib.pyplot as plt


class Test_SHD():
    def __init__(self, args):
        self.args = args
        _, test_data_loader = data_utils.load_data(args)
        self.test_data_loader = test_data_loader
        checkpoint = torch.load(self.args.resume, map_location='cpu')
        self.net = model.SHD_JSCC(self.args)
        functional.set_step_mode(self.net, 'm')
        self.net.to(self.args.device)
        
        self.net.load_state_dict(checkpoint['net'])
        # Train_epoch = checkpoint['epoch']
        print("Loading Successfully!")
        # print(f'Train_epoch = {Train_epoch}')

    def bandwidth_clip(self, tx_data, bits_per_symbol=16):
        [T,B,N] = tx_data.shape
        BW_bps = N
        BW_sps= BW_bps//bits_per_symbol

        frames = torch.zeros(T, B, 16, N)
        for l in range(bits_per_symbol - 1):
            end_idx = (l + 1) * BW_sps
            
            frames[:, :, l, :end_idx] = tx_data[:, :, :end_idx]

        frames[:, :, l+1, :] = tx_data[:, :, :]
        
        # frames = frames.reshape(-1,1,B,N)
        
        return frames.to(self.args.device)
    
    def plot_line(self, data):
        
        # x = range(len(data))
        x = [i for i in range(len(data))]

        plt.plot(x, data, linestyle='-', color='b', linewidth=2)

        plt.title('Test Accuracy Over Time')
        plt.xlabel('Time')
        plt.ylabel('Test Accuracy')

        # plt.legend(['Test Accuracy'])

        plt.grid(True)
        # plt.ion()
        plt.show()



    def SHD_test(self):
        self.net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        fire_rate = 0
        L=16 #bits per symobl
        if self.args.net == 'SNN':
            test_acc_cum = torch.zeros(self.args.T+1, device = self.args.device)
        else:
            test_acc_cum = torch.zeros(self.args.T*L+1, device = self.args.device)
        with torch.no_grad():
            for frame, label in self.test_data_loader:
                frame = frame.to(self.args.device)
                frame = frame.transpose(0, 1)   # [B, T, N] -> [T, B, N]
                label = label.to(self.args.device)
                label_onehot = F.one_hot(label.to(torch.int64), 20).float()
                out_fr = None
                
                if self.args.loss == "MSE":
                    if self.args.net == 'SNN':
                        out_fr, fire_rate= self.net(frame)
                        out_fr_mean = out_fr.mean(0)
                        loss = F.mse_loss(out_fr_mean, label_onehot)
                        for t in range (1, self.args.T+1):
                            out_fr_cum = out_fr[:t].mean(0)
                            test_acc_cum[t] +=  (out_fr_cum.argmax(1) == label).float().sum()
                    else:
                        # out_fr = self.net(frame)
                        # out_fr_mean = out_fr.mean(0)
                        # loss = F.mse_loss(out_fr_mean, label_onehot)
                        frame_clip = self.bandwidth_clip(frame)
                        
                        i=1
                        for t in range (1, self.args.T+1):
                            for l in range (0, L):
                                out_fr = self.net(frame_clip[:t, :, 15, :])
                                out_fr_mean = out_fr.mean(0)
                                loss = F.mse_loss(out_fr_mean, label_onehot)
                                test_acc_cum[i] +=  (out_fr_mean.argmax(1) == label).float().sum()
                                # test_acc_cum[t*L:(t+1)*L] +=  (out_fr_mean.argmax(1) == label).float().sum()
                                i = i + 1
                            # test_acc_cum[t*L:(t+1)*L] = test_acc_cum[t*L:(t+1)*L]/L
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc +=  (out_fr_mean.argmax(1) == label).float().sum().item()
                functional.reset_net(self.net)


        test_loss /= test_samples
        test_acc /= test_samples
        test_acc_cum /=test_samples

        # for i, test_acc_cum_i in enumerate(test_acc_cum):
        #     print(f'test_acc_cum[{i}]={test_acc_cum_i}')
        self.plot_line(test_acc_cum.cpu())
        
        return test_loss, fire_rate, test_acc, test_acc_cum

class Custom_Args:
    def __init__(self, net, resume, batch_size, channel, attention, snr_db, taps, alpha, fire_opt, quant):
        self.dataset = 'SHD'
        self.batch_size = batch_size
        self.T = 15
        self.dt = 60
        self.device = 'cuda:0'
        self.cupy = True
        self.loss = 'MSE'
        self.net = net
        self.resume = resume
        self.channel = channel
        self.snr_db = snr_db
        self.attention =attention
        self.taps = taps
        self.alpha = alpha
        self.fire_opt = fire_opt
        self.quant = quant

def Channel_Test_SNR(channels, attention, train_snr, snr_dbs, taps=8):
    base_path = './logs/'
    
    acc_data = {channel: [] for channel in channels}
    # Iterating through each channel and signal-to-noise ratio (SNR)
    for channel in channels:
        if channel == 'multipath':
            resume = f'{base_path}SHD_checkpoint_max_15_{taps}{channel}_{train_snr}_{attention}.pth'
        else:
            resume = f'{base_path}SHD_checkpoint_max_15_{channel}_{train_snr}_{attention}.pth'
        for snr_db in snr_dbs:
            args = Custom_Args(resume, channel, attention, snr_db, taps)
            _, test_acc =Test_SHD(args).SHD_test()
            acc_data[channel].append(test_acc)

    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'D', 'x', '*']
    for i, channel in enumerate(channels):
        # Plot the line for each channel
        plt.plot(snr_dbs, acc_data[channel], label=f'Channel {channel}', marker=markers[i % len(markers)], linewidth=2)
        # Adding data point values as text on the plot
        for x, y in zip(snr_dbs, acc_data[channel]):
            if y is not None:
                pass
                # plt.text(x, y, f'{y:.2f}', fontsize=12, ha='center', va='bottom')

    plt.xlabel('SNR dB', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.title(f'Test Accuracy vs SNR | Train SNR = {train_snr} dB', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def Attention_Test_SNR(channel, attentions, train_snr, snr_dbs, taps=8):
    base_path = './logs/'
    
    acc_data = {attention: [] for attention in attentions}
    # Iterating through each channel and signal-to-noise ratio (SNR)
    for attention in attentions:
        if channel == 'multipath':
            resume = f'{base_path}SHD_checkpoint_max_15_{taps}{channel}_{train_snr}_{attention}.pth'
        else:
            resume = f'{base_path}SHD_checkpoint_max_15_{channel}_{train_snr}_{attention}.pth'
        for snr_db in snr_dbs:
            args = Custom_Args(resume, channel, attention, snr_db, taps)
            _, test_acc =Test_SHD(args).SHD_test()
            acc_data[attention].append(test_acc)

    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'D', 'x', '*']
    for i, attention in enumerate(attentions):
        # Plot the line for each channel
        plt.plot(snr_dbs, acc_data[attention], label=f'Attention {attention} | Train SNR = {train_snr} dB', marker=markers[i % len(markers)], linewidth=2)
        # Adding data point values as text on the plot
        for x, y in zip(snr_dbs, acc_data[attention]):
            if y is not None:
                plt.text(x, y, f'{y:.2f}', fontsize=12, ha='center', va='bottom')

    plt.xlabel('SNR dB', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.title(f'Test Accuracy vs SNR ', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

            
        

        
        