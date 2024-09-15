
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import platform

class SHD(Dataset):
    def __init__(self, train:bool, dt:int, T :int):
        super(SHD, self).__init__()
        
        # dt = 60ms and T = 15
        assert dt == 60, 'only SHD with dt=60ms is supported'
        self.train = train
        self.dt = dt
        self.T = T
        os_name = platform.system()

        # Local
        if os_name == "Windows":
            pre_path = 'D:/Study/Imperial/Project/SNN_Study/STSC-SNN'
        # Remote
        elif os_name == "Linux":
            pre_path = '/root/Project/SNN_Study/STSC-SNN'

        if train:
           
            X = np.load(pre_path + '/datasets/SHD/trainX_60ms.npy')[:,:T,:]
            y = np.load(pre_path + '/datasets/SHD/trainY_60ms.npy')
        else:
            X = np.load(pre_path + '/datasets/SHD/testX_60ms.npy')[:,:T,:]
            y = np.load(pre_path + '/datasets/SHD/testY_60ms.npy')

        self.len = 8156
        if train == False:
            self.len = 2264
        self.eventflow = X
        self.label = y
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.eventflow[idx, ...].astype(np.float32)    
        y = self.label[idx].astype(np.float32)                
        return (x, y)
    
def load_data(args):
    if args.dataset == "SHD":
        train_ds = SHD(train=True, dt=args.dt, T=args.T)
        test_ds = SHD(train=False, dt=args.dt, T=args.T)
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=True)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size, pin_memory=True)

    elif args.dataset == "DVSG":
        train_ds = DVS128Gesture(root='..\Spikingjelly\DVS_Gesture\dataset\DVS128Gesture', train=True, data_type='frame', frames_number=args.T, split_by='number')
        test_ds = DVS128Gesture(root='..\Spikingjelly\DVS_Gesture\dataset\DVS128Gesture', train=False, data_type='frame', frames_number=args.T, split_by='number')

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=True)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size, pin_memory=True)
    
    elif args.dataset == "NMNIST":
        train_ds = NMNIST(root='./datasets/NMNIST/',train=True, data_type='frame',frames_number = args.T, split_by='number')
        test_ds = NMNIST(root='./datasets/NMNIST/',train=False, data_type='frame',frames_number = args.T, split_by='number')
        
        # Define the proportion of the data to retain
        train_subset_fraction = 0.05
        test_subset_fraction = 0.05 

        
        train_subset_size = int(len(train_ds) * train_subset_fraction)
        test_subset_size = int(len(test_ds) * test_subset_fraction)

        # Generate random indices
        train_indices = random.sample(range(len(train_ds)), train_subset_size)
        test_indices = random.sample(range(len(test_ds)), test_subset_size)

        train_subset = Subset(train_ds, train_indices)
        test_subset = Subset(test_ds, test_indices)

        # using the subsets
        train_dl = DataLoader(train_subset, shuffle=True, batch_size=16, pin_memory=True)
        test_dl = DataLoader(test_subset, shuffle=False, batch_size=16, pin_memory=True)
    
    return train_dl, test_dl

import torch

# Int tensor
def tensor_to_binary(float_tensor, auto_bits=True, specified_bits=32, out_type = 'string'):
    int_tensor = float_tensor.to(torch.int32)
    
    if auto_bits: #based on max value to determine bit len
        max_value = torch.max(torch.abs(int_tensor)).item()
        bit_len = max_value.bit_length() if max_value > 0 else 1
    else:
        bit_len = specified_bits

    binary_tensor = torch.zeros((int_tensor.numel(), bit_len), dtype=torch.uint8)
    
    for i in range(int_tensor.numel()):
        # Convert each integer to a binary string, padded to the correct length
        binary_str = format(int_tensor[i].item(), '0' + str(bit_len) + 'b')
        # Convert the binary string into a tensor of uint8
        binary_tensor[i] = torch.tensor([int(bit) for bit in binary_str], dtype=torch.uint8)
    
    if out_type == 'string':
        binary_string = ''.join([''.join([str(bit.item()) for bit in row]) for row in binary_tensor])
        return binary_string, bit_len
    else:
        return binary_tensor, bit_len

# To Int tensor
def binary_to_tensor(encoded_string, bit_len, dtype=torch.float32):
        num_elements = len(encoded_string) // bit_len
        decoded_integers = [encoded_string[i*bit_len:(i+1)*bit_len] for i in range(num_elements)]
        
        decoded_values = [int(binary_str, 2) for binary_str in decoded_integers]
        
        decoded_tensor = torch.tensor(decoded_values, dtype=dtype)
        
        return decoded_tensor
import torch
import numpy as np

def float32_to_binary16(input_tensor):
    sign_bit = input_tensor.lt(0).float()

    abs_input_tensor = input_tensor.abs()
    exponent_tensor = abs_input_tensor.log2().floor().clamp(-14, 15) + 15
    mantissa_tensor = (abs_input_tensor / (2 ** (exponent_tensor - 15)) - 1) * 1024

    exponent_bits = exponent_tensor.to(torch.int32) & 0x1F  
    mantissa_bits = mantissa_tensor.to(torch.int32) & 0x3FF

    combined = (sign_bit.to(torch.int32) << 15) | (exponent_bits << 10) | mantissa_bits

    output_tensor = torch.zeros((*input_tensor.shape, 16), dtype=torch.float32, device=input_tensor.device)
    for i in range(16):
        output_tensor[..., i] = (combined >> (15 - i)) & 1

    output_tensor = output_tensor.reshape(*input_tensor.shape[:-1], -1)
    return output_tensor

def binary16_to_float32(input_tensor):
    
     # Calculate the original last dimension size (N*16 to N)
    N = input_tensor.shape[-1] // 16
    
    # Reshape the binary tensor to isolate each 16-bit binary number
    input_tensor = input_tensor.view(*input_tensor.shape[:-1], N, 16)
    # print(input_tensor.shape)
    
    sign_bit = input_tensor[..., 0].unsqueeze(-1)
    exponent_bits = input_tensor[..., 1:6]
    mantissa_bits = input_tensor[..., 6:]
    
    sign = (-1) ** sign_bit
    
    exponent_weights = torch.tensor([2**i for i in range(4, -1, -1)], dtype=torch.float32).to(input_tensor.device)
    exponent = torch.sum(exponent_bits * exponent_weights, dim=-1) - 15
    
    mantissa_weights = torch.tensor([2**(-i) for i in range(1, 11)], dtype=torch.float32).to(input_tensor.device)
    mantissa = 1 + torch.sum(mantissa_bits * mantissa_weights, dim=-1)
    
    value = sign.squeeze(-1) * (2 ** exponent) * mantissa
    
    return value.view(*input_tensor.shape[:-1])