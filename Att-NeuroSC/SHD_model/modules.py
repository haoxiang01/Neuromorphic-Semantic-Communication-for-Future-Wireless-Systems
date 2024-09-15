"""
+ Author: Haoxiang Huang
+ Date: 21-Jan-2023
"""

import torch
import torch.nn as nn 
from spikingjelly.activation_based import base, layer, surrogate, neuron
import torch.nn.functional as F
import math
from Utils.data_utils import float32_to_binary16, binary16_to_float32
from torch.autograd import Function

class STSC_Attention(nn.Module, base.StepModule):
    def __init__(self, n_channel: int, dimension: int = 4, time_rf: int = 4, reduction:int=2):

        super().__init__()
        self.step_mode = 'm'    # used in activation_based SpikingJelly
        assert dimension == 4 or dimension == 2, 'dimension must be 4 or 2'
 
        self.dimension = dimension

        if self.dimension == 4:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.time_padding = (time_rf-1) // 2
        self.n_channels = n_channel
        r_channel = n_channel//reduction  
        self.recv_T = nn.Conv1d(n_channel, r_channel, kernel_size=time_rf, padding=self.time_padding, groups=1,bias=True)
        self.recv_C = nn.Sequential(
            nn.ReLU(),
            nn.Linear(r_channel, n_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f'expected 3D or 5D input with shape [T, B, N] or [T, B, C, H, W], but got input with shape {x_seq.shape}')
        x_seq_C = x_seq.transpose(0, 1) # x_seq_C.shape = [B, T, N] or [B, T, C, H, W]
        x_seq_T = x_seq_C.transpose(1, 2) # x_seq_T.shape = [B, C, N] or [B, C, T, H, W]

        if self.dimension == 2:
            recv_h_T = self.recv_T(x_seq_T)
            recv_h_C = self.recv_C(recv_h_T.transpose(1, 2))
            D_ = 1 - self.sigmoid(recv_h_C)
            D = D_.transpose(0, 1)
            
        elif self.dimension == 4:
            avgout_C = self.avg_pool(x_seq_C).view([x_seq_C.shape[0], x_seq_C.shape[1], x_seq_C.shape[2]]) # avgout_C.shape = [N, T, C]
            avgout_T = avgout_C.transpose(1, 2)
            recv_h_T = self.recv_T(avgout_T)
            recv_h_C = self.recv_C(recv_h_T.transpose(1, 2))
            D_ = 1 - self.sigmoid(recv_h_C)
            D = D_.transpose(0, 1)
            
        return D


class STSC_Attention_LIF(nn.Module, base.StepModule):
    def __init__(self, n_channel: int, dimension: int = 4, time_rf: int = 4, reduction:int=2):

        super().__init__()
        self.step_mode = 'm'    # used in activation_based SpikingJelly
        assert dimension == 4 or dimension == 2, 'dimension must be 4 or 2'
 
        self.dimension = dimension

        if self.dimension == 4:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.time_padding = (time_rf-1) // 2
        self.n_channels = n_channel
        r_channel = n_channel//reduction  
        self.recv_T = nn.Conv1d(n_channel, r_channel, kernel_size=time_rf, padding=self.time_padding, groups=1,bias=True)
        self.recv_C = nn.Sequential(
            # nn.ReLU(),
            # neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Linear(r_channel, n_channel, bias=False),
            # neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f'expected 3D or 5D input with shape [T, B, N] or [T, B, C, H, W], but got input with shape {x_seq.shape}')
        x_seq_C = x_seq.transpose(0, 1) # x_seq_C.shape = [B, T, N] or [B, T, C, H, W]
        x_seq_T = x_seq_C.transpose(1, 2) # x_seq_T.shape = [B, C, N] or [B, C, T, H, W]

        if self.dimension == 2:
            recv_h_T = self.recv_T(x_seq_T)
            recv_h_C = self.recv_C(recv_h_T.transpose(1, 2))
            # D_ = 1 - self.sigmoid(recv_h_C)
            D_ = 1 - recv_h_C
            D = D_.transpose(0, 1)
            
        elif self.dimension == 4:
            avgout_C = self.avg_pool(x_seq_C).view([x_seq_C.shape[0], x_seq_C.shape[1], x_seq_C.shape[2]]) # avgout_C.shape = [N, T, C]
            avgout_T = avgout_C.transpose(1, 2)
            recv_h_T = self.recv_T(avgout_T)
            recv_h_C = self.recv_C(recv_h_T.transpose(1, 2))
            D_ = 1 - self.sigmoid(recv_h_C)
            D = D_.transpose(0, 1)
            
        return D


class STSC_Temporal_Conv(nn.Module, base.StepModule):
    def __init__(self, channels: int, dimension: int = 4, time_rf:int=2):

        super().__init__()
        self.step_mode = 'm'    # used in activation_based SpikingJelly
        assert dimension == 4 or dimension == 2, 'dimension must be 4 or 2'
        self.dimension = dimension

        time_padding = (time_rf-1)//2
        self.time_padding = time_padding

        if dimension == 4:
            kernel_size = (time_rf, 1, 1)
            padding = (time_padding,0,0)
            self.conv = nn.Conv3d(channels,channels,kernel_size=kernel_size,padding=padding,groups=channels,bias=False)
        else:
            kernel_size = time_rf
            self.conv = nn.Conv1d(channels,channels,kernel_size=kernel_size,padding=time_padding,groups=channels,bias=False)
        

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f'expected 3D or 5D input with shape [T, B, N] or [T, B, C, H, W], but got input with shape {x_seq.shape}')
        
        # x_seq.shape = [T, B, N] or [T, B, C, H, W]

        x_seq = x_seq.transpose(0,1) # x_seq.shape = [B, T, N] or [B, T, C, H, W]
        x_seq = x_seq.transpose(1,2) # x_seq.shape = [B, N, T] or [B, C, T, H, W]
        x_seq = self.conv(x_seq)
        x_seq = x_seq.transpose(1,2) # x_seq.shape = [B, T, N] or [B, T, C, H, W]
        x_seq = x_seq.transpose(0,1) # x_seq.shape = [T, B, N] or [T, B, C, H, W]
        
        return x_seq


class STSC(nn.Module, base.StepModule):
    def __init__(self, in_channel: int, dimension: int = 4, time_rf_conv: int=3, time_rf_at: int=3, use_gate=True, use_filter=True, reduction:int=1, attention='SE'):

        super().__init__()
        self.step_mode = 'm'    # used in activation_based SpikingJelly

        assert dimension == 4 or dimension == 2, 'dimension must be 4 or 2'
        self.dimension = dimension

        self.time_rf_conv = time_rf_conv
        self.time_rf_at = time_rf_at
        self.att = attention

        if use_filter:
            self.temporal_conv = STSC_Temporal_Conv(in_channel,time_rf=time_rf_conv, dimension=dimension)
        
        if use_gate:
            if attention == 'SE':
                self.spatio_temporal_attention = STSC_Attention(in_channel, time_rf=time_rf_at, reduction=reduction, dimension=dimension)
            elif attention == 'LIF':
                self.spatio_temporal_attention = STSC_Attention_LIF(in_channel, time_rf=time_rf_at, reduction=reduction, dimension=dimension)
                self.LIF = neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.use_gate = use_gate
        self.use_filter = use_filter
          
    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f'expected 3D or 5D input with shape [T, B, N] or [T, B, C, H, W], but got input with shape {x_seq.shape}')

        if self.use_filter:
            # Filitering
            x_seq_conv   = self.temporal_conv(x_seq)
        else:
            # without filtering
            x_seq_conv = x_seq

        if self.dimension == 2:
            if self.use_gate:
                # Gating
                x_seq_D = self.spatio_temporal_attention(x_seq)     
                y_seq = x_seq_conv * x_seq_D
            else:
                # without gating
                y_seq = x_seq_conv          
        else:
            if self.use_gate:
                # Gating
                x_seq_D = self.spatio_temporal_attention(x_seq)     
                y_seq = x_seq_conv * x_seq_D[:, :, :, None, None]   # broadcast
            else:
                # without gating
                y_seq = x_seq_conv 
        if self.att == 'LIF':
            y_seq = self.LIF(y_seq)
        
        return y_seq


class SHD_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        time_rf_conv = 5
        time_rf_at = 3
        self.args = args
        self.rnn_layers = 2
        self.rnn_dropout = 0.2

        layers = []
        # use STSC attention module
        if self.args.attention == 'STSC':
            if self.args.net == 'SNN':
                layers.append(STSC(700,dimension=2,time_rf_conv=time_rf_conv, time_rf_at=time_rf_at, use_gate=True, use_filter=True, attention='LIF'))
            else:
                layers.append(STSC(700,dimension=2,time_rf_conv=time_rf_conv, time_rf_at=time_rf_at, use_gate=True, use_filter=True))
        
        if self.args.net == 'FC':
            fc_layers = []
            fc_layers.extend([
                nn.Linear(700,128),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.ReLU(),
            ])
            self.stsc = nn.Sequential(*layers)
            self.fc = nn.Sequential(*fc_layers)


        elif self.args.net == 'LSTM':
             self.fc = nn.Sequential(*layers)
             self.rnn = nn.LSTM(input_size=700, hidden_size=128, num_layers = self.rnn_layers, dropout = self.rnn_dropout)
        
        elif self.args.net == 'RNN':
             self.fc = nn.Sequential(*layers)
             self.rnn = nn.RNN(input_size=700, hidden_size=128, num_layers = self.rnn_layers, dropout = self.rnn_dropout)

        # SNN 
        elif self.args.net == 'SNN':
            layers.extend([
                layer.Linear(700,128),
                neuron.LIFNode(tau=10.0, decay_input=False, v_threshold=0.3, surrogate_function=surrogate.ATan(), detach_reset=True),
                layer.Linear(128,128),
                neuron.LIFNode(tau=10.0, decay_input=False, v_threshold=0.3, surrogate_function=surrogate.ATan(), detach_reset=True),
            ])
            self.fc = nn.Sequential(*layers)
        else:
             raise ValueError("Not implemented exception: The provided model type '{}' is not supported".format(self.args.net))
        
    def forward(self, x: torch.Tensor):
        if self.args.net == 'LSTM' or self.args.net == 'RNN':
            out_stsc= self.fc(x) #[T,B,N]
            # Set initial hidden and cell states 
            h0 = torch.zeros(self.rnn_layers, out_stsc.size(1), 128).to(self.args.device)
            if self.args.net == 'LSTM':
                c0 = torch.zeros(self.rnn_layers, out_stsc.size(1), 128).to(self.args.device)
                state0 =  (h0, c0)
            else:
                state0 = h0
            Y, _  = self.rnn(out_stsc, state0)
            return Y
        elif self.args.net == 'FC':
            out_stsc = self.stsc(x)
            # print(out_stsc.shape)
            # out_stsc = out_stsc.mean(0,keepdim=True)
            # print(out_stsc.shape)
            return self.fc(out_stsc)
        else:
            return self.fc(x)
        
class SHD_Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        time_rf_conv = 5
        time_rf_at = 3
        self.rnn_layers = 1

        layers = []
        # Real flat channnel
        if self.args.channel == 'noiseless' or self.args.channel == 'awgn':
           in_channel = 128
        # Multipath complex channnel
        elif self.args.channel == 'multipath':
           in_channel = (128 + self.args.taps-1) *2
        # Multipath real channnel
        elif self.args.channel == 'multipathreal':
           in_channel = 128 + self.args.taps - 1
        # Complex flat channel
        else:
           in_channel = 2 * 128
        
        # use STSC attention module
        if self.args.attention == 'STSC' and self.args.channel != 'noiseless':
            if self.args.net == 'SNN':
                layers.append(STSC(in_channel,dimension=2,time_rf_conv=time_rf_conv, time_rf_at=time_rf_at, use_gate=True, use_filter=True, attention='LIF'))
            else:
                layers.append(STSC(in_channel,dimension=2,time_rf_conv=time_rf_conv, time_rf_at=time_rf_at, use_gate=True, use_filter=True))
        
        if self.args.net == 'FC':
            layers.extend([
                nn.Linear(in_channel, 128),
                nn.ReLU(),
                nn.Linear(128,100),
                layer.VotingLayer(5)
            ])
            self.fc = nn.Sequential(*layers)
        elif self.args.net == 'LSTM' or self.args.net == 'RNN':
             self.fc = nn.Sequential(*layers)
             if self.args.net == 'LSTM':
                self.rnn = nn.LSTM(input_size=in_channel, hidden_size=128, num_layers = self.rnn_layers)
             else:
                self.rnn = nn.RNN(input_size=in_channel, hidden_size=128, num_layers = self.rnn_layers)
             dense = []
             dense.extend([
                nn.Linear(128,100),
                layer.VotingLayer(5)
            ])
             self.dense =  nn.Sequential(*dense)
        else: #SNN
            layers.extend([
                layer.Linear(in_channel, 128),
                neuron.LIFNode(tau=10.0, decay_input=False, v_threshold=0.3, surrogate_function=surrogate.ATan(), detach_reset=True),
                layer.Linear(128,100),
                neuron.LIFNode(tau=10.0, decay_input=False, v_threshold=0.3, surrogate_function=surrogate.ATan(), detach_reset=True),
                layer.VotingLayer(5)
            ])
            self.fc = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor):
        if self.args.net == 'LSTM' or self.args.net == 'RNN':
            out_stsc= self.fc(x)
            # Set initial hidden and cell states 
            h0 = torch.zeros(self.rnn_layers, out_stsc.size(1), 128).to(self.args.device)
            if self.args.net == 'LSTM':
                c0 = torch.zeros(self.rnn_layers, out_stsc.size(1), 128).to(self.args.device)
                state0 =  (h0, c0)
            else:
                state0 = h0
            Y, _  = self.rnn(out_stsc, state0)
            out = self.dense(Y)
            return out
        else:
            return self.fc(x)

class Digital_Channel(Function):
    @staticmethod
    def forward(ctx, x, channel):
        ctx.channel = channel
        ctx.save_for_backward(x)
        x, clip_value =  channel.Quant(x) 
        x = channel.deQuant(x, clip_value)
        # Transmitting through channel
        if(channel.channel=='multipath'):
            y = channel.Multipath_channel(x)
        elif(channel.channel=='multipathreal'):
            y = channel.MultipathReal_channel(x)
        elif(channel.channel=='noiseless'):
            y = channel.noiseless_channel(x)
        elif(channel.channel=='awgn'):
            y = channel.AWGN_channel(x)
        elif(channel.channel=='awgncomplex'):
            y = channel.AWGNComplex_channel(x)
        elif(channel.channel=='rayleigh'):
            y = channel.Rayleigh_channel(x)
        else:
            raise ValueError("Not implemented exception: The provided channel type '{}' is not supported".format(channel.channel))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# Realization of channels as a nn module
class Channel(nn.Module):
     def __init__(self, args, out_typ = 'real'):
         '''
            x: [T, B, N]
            h: [T, B, L]
         '''
         super().__init__()
         self.args = args
         self.channel = self.args.channel
         self.device = self.args.device
         self.snr_db = self.args.snr_db
         self.out_typ = out_typ

         
     def forward(self, x: torch.Tensor):
            if self.args.quant == 'half':
                x, clip_value =  self.Quant(x)
                x = self.deQuant(x, clip_value)
            # Transmitting through the channel
            if(self.channel=='multipath'):
                output = self.Multipath_channel(x)
            elif(self.channel=='multipathreal'):
                output = self.MultipathReal_channel(x)
            elif(self.channel=='noiseless'):
                output = self.noiseless_channel(x)
            elif(self.channel=='awgn'):
                output = self.AWGN_channel(x)
            elif(self.channel=='awgncomplex'):
                output = self.AWGNComplex_channel(x)
            elif(self.channel=='rayleigh'):
                output = self.Rayleigh_channel(x)
            else:
                raise ValueError("Not implemented exception: The provided channel type '{}' is not supported".format(self.channel))
            return output
     
     def Quant(self, x):
         clip_value = torch.max(x)
         if self.args.quant == 'half':
            bin = float32_to_binary16(x)
         bin = bin*2 - 1 #BPSK 0->-1, 1->1
         return bin, clip_value
      
     def deQuant(self, x, clip_value, device = 'cuda:0'):
         # BPSK decoding  
         if(torch.is_complex(x)):
             x = x.real
         x = torch.where(x > 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
         out = binary16_to_float32(x)
         out = torch.where(out < 1e-04, torch.tensor(0.0,device=device), out)
         out = torch.where(out > clip_value, clip_value.clone().detach().float(), out)
         return out
        
     def convolution(self, x, h):
        '''
        Perform batch-wise convolution using conv1d
        x: [T, B, N]
        h: [T, B, L]
        '''
        assert x.shape[0] == h.shape[0]
        x = x.to(torch.float)
        [T, B, N] = x.shape
        L = h.shape[2]
        G = T*B
    
        # Reshape x and h to fit conv1d requirements: (N, C, L)
        h = h.reshape(G, 1, L)  # [G, 1, L]
        x = x.reshape(1, G, N)  # [1, G, N]

        # Pad [1, G, （N + 2 *（L-1))]
        x_padded = F.pad(x, (L - 1, L - 1))
        
        # [1, G, （N + 2*L-2） - L + 1] = [1, G, N + L - 1]
        y = F.conv1d(x_padded, weight=h, bias=None, stride=1, dilation=1, groups=G) 

        # Ensure keep dtype torch.float32. torch.float16 will raise error in trainning
        y = y.reshape(T, B, -1).type(torch.float32)
        return y
    

     def sample_h_mp(self, sample_size, L):
        '''Generate real and imaginary part of channel response with Power Delay Profile (PDP)'''
    
        # PDP generation
        PDP = torch.ones(L)
        PDP = PDP / torch.sum(PDP)

        # Generate channel samples h= 1/2*N(0,1)
        h = 1 / torch.sqrt(torch.tensor(2.)) * torch.normal(mean=0.0, std=1, size=sample_size)
        h = h * torch.sqrt(PDP).reshape(-1, 1)
        
        return h
     
     def Multipath_channel(self, x):
         '''SISO multipath complex channel'''
         [T, B, N] = x.shape
         L = self.args.taps  #path taps
         
         h = self.sample_h_mp([T, B, L, 2], L)
         
         # Reshape h for real and imaginary parts
         h_r = h[:, :, :, 0].view(T, B, L).to(self.device)
         h_i = h[:, :, :, 1].view(T, B, L).to(self.device)
         
         # Reshape signal x to real and imag part
         if(torch.is_complex(x)):
            x_r = x.real
            x_i = x.imag
         else:
            x_r = x
            x_i = torch.zeros_like(x)

         # Complex Convolution  
         o_r = self.convolution(x_r, h_r) - self.convolution(x_i, h_i)
         o_i = self.convolution(x_r, h_i) + self.convolution(x_i, h_r)
             
         
         fading_x = torch.view_as_complex(torch.stack((o_r, o_i), dim=-1))

         # Add AWGN
         sig_avg_pwr = torch.mean(abs(x)**2)
         sigma2 = sig_avg_pwr*10**(-self.snr_db/10)
         output_comp = self.AWGNComplex_channel(fading_x, sigma2)
         
         return output_comp
    
     def MultipathReal_channel(self, x):
         '''SISO multipath real channel'''
         [T, B, N] = x.shape
         #path taps
         L = self.args.taps  
         
         h = self.sample_h_mp([T, B, L, 1], L)
         
         # Reshape h for real and imaginary parts
         h = h.view(T, B, L).to(self.device)
         
         # Real Convolution  
         fading_x = self.convolution(x, h)
             
         # Add real AWGN
         sig_avg_pwr = torch.mean(abs(x)**2)
         sigma2 = sig_avg_pwr*10**(-self.snr_db/10)
         output_comp = self.AWGN_channel(fading_x, sigma2)
         
         return output_comp

     
     def noiseless_channel(self, x):
        '''Noiseless channel'''
        return x
     
     def AWGN_channel(self, x, sigma2 = None):
        '''AWGN real channel'''
        if (sigma2 == None):
            sig_avg_pwr = torch.mean(x.abs()**2)
            sigma2 = sig_avg_pwr*10**(-self.snr_db/10)

        #Normal(0, sigma2)
        noise = torch.sqrt(sigma2) * torch.randn_like(x) 
        output = x + noise
        
        return output
     
     def AWGNComplex_channel(self, x, sigma2 = None):
        '''AWGN complex channel'''
        if (sigma2 == None):
            sig_avg_pwr = torch.mean(x.abs()**2)
            sigma2 = sig_avg_pwr*10**(-self.snr_db/10)

        if(torch.is_complex(x)):
            x_r = x.real
            x_i = x.imag
        else:
            x_r = x
            x_i = torch.zeros_like(x_r)
        
        # Generate random noise
        noise_real = torch.sqrt(sigma2/2) * torch.randn_like(x_r)
        noise_imag = torch.sqrt(sigma2/2) * torch.randn_like(x_i)

        if (self.out_typ == 'complex'):
            # complex output
            output_comp = torch.view_as_complex(torch.stack((x_r + noise_real, x_i + noise_imag), dim=-1))
        else:
            output_comp = torch.cat((x_r + noise_real, x_i + noise_imag), dim=-1)
        
        return output_comp
    
    
     def Rayleigh_channel(self, x):
        '''
        SISO Rayleigh flat fading Channel
        Same as 1-tap SISO multipath real channel
        '''
        [T, B, N] = x.shape
        G = T * B
         
        # [G, N]
        if(torch.is_complex(x)):
            x_r = x.real
            x_i = x.imag
        else:
            x_r = x
            x_i = torch.zeros_like(x_r)
        
        x_r = x_r.reshape(G, -1) 
        x_i = x_i.reshape(G, -1)

        # [G, 1]
        h_r =  1 / torch.sqrt(torch.tensor(2.)) * torch.normal(mean=0.0, std=1, size=[G, 1])
        h_i =  1 / torch.sqrt(torch.tensor(2.)) * torch.normal(mean=0.0, std=1, size=[G, 1])
        
        h_r = h_r.to(self.device)
        h_i = h_i.to(self.device)
        
        o_r = torch.reshape((x_r * h_r - x_i * h_i),(T, B, N))
        o_i = torch.reshape((x_r * h_i + x_i * h_r), (T, B, N))

        # Complex
        fading_x = torch.view_as_complex(torch.stack((o_r, o_i), dim=-1))

        # Add AWGN
        sig_avg_pwr = torch.mean(x.abs()**2)
        sigma2 = sig_avg_pwr*10**(-self.snr_db/10)
        output_comp = self.AWGNComplex_channel(fading_x, sigma2)
        
        return output_comp
     

     
    #  # convolution function
    #  def convolution(self, x, h):
    #     '''
    #     Enable batch-wise convolution using group convolution operations
    #     x: [T, B, N + L]
    #     h: [T, B, L]
    #     '''
    #     assert x.shape[0] == h.shape[0]
    #     [T, B, L] = h.shape
    #     G = T*B

    #     x = x.reshape(G, -1, 1)  # [G, N + L, 1]
    #     h = h.reshape(G, L, 1) # [G, L, 1]

    #     y = x * h[:, 0, 0].view(-1, 1, 1)
    #     for i in range(1, h.shape[1]):
    #         cur = x * h[:, i, 0].view(-1, 1, 1)
    #         cur = torch.cat([cur[:, -i:], cur[:, :-i]], 1)
    #         y += cur
    #     # [G, N + L, 1]
    #     y = y.reshape(T, B, -1)
    #     print(y.dtype)
    #     return y 
     

    
