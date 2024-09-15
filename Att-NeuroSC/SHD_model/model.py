import torch
import torch.nn as nn 
from spikingjelly.activation_based import base, layer, surrogate, neuron
from modules import Channel, SHD_Encoder,SHD_Decoder
class SHD_JSCC(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.encoder = SHD_Encoder(args)
        self.channel = Channel(args)
        self.decoder = SHD_Decoder(args)
       
    def forward(self, x: torch.Tensor):

        feature = self.encoder(x)
        Rx_feature = self.channel(feature)
        classification = self.decoder(Rx_feature)

        if self.args.net == 'SNN':
            # fire_rate = torch.sum(feature)/feature.numel() #L1 loss
            fire_out = feature
            return classification, fire_out
        else:
            return classification 
        
        
