"""
+ Author: Haoxiang Huang
+ Date: 20-Jan-2023
+ This is the auto batch training script
"""
import os
epochs = 500
lr = 0.0001
channels = ['multipath']
taps = [8]
snr_db = [-10, 14]
# snr_db = [i for i in range(-10, -4, 2)]
# snr_db = [i for i in range(-4, 4, 2)]
# snr_db = [i for i in range(4, 10, 2)]
# snr_db = [i for i in range(10, 15, 2)]
attention = ['STSC']
# attention = ['None']
nets = ['SNN']
order = 1e-10
alphas = [i * order for i in range(1, 10, 2)]

combinations = [
    (channel, snr, att, tap, net, alpha)
    for channel in channels
    for snr in snr_db
    for att in attention
    for tap in taps
    for net in nets
    for alpha in alphas
    # if not (net == 'ANN' and att == 'None')
]

for combo in combinations:
    channel, snr, att, tap, net, alpha = combo
    print(combo)
    if channel == 'multipath' or 'multipathreal':
            print(f'{tap} taps-multipath')
            cmd = f"python ../SHD_model/train_main.py -dataset SHD -T 15 -dt 60 -device cuda:0 -batch_size 64 -epochs {epochs} -opt adam -lr {lr} -loss MSE -net {net} -attention {att} -channel {channel} -snr_db {snr} -taps {tap} -alpha {alpha}"
            print(cmd)
            os.system(cmd)
    else:
        cmd = f"python ../SHD_model/train_main.py -dataset SHD -T 15 -dt 60 -device cuda:0 -batch_size 64 -epochs {epochs} -opt adam -lr {lr} -loss MSE -attention {att} -channel {channel} -snr_db {snr}"
        print(cmd)
        os.system(cmd)

