import torch
from spikingjelly.activation_based import functional, neuron
import datetime
import os
import time
from torch.cuda import amp
import torch.nn.functional as F

class Train_SHD:
    def __init__(self, net, optimizer, lr_scheduler, args, train_data_loader, test_data_loader,start_epoch,scaler,max_test_acc):
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.max_test_acc = 0
        minloss = float('inf')
        self.scaler = scaler
        epoch = start_epoch

        for epoch in range(start_epoch, self.args.epochs):
            self.net.train()
            train_loss = 0
            train_acc = 0
            train_fr = 0
            train_samples = 0
            for frame, label in self.train_data_loader:
                self.optimizer.zero_grad()
                frame = frame.to(self.args.device)
                frame = frame.transpose(0, 1) 
                label = label.to(self.args.device)
                label_onehot = F.one_hot(label.to(torch.int64), 20).float()
                if self.scaler is not None:
                    with amp.autocast():
                        if  self.args.loss == "MSE":
                             if self.args.net == 'SNN':
                                out_fr, fire_out= self.net(frame)
                                fire_rate = torch.sum(fire_out)/fire_out.numel()
                                out_fr = out_fr.mean(0)
                                loss = F.mse_loss(out_fr, label_onehot)

                                # L1 loss on total number of spikes
                                # reg_loss = self.args.alpha * torch.mean(torch.sum(fire_out)）

                                # L2 loss on spikes per neuron
                                reg_loss = self.args.alpha * torch.mean(torch.sum(torch.sum(fire_out,dim=0),dim=0)**2) 
                                
                                loss = loss + reg_loss
                             else:
                                out_fr = self.net(frame).mean(0)
                                loss = F.mse_loss(out_fr, label_onehot)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.args.loss == "MSE":
                        if  self.args.net == 'SNN':
                            out_fr, fire_out = self.net(frame)
                            fire_rate = torch.sum(fire_out)/fire_out.numel()
                            out_fr = out_fr.mean(0)
                            loss = F.mse_loss(out_fr, label_onehot)

                            # reg_loss = self.args.alpha * torch.sum(fire_out) # L1 loss on total number of spikes
                            reg_loss = self.args.alpha * torch.mean(torch.sum(torch.sum(fire_out,dim=0),dim=0)**2) # L2 loss on spikes per neuron
                            
                            loss = loss + reg_loss
                        else:
                            out_fr = self.net(frame).mean(0)
                            loss = F.mse_loss(out_fr, label_onehot)
                    loss.backward()
                    self.optimizer.step()

                train_samples += label.numel()
                train_loss += loss.item() * label.numel()
                train_acc += (out_fr.argmax(1) == label).float().sum().item()
                if  self.args.net == 'SNN':
                    train_fr += fire_rate.item() * label.numel()

                functional.reset_net(self.net)

            train_loss /= train_samples
            train_acc /= train_samples
            if self.args.net == 'SNN':
                train_fr /= train_samples
            self.lr_scheduler.step()

            self.net.eval()
            test_loss = 0
            test_acc = 0
            test_samples = 0
            test_fr = 0
            with torch.no_grad():
                for frame, label in self.test_data_loader:
                    frame = frame.to(self.args.device)
                    frame = frame.transpose(0, 1)   # [B, T, N] -> [T, B, N]
                    label = label.to(self.args.device)
                    label_onehot = F.one_hot(label.to(torch.int64), 20).float()
                    out_fr = None
                    if self.args.loss == "MSE":
                        if self.args.net == 'SNN':
                            out_fr, fire_out = self.net(frame)
                            fire_rate = torch.sum(fire_out)/fire_out.numel()
                            out_fr = out_fr.mean(0)
                            loss = F.mse_loss(out_fr, label_onehot)
                        else:
                            out_fr = self.net(frame).mean(0)
                            loss = F.mse_loss(out_fr, label_onehot)
                    test_samples += label.numel()
                    test_loss += loss.item() * label.numel()
                    test_acc += (out_fr.argmax(1) == label).float().sum().item()
                    if  self.args.net == 'SNN':
                        test_fr += fire_rate.item() * label.numel()
                    functional.reset_net(self.net)

            test_loss /= test_samples
            test_acc /= test_samples
            if self.args.net == 'SNN':
                test_fr /= test_samples

            save_max = False 
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                max_test_fr =  test_fr
                save_max = True
            
            # hhx: Resume minloss need to change
            save_minloss = False
            if test_loss < minloss:
                minloss  = test_loss
                minloss_acc =  test_acc
                minloss_fr =  test_fr
                save_minloss = True


            checkpoint = {
                'net': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }
        
            
            if args.channel == 'multipath' or 'multipathreal':
                suffix = f'{args.taps}{args.channel}_{args.snr_db}_{args.attention}_{args.net}_{args.alpha}'
            else:
                suffix = f'{args.channel}_{args.snr_db}_{args.attention}_{args.net}_{args.alpha}'
                
            if save_max:
                torch.save(checkpoint, os.path.join(self.args.out_dir, f'SHD_checkpoint_max_{self.args.T}_{suffix}.pth'))
            
            if save_minloss:
                torch.save(checkpoint, os.path.join(self.args.out_dir, f'SHD_checkpoint_minloss_{self.args.T}_{suffix}.pth'))

            torch.save(checkpoint, os.path.join(self.args.out_dir, f'SHD_checkpoint_latest_{self.args.T}_{suffix}.pth'))
        
            with open(os.path.join('./logs/txt', f'{args.dataset}_logs_{args.T}_{suffix}.txt'), 'a', encoding='utf-8') as args_txt:
                if self.args.net == 'SNN':
                    args_txt.write(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, train_fr ={train_fr: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, test_fr ={test_fr: .4f}, max_test_acc ={max_test_acc: .4f}, max_test_fr ={max_test_fr: .4f}, minloss_acc = {minloss_acc: .4f}, minloss_fr = {minloss_fr: .4f}\n')
                    print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, train_fr ={train_fr: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, test_fr ={test_fr: .4f}, max_test_acc ={max_test_acc: .4f}, max_test_fr ={max_test_fr: .4f}, minloss_acc = {minloss_acc: .4f}, minloss_fr = {minloss_fr: .4f}')
                else:
                    args_txt.write(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}\n')
                    print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
                args_txt.write(f'escape time = {(datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")}\n\n')
                print(f'escape time = {(datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")}\n')
