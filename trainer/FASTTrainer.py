import csv
import os, sys
import pdb
from random import randint
import time
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from trainer.base import BaseTrainer
from utils.meters import AvgMeter
from utils.eval import predict, calc_accuracy
from utils.utils_pointcloud import write_obj
from pytorch3d.loss import chamfer_distance
import numpy as np
import cv2



class FASTrainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, lr_scheduler, device, trainloader, valloader, writer,
                 logs_path, model_path=''):
        super(FASTrainer, self).__init__(cfg, network, optimizer, lr_scheduler, device, trainloader,
                                         valloader, writer)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.network = torch.nn.DataParallel(self.network)
        
        self.network = self.network.to(device)
        
        self.writer = writer

        # ! AvgMeter not using tensorboard writer
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train',
                                          num_iter_per_epoch=len(self.trainloader),
                                          per_iter_vis=True)

        self.train_acc_metric = AvgMeter(writer=writer, 
                                         name='Accuracy/train', num_iter_per_epoch=len(self.trainloader),
                                         per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))
        
        
        self.logs_path = logs_path

        self.weights_path = os.path.join(logs_path, 'weights')
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
            
        if model_path is not None:
            self.load_model(model_path)
            print(f"Model {model_path} loaded!")
        else:
            self.last_epoch = 0
            

    def load_model(self, model_path):
        state = torch.load(model_path)

        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])
        self.lr_scheduler.load_state_dict(state['scheduler'])
        self.last_epoch = state['epoch'] + 1

    def save_model(self, epoch):
        
        saved_name = os.path.join(self.weights_path,
                                  f"{self.cfg['model']['base']}_"\
                                  f"{self.cfg['dataset']['name']}_{epoch}.pth")

        state = {
            'epoch': epoch,
            'scheduler': self.lr_scheduler.state_dict(),
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(state, saved_name)
    
    def train_one_epoch(self, epoch, args):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        pbar = tqdm(self.trainloader, total=len(self.trainloader), dynamic_ncols=True)

        # Bernardo
        if args.save_samples:
            dir_samples = f'train_samples/epoch={epoch}'
            path_dir_samples = os.path.join(self.logs_path, dir_samples)
            os.makedirs(path_dir_samples, exist_ok=True)

        for i, (img, point_map, label, _) in enumerate(pbar):
            # if i >= 100:
            #     break
            
            point_map[label > 0, 2] = point_map[label > 0, 2] + 0.1
            point_map[label == 0, 2] = point_map[label == 0, 2] - 0.1
            
            #! Equilibrar as classes por batch
            img, point_map, label = img.to(self.device), point_map.to(self.device), label.to(self.device)
            # B x C x Num_Points
            # print()
            # pm = torch.mean
            cond = label == 0
            # print(torch.mean(point_map[cond], dim=(0, 2)))
            # print(torch.amax(point_map[cond], dim=(0, 2)))
            # print(torch.amin(point_map[cond], dim=(0, 2)))
            # print()
            net_point_map = self.network(img)
            
            self.optimizer.zero_grad()
            
            loss, _ = chamfer_distance(point_map, net_point_map)

            loss.backward()
            self.optimizer.step()
            #torch.cuda.synchronize()

            # print("==========  [DEBUG]  ==========")
            preds, score = predict(net_point_map)
            
            # print('-'*50)
            # print(score[cond])

            # print(score)
            # print(label)
            accuracy = calc_accuracy(preds, label)

            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy)

            pbar.set_description(f"Epoch {epoch}/Train - Loss: {self.train_loss_metric.avg:.4f}, "\
                                 f"Acc: {self.train_acc_metric.avg:.4f}")
            
            # Bernardo
            if args.save_samples and i==0:
                print(f'Saving samples at \'{path_dir_samples}\'...')
                self.save_samples(imgs=img, gt_pcs=point_map, gt_labels=label,
                                  pred_pcs=net_point_map, pred_labels=preds, pred_scores=score,
                                  batch_idx=i, path_to_save=path_dir_samples)
        
        epoch_loss = self.train_loss_metric.avg
        epoch_acc = self.train_acc_metric.avg
        
        return epoch_loss, epoch_acc
    
    def validate_one_epoch(self, epoch, args):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)

        pbar = tqdm(self.valloader, total=len(self.valloader), dynamic_ncols=True)

        # Bernardo
        if args.save_samples:
            dir_samples = f'val_samples/epoch={epoch}'
            path_dir_samples = os.path.join(self.logs_path, dir_samples)
            os.makedirs(path_dir_samples, exist_ok=True)
        
        with torch.no_grad():
            for i, (img, point_map, label, _) in enumerate(pbar):
                # if i >= 100:
                #     break
                
                point_map[label > 0, 2] = point_map[label > 0, 2] + 0.1
                point_map[label == 0, 2] = point_map[label == 0, 2] - 0.1
            
                img, point_map, label = img.to(self.device), point_map.to(self.device), label.to(self.device)
                net_point_map = self.network(img)
                
                loss, _ = chamfer_distance(point_map, net_point_map)

                # print("==========  [DEBUG]  ==========")
                preds, score = predict(net_point_map)
                
                # print(score)
                # print(label)

                accuracy = calc_accuracy(preds, label)

                # Update metrics
                self.val_loss_metric.update(loss.item())
                self.val_acc_metric.update(accuracy)
                
                pbar.set_description(f"Epoch {epoch}/Val - Loss: {self.val_loss_metric.avg:.4f}, "\
                        f"Acc: {self.val_acc_metric.avg:.4f}")

                # Bernardo
                if args.save_samples and i==0:
                    print(f'Saving samples at \'{path_dir_samples}\'...')
                    self.save_samples(imgs=img, gt_pcs=point_map, gt_labels=label,
                                      pred_pcs=net_point_map, pred_labels=preds, pred_scores=score,
                                      batch_idx=i, path_to_save=path_dir_samples)

        return self.val_loss_metric.avg, self.val_acc_metric.avg
        
    def train(self, args):
        
        min_loss = float('inf')
        
        for epoch in range(self.last_epoch, self.cfg['train']['num_epochs']):

            train_epoch_loss, train_epoch_acc = self.train_one_epoch(epoch, args)

            val_epoch_loss, val_epoch_acc = self.validate_one_epoch(epoch, args)
            
            print(f'\nEpoch: {epoch}, Train loss: {train_epoch_loss:.4f}, Val loss: {val_epoch_loss:.4f}, '\
                f'Train Acc: {train_epoch_acc:.4f}, Val Acc: {val_epoch_acc:.4f} - Lr: {self.lr_scheduler.get_last_lr()[0]}\n')
            
            self.lr_scheduler.step()
            
            self.writer.add_scalar('Loss/Train', train_epoch_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_epoch_loss, epoch)
            
            self.writer.add_scalar('Acc/Train', train_epoch_acc, epoch)
            self.writer.add_scalar('Acc/Val', val_epoch_acc, epoch)

            # TODO: Implement early stopping
            if val_epoch_loss < min_loss:
                min_loss = val_epoch_loss
                self.save_model(epoch)

    # Bernardo
    def save_samples(self, imgs, gt_pcs, gt_labels,
                     pred_pcs, pred_labels, pred_scores,
                     batch_idx, path_to_save):
        dir_batch = f'batch_{batch_idx}'
        path_dir_batch = os.path.join(path_to_save, dir_batch)
        os.makedirs(path_dir_batch, exist_ok=True)

        for i in range(imgs.shape[0]):
            img, gt_pc, gt_label, \
            pred_pc, pred_label, pred_score = imgs[i].cpu().numpy(), gt_pcs[i].detach().cpu().numpy(), gt_labels[i].detach().cpu().numpy(), \
                                              pred_pcs[i].detach().cpu().numpy(), pred_labels[i].detach().cpu().numpy(), pred_scores[i].detach().cpu().numpy()
            sample_name = f'sample={i}_gtlabel={gt_label}_predlabel={pred_label}_score=' + '{:.3f}'.format(pred_score)

            img_rgb = np.transpose(img, (1, 2, 0))  # from (3,224,224) to (224,224,3)
            img_rgb = (((img_rgb*0.5)+0.5)*255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            path_img = os.path.join(path_dir_batch, f'{sample_name}_img.png')
            cv2.imwrite(path_img, img_bgr)

            gt_pc = np.transpose(gt_pc, (1, 0))
            path_true_pc = os.path.join(path_dir_batch, f'{sample_name}_true_pointcloud.obj')
            write_obj(path_true_pc, gt_pc)

            pred_pc = np.transpose(pred_pc, (1, 0))
            path_pred_pc = os.path.join(path_dir_batch, f'{sample_name}_pred_pointcloud.obj')
            write_obj(path_pred_pc, pred_pc)



