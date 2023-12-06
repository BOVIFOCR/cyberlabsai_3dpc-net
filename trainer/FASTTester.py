import csv
import os, sys
import pdb
from random import randint
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from trainer.base import BaseTrainer
from utils.meters import AvgMeter
from utils.eval import predict, calc_accuracy, calc_tp_tn_fp_fn, calc_fas_metrics
from utils.utils_pointcloud import write_obj
from pytorch3d.loss import chamfer_distance
import csv
import numpy as np
import cv2



class FASTTester(BaseTrainer):
    def __init__(self, cfg, network, device, testloader, csv_path, exp_folder,
                 model_path=''):
        
        # Initialize variables
        self.cfg = cfg
        self.network = network
        self.device = device
        self.testloader = testloader
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.network = torch.nn.DataParallel(self.network)
        
        self.network = self.network.to(device)

        # ! AvgMeter not using tensorboard writer
        self.test_acc_metric = AvgMeter(writer='', 
                                         name='Accuracy/train', num_iter_per_epoch=len(self.testloader),
                                         per_iter_vis=True)
        
        

        self.load_model(model_path)
        print(f"Model {model_path} loaded!")
        
        self.exp_folder = exp_folder
        
        self.csv_path = csv_path
            

    def load_model(self, model_path):
        
        print(model_path)

        state = torch.load(model_path)

        self.network.load_state_dict(state['state_dict'])
        self.network.eval()
        
    def save_csv(self, filename, label, score):
        dataset_info = self.cfg['dataset']['name'] + '_' + self.csv_path
        save_path = os.path.join(self.exp_folder, dataset_info)
        label_dict = {True: 'Live', False: 'Spoof'}
        # open the file in the write mode
        with open(save_path, 'w') as f:
            # create the csv writer
            writer = csv.writer(f, delimiter=',')
            
            
            for i in range(len(filename)):
                # write a row to the csv file
                writer.writerow([filename[i], label_dict[label[i]], score[i]])
        
    
    def test(self, args):
        self.test_acc_metric.reset(0)

        pbar = tqdm(self.testloader, total=len(self.testloader), dynamic_ncols=True)
        
        # Open CSV file
        # dataset_info = self.cfg['dataset']['name'] + '_' + self.csv_path
        # save_path = os.path.join(self.exp_folder, dataset_info)
        # csv_file = open(save_path, 'w')
        # writer = csv.writer(csv_file, delimiter=',')
        filenames = []
        labels = []
        scores = []

        # Bernardo
        if args.save_samples:
            dir_samples = 'test_samples'
            path_dir_samples = os.path.join(args.exp_folder, dir_samples)
            os.makedirs(path_dir_samples, exist_ok=True)

        tp_total, fp_total, tn_total, fn_total = 0., 0., 0., 0.
        
        for i, (img, point_map, label, filename) in enumerate(pbar):
            # if i >= 100:
            #     break
            
            #! Point cloud maps don't come normalized in test
            img, point_map, label = img.to(self.device), point_map.to(self.device), label.to(self.device)
            net_point_map = self.network(img)
            
            # point_map[label > 0, 2] = point_map[label > 0, 2] + 0.1
            # point_map[label == 0, 2] = point_map[label == 0, 2] - 0.1

            # print(filename)
            # print("==========  [DEBUG]  ==========")
            cond = label > 0
            # print(torch.mean(point_map[cond], dim=(0, 2)))
            # print(torch.amax(point_map[cond], dim=(0, 2)))
            # print(torch.amin(point_map[cond], dim=(0, 2)))
            # print("==========  [DEBUG]  ==========")
            preds, score = predict(net_point_map)

            # print('-'*50)
            # print(score[cond])
            # preds, score = predict(net_point_map)

            # print(score)
            # print(label)
            accuracy = calc_accuracy(preds, label)

            tp_batch, fp_batch, tn_batch, fn_batch = calc_tp_tn_fp_fn(preds, label)
            tp_total += tp_batch
            fp_total += fp_batch
            tn_total += tn_batch
            fn_total += fn_batch

            # Update metrics
            self.test_acc_metric.update(accuracy)
            
            pbar.set_description(f"Test - Acc: {self.test_acc_metric.avg:.4f}")
            
            filenames.append(filename)
            labels.append(label.to('cpu'))
            scores.append(score.to('cpu'))

            # Bernardo
            if args.save_samples and i==0:
                print(f'Saving samples at \'{path_dir_samples}\'...')
                self.save_samples(imgs=img, gt_pcs=point_map, gt_labels=label,
                                  pred_pcs=net_point_map, pred_labels=preds, pred_scores=score,
                                  batch_idx=i, path_to_save=path_dir_samples)
        
        Acc = self.test_acc_metric.avg
        
        print(f'\nTest Acc: {Acc}\n')
        
        print("Saving CSV...")
        filenames = np.concatenate(filenames, axis=0)
        labels = np.concatenate(labels, axis=0)
        scores = np.concatenate(scores, axis=0)
        
        self.save_csv(filenames, labels, scores)

        metrics_total = calc_fas_metrics(tp_total, fp_total, tn_total, fn_total)
        for key in metrics_total.keys():
            print(f'{key}: {metrics_total[key]}')
        print()


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
            
        # sys.exit(0)


