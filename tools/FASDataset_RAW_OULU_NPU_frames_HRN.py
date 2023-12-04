import sys
sys.path.append(".") # Adds higher directory to python modules path.
import os
import torch
from torchvision import transforms
from utils.utils import read_cfg
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import random
import glob


class FASDataset_RAW_OULU_NPU_frames_HRN(Dataset):
    ''' Dataloader for face PAD
    Args:
        root_dir (str): Root directory path
        txt_file (str): txt file to dataset annotation
        depth_map_size (int): Size of pixel-wise binary supervision map
        transform: takes in a sample and returns a transformed version
        smoothing (bool): use label smoothing
        num_frames (int): num frames per video per epoch
    '''
    def __init__(self, root_dir, protocol_id, frames_path, images_file, transform, smoothing, num_frames=1):
        super().__init__()
        self.root_dir = root_dir
        self.protocol_id = protocol_id
        self.protocols_path = os.path.join(root_dir, 'Protocols', 'Protocol_'+str(protocol_id))
        self.frames_path = frames_path

        # self.img_list, self.point_cloud_list, self.labels = self.load_image_data(os.path.join(self.root_dir, images_file))
        if images_file == 'train' or images_file == 'Train':
            self.root_dir_part = os.path.join(root_dir, 'train')
            self.protocol_file_path = os.path.join(self.protocols_path, 'Train.txt')
        elif images_file == 'dev' or images_file == 'Dev':
            self.root_dir_part = os.path.join(root_dir, 'dev')
            self.protocol_file_path = os.path.join(self.protocols_path, 'Dev.txt')
        elif images_file == 'test' or images_file == 'Test':
            self.root_dir_part = os.path.join(root_dir, 'test')
            self.protocol_file_path = os.path.join(self.protocols_path, 'Test.txt')
        else:
            raise Exception(f'Error, dataset partition not recognized: \'{images_file}\'')

        self.img_list, self.point_cloud_list, self.labels = self.load_image_data(os.path.join(self.frames_path, images_file.lower()))
        # print('self.img_list:', self.img_list)
        # sys.exit(0)

        self.transform = transform

        self.filenames = list(self.img_list.keys())

        self.num_frames = num_frames

        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99
            
    #=================================================================
    def load_image_data(self, path_to_train_images):
        img_paths = {}
        point_cloud_paths = {}
        labels = {}
        
        #check if folder is empty
        if (len(os.listdir(path_to_train_images))==0):
            print("Error: " + path_to_train_images + " is empty!!")
            sys.exit(0)

        pattern_img = '_input_face.jpg'
        pattern_pc = '_hrn_high_mesh_10000points.npy'

        for folder_name in sorted(os.listdir(path_to_train_images)):    # Bernardo
            path_folder_sample = os.path.join(path_to_train_images, folder_name)

            filePath = glob.glob(path_folder_sample + '/*' + pattern_img)
            assert len(filePath) > 0, f'Error, no file found with patter \'*{pattern_img}\' in path \'{path_folder_sample}\''
            assert len(filePath) < 2, f'Error, more than 1 file found with \'*{pattern_img}\' in path \'{path_folder_sample}\''
            filePath = filePath[0]
            # print('filePath:', filePath)

            point_cloud_path = glob.glob(path_folder_sample + '/*' + pattern_pc)
            assert len(point_cloud_path) > 0, f'Error, no file found with patter \'*{pattern_pc}\' in path \'{path_folder_sample}\''
            assert len(point_cloud_path) < 2, f'Error, more than 1 file found with \'*{pattern_pc}\' in path \'{path_folder_sample}\''
            point_cloud_path = point_cloud_path[0]
            # print('point_cloud_path:', point_cloud_path)
            # sys.exit(0)

            if folder_name not in img_paths:         img_paths[folder_name] = []
            if folder_name not in point_cloud_paths: point_cloud_paths[folder_name] = []
            if folder_name not in labels:            labels[folder_name] = []

            img_paths[folder_name].append(filePath)
            point_cloud_paths[folder_name].append(point_cloud_path)

            label=self.get_label(folder_name)
            labels[folder_name].append(label)

            # print('filePath:', filePath)
            # print('point_cloud_path:', point_cloud_path)
            # print('label:', label)
            # print('------------------')

        # print('img_paths:', img_paths)
        # print('point_cloud_paths:', point_cloud_paths)
        # print('labels:', labels)
        # sys.exit(0)

        return img_paths, point_cloud_paths, labels
    #=============================================================
    def get_label(self, string_input): # OULU protocol 1
        label=int(string_input[7])
        if label == 1 : return True #live
        else: return False # Spoof
    
    #=============================================================
    def __getitem__(self, index):
        ''' Get image, output map and label for a given index
        Args:
            index (int): index of image
        Returns:
             img (PIL image)
             mask: output map (32x32)
             label: 1 (live), 0 (spoof)
        '''
        #! Criar toda iteraçao uma nova lista com o tamanho de len(self.ids) * num_frames e pegar um item dessa lista
        #! Ou buscar dentro do id o numero de num_frames necessario
        #index = index % len(self.filenames)
        filename = self.filenames[index]
        
        imgPath = self.img_list[filename][0]                #[file_index]
        PointCloudPath = self.point_cloud_list[filename][0] #[file_index]
        label = self.labels[filename][0]                    #[file_index]

        img = Image.open(imgPath) # read image

        point_cloud_map = np.load(PointCloudPath) #! read pointcloud 10k points

        n_points = point_cloud_map
        rot_img_crop = img

        if self.transform:
            img = self.transform(rot_img_crop)
            n_points = transforms.ToTensor()(n_points.astype(np.float32)).squeeze()

        return img, n_points, label
    #=============================================================
    def __len__(self):
        return len(self.filenames) #* self.num_frames
        # return len(self.ids) #* self.num_frames

    #=============================================================
    def downsample(self,vertices, n_samples=2500): #Downsample point cloud vertices randomly
        # vertices_df = pd.DataFrame(vertices.transpose((1, 0)))   # original
        vertices_df = pd.DataFrame(vertices)                       # Bernardo

        samp= vertices_df.shape[0]

        if samp >= n_samples:
            vertices_downsampled = vertices_df.sample(n=n_samples, replace=False)
        else :
            vertices_downsampled = vertices_df.sample(n=n_samples, replace=True)

        vertices_downsampled = np.array(vertices_downsampled).transpose((1, 0))
        #vertices_downsampled = np.array(vertices_downsampled)
        return vertices_downsampled
    
#=================================================================
"""
if __name__=="__main__":
    
    cfg=read_cfg(cfg_file="../config/DPC3_NET_config.yaml")
    root_path= '../'
    images_path=cfg['dataset']['train_images']

    train_transform = transforms.Compose ([transforms.Resize(cfg['model']['input_size']),
                                         transforms.ToTensor(),
                                         transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])])
                                         #TODO: Check Normalization

    dt=FASDataset(root_dir=root_path,
                  images_file=images_path,
                  transform=train_transform,
                  smoothing=True)


    trainloader = torch.utils.data.DataLoader(dataset=dt,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=1)

    for i, (img, point_map, label) in enumerate(trainloader):
        print(img.size())
        print("i: " + str(i))
        #print(point_map)
        #print(label)
"""



    