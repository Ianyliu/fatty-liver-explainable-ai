import numpy as np
import pandas as pd
import torch
import ast
from torch_geometric.utils import subgraph
from torch_geometric.data import Dataset, Data
from numpy.random import choice
from PIL import Image

import os
from dotenv import load_dotenv
load_dotenv()

CROP_IMAGE_DIR = os.getenv('CROP_IMAGE_DIR_PATH')

class AUSDataset_train(Dataset):
    
    def __init__(self, task, meta_data_dir, source_data_dir, data_id, pretrained_encoder_id, data_aug, device, transform=None):
        
        super().__init__(meta_data_dir, data_id)
        
        self.task = task
        
        meta_data_name = meta_data_dir+'/train_dataset'+str(data_id)+'.csv'
        self.meta_data = pd.read_csv(meta_data_name, sep=",")
        
        self.source_data_dir = source_data_dir
        self.transform = transform
        self.pretrained_encoder_id = pretrained_encoder_id
        self.data_aug = data_aug
        self.device = device
        
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]


    def len(self):
        return len(self.meta_data)

    def get(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        mi_id = self.meta_data['MI_ID'][idx]
        
        data_name = self.source_data_dir+'/pretrained_'+self.pretrained_encoder_id+'_'+mi_id+'.pt'
        
        data = torch.load(data_name)
        
        x = data.x.to(self.device)
        y = data.y
        
        if self.task=="2_class":
            
            if y > 0:
                y = 1
        
        if self.task=="3_class":
            
            if y==2:
                y=1

            if y==3:
                y=2
        
        if self.task=="3_class_ms":
            
            if y==3:
                y=2
        
        
        if self.task=="uncertainty":
            
            if y != 4:
                y = 0
        
            if y == 4:
                y = 1
        
        
        if self.data_aug == "corr":
            
            #edge_index = data.edge_index_corr.to(self.device)
            edge_index = data.edge_index_corr.to(self.device)
            
            train_mask = torch.unsqueeze(torch.tensor(choice([0, 1], len(x), p=[0.0, 1.0])), 1)
            train_mask = train_mask.to(self.device)
            x = x*(train_mask) 
            
            #indices, _=(train_mask!=0).nonzero(as_tuple=True)
            #edge_index2, _ = subgraph(subset=indices, edge_index=edge_index, edge_attr=None)
            
            #edge_index = edge_index2.to(self.device)
        
        if self.data_aug == "subgraph":
            
            edge_index = data.edge_index.to(self.device)
            
            train_mask = torch.unsqueeze(torch.tensor(choice([0, 1], len(x), p=[0.2, 0.8])), 1)
            indices, _=(train_mask!=0).nonzero(as_tuple=True)
            
            x = x*(train_mask.to(self.device)) 
            
            edge_index, _ = subgraph(subset=indices, edge_index=edge_index, edge_attr=None)
            

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
        
        
        return data

    
class AUSDataset_valid(Dataset):
    
    def __init__(self, task, meta_data_dir, source_data_dir, data_id, pretrained_encoder_id, data_aug, device, transform=None):
        
        super().__init__(meta_data_dir, data_id)
        
        self.task = task
        
        meta_data_name = meta_data_dir+'/valid_dataset'+str(data_id)+'.csv'
        self.meta_data = pd.read_csv(meta_data_name, sep=",")
        
        self.source_data_dir = source_data_dir
        self.transform = transform
        self.pretrained_encoder_id = pretrained_encoder_id
        self.data_aug = data_aug
        self.device = device        

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]


    def len(self):
        return len(self.meta_data)

    def get(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        mi_id = self.meta_data['MI_ID'][idx]
        
        data_name = self.source_data_dir+'/pretrained_'+self.pretrained_encoder_id+'_'+mi_id+'.pt'
        
        data = torch.load(data_name)
        
        x = data.x.to(self.device)
        y = data.y
        
        if self.task=="2_class":
            
            if y > 0:
                y = 1
        
        if self.task=="3_class":
            
            if y==2:
                y=1

            if y==3:
                y=2
        
        if self.task=="3_class_ms":
            
            if y==3:
                y=2
        
        if self.task=="uncertainty":
            
            if y != 4:
                y = 0
        
            if y == 4:
                y = 1
        
        
        if self.data_aug == "corr":
            edge_index = data.edge_index_corr.to(self.device)
        
        if self.data_aug == "subgraph":
            edge_index = data.edge_index.to(self.device)
            
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=1)
        
        return data    
    
class AUSDataset_test(Dataset):
    
    def __init__(self, task, meta_data_dir, source_data_dir, data_id, pretrained_encoder_id, data_aug, device, transform=None):
        
        super().__init__(meta_data_dir, data_id)

        self.task = task
        
        meta_data_name = meta_data_dir+'/test_dataset'+str(data_id)+'.csv'
        self.meta_data = pd.read_csv(meta_data_name, sep=",")
        
        self.source_data_dir = source_data_dir
        self.transform = transform
        self.pretrained_encoder_id = pretrained_encoder_id
        self.data_aug = data_aug
        self.device = device

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]


    def len(self):
        return len(self.meta_data)

    def get(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        mi_id = self.meta_data['MI_ID'][idx]
        
        data_name = self.source_data_dir+'/pretrained_'+self.pretrained_encoder_id+'_'+mi_id+'.pt'
        
        data = torch.load(data_name)
        
        x = data.x.to(self.device)
        y = data.y
        
        if self.task=="2_class":
            
            if y > 0:
                y = 1
        
        if self.task=="3_class":
            
            if y==2:
                y=1

            if y==3:
                y=2
        
        if self.task=="3_class_ms":
            
            if y==3:
                y=2
        
        if self.task=="uncertainty":
            
            if y != 4:
                y = 0
        
            if y == 4:
                y = 1
        
        
        if self.data_aug == "corr":
            edge_index = data.edge_index_corr.to(self.device)
        
        if self.data_aug == "subgraph":
            edge_index = data.edge_index.to(self.device)
            
        
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=1)
        
        
        return data
    
    
def dataset_container(task, expr_name, dataset_name, data_id, pretrained_encoder_id, data_aug, device):
    
    #if expr_name == "standard":
        
    meta_data_dir = 'fattyliver_'+str(task)+str(expr_name)+'_dataset_lists/dataset'+str(data_id)
    source_data_dir = 'fattyliver_'+str(pretrained_encoder_id)+'_'+str(dataset_name)+'_dataset'
        
    #if expr_name == "oversampling":

    #    meta_data_dir = 'liver_fatty_oversampling_dataset_lists/dataset'+str(data_id)
    #    source_data_dir = 'liver_fatty_'+str(pretrained_encoder_id)+'_gasex_dataset'
    
    #if expr_name == "512x512":

    #    meta_data_dir = 'liver_fatty_3_classes_cleaned_dataset_lists/dataset'+str(data_id)
    #    source_data_dir = 'liver_fatty_'+str(pretrained_encoder_id)+'_512x512_dataset'
        
    
    #if expr_name == "corr080":

    #    meta_data_dir = 'liver_fatty_3_classes_cleaned_dataset_lists/dataset'+str(data_id)
    #    source_data_dir = 'liver_fatty_'+str(pretrained_encoder_id)+'_corr080_dataset'
    
    #if expr_name == "corr095":

    #    meta_data_dir = 'liver_fatty_3_classes_cleaned_dataset_lists/dataset'+str(data_id)
    #    source_data_dir = 'liver_fatty_'+str(pretrained_encoder_id)+'_gasex_dataset'

    
    train_dataset=AUSDataset_train(task=task,
                                   meta_data_dir=meta_data_dir, 
                                   source_data_dir=source_data_dir, 
                                   data_id=data_id, 
                                   pretrained_encoder_id=pretrained_encoder_id,
                                   data_aug = data_aug,
                                   device=device)
    
    valid_dataset=AUSDataset_valid(task=task,
                                   meta_data_dir=meta_data_dir, 
                                   source_data_dir=source_data_dir, 
                                   data_id=data_id, 
                                   pretrained_encoder_id=pretrained_encoder_id,
                                   data_aug = data_aug,
                                   device=device)
    
    test_dataset=AUSDataset_test(task=task,
                                 meta_data_dir=meta_data_dir,
                                 source_data_dir=source_data_dir, 
                                 data_id=data_id, 
                                 pretrained_encoder_id=pretrained_encoder_id, 
                                 data_aug = data_aug,
                                 device=device)
    
    # random permutation #
    
    input_dim = train_dataset[0].x.shape[1]

    
    return train_dataset, valid_dataset, test_dataset, input_dim




class AUSDataset_all(Dataset):
    
    def __init__(self, task, meta_data_name, source_data_dir, pretrained_encoder_id, data_aug, device, transform=None):
        
        super().__init__()

        self.task = task
        
        #meta_data_name = 'TWB_ABD_4_classes_cleaned_50_23072022.csv'
        #source_data_dir = 'liver_fatty_'+str(pretrained_encoder_id)+'_gasex_dataset'
     
        self.meta_data = pd.read_csv(meta_data_name, sep=",")
        
        self.source_data_dir = source_data_dir
        self.transform = transform
        self.pretrained_encoder_id = pretrained_encoder_id
        self.data_aug = data_aug
        self.device = device

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]


    def len(self):
        return len(self.meta_data)

    def get(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        mi_id = self.meta_data['MI_ID'][idx]
        
        data_name = self.source_data_dir+'/pretrained_'+self.pretrained_encoder_id+'_'+mi_id+'.pt'
        
        data = torch.load(data_name)
        
        x = data.x.to(self.device)
        y = data.y
        
        if self.task=="2_classes":
            
            if y > 0:
                y = 1
        
        if self.task=="3_classes":
            
            if y==2:
                y=1

            if y==3:
                y=2
                  
        if self.task=="3_classes_ms":
            
            if y==3:
                y=2
        
        if self.task=="uncertainty":
            
            if y != 4:
                y = 0
        
            if y == 4:
                y = 1
        
        if self.data_aug == "corr":
            edge_index = data.edge_index_corr.to(self.device)
        
        if self.data_aug == "subgraph":
            edge_index = data.edge_index.to(self.device)
            
        
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=1)
        
        
        return data


def single_data_loader(mi_id, img_id_list, image_transform, pretrained_image_encoder, y, num_classes, device, crop_image_dir = None):
    
    ## Encode attributes ##
    
    img_list = []
    x_attr = []
    x = []
    if crop_image_dir is None:
        crop_image_dir = CROP_IMAGE_DIR
    
    for i in range(len(img_id_list)):

        img_path = crop_image_dir +str(mi_id)+'_'+str(img_id_list[i])+'.jpg'
        img_list.append(image_transform(Image.open(img_path)))
        
    images = torch.stack(img_list, dim=0).to(device)
                                        
    #pretrained_image_encoder.eval() 
    x = pretrained_image_encoder(images).detach()
    
    # create links #

    n = len(img_id_list)

    tail=[]
    head=[]

    for u in range(n):
        for v in range(n):

            if v!=u:
                tail.append(u)
                head.append(v)

    edge_index = torch.tensor([tail,head], dtype=torch.long) 
    
    # create links based on correlation > 0.95 #

    corr_x = torch.corrcoef(x)
    corr_adj = (corr_x > 0.95) + 0

    tail=[]
    head=[]

    for u in range(n):
        for v in range(n):

            if (v!=u) and (corr_adj[u,v] > 0):
                tail.append(u)
                head.append(v)

    edge_index_corr = torch.tensor([tail,head], dtype=torch.long) 

    
    if num_classes==2:
            
            if y > 0:
                y = 1
                  
    if num_classes==3:
            
            if y==3:
                y=2
    
    
    mydata=Data(x = x, x_attr=x_attr, edge_index=edge_index, edge_index_corr=edge_index_corr, y=y, corr_adj=corr_adj)
    
    return mydata
