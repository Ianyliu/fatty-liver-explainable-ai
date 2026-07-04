import numpy as np
import pandas as pd
import torch
import ast
import torch_geometric.utils.subgraph as subgraph
from torch_geometric.data import Dataset, Data
from numpy.random import choice
import training.forward_backward_prop as forward_backward_prop

def conformal_score_calculation(num_classes, 
                                encoder, 
                                pretrained_model, 
                                loss, 
                                optimizer, 
                                data_loader, 
                                batch_size, 
                                perform_eval, 
                                alpha,
                                device):
     
    l = 0
    n = 0
    score_list = 0
    
    with torch.no_grad():
    
        encoder.eval()

        for batch in data_loader:
            
            _, y, h, loss_fn = forward_backward_prop(encoder=encoder,
                                                     pretrained_model=pretrained_model,
                                                     loss=loss,
                                                     optimizer=optimizer,
                                                     batch=batch,
                                                     device=device, 
                                                     mode="forwardpropagation")
            
            for i in range(len(batch)): 
                
                score_list.append(score_stat(h[i], y[i]))
            
            
            #crt += perform_eval(prediction=h, ground_truth=y, mode="count")
            #conf_m += confusion_matrix(num_classes=num_classes, prediction=h, ground_truth=y)
            
            l += loss_fn.item() * y.size(0)
            n += y.size(0)
            
        return l / n, crt / n, conf_m
