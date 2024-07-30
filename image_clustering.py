import inspect
import sys 
from tqdm import tqdm
import ast
import pandas as pd 
import datetime
import glob
import json 
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv
import seaborn as sns
from usflc_xai import models, datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import multiprocessing as mp

# from sklearnex import patch_sklearn
# patch_sklearn()

from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
)
from openTSNE import TSNE as open_TSNE
import umap

from communities.algorithms import spectral_clustering
from communities.algorithms import girvan_newman

from sklearn.cluster import KMeans
import time 
from sklearn.cluster import (AffinityPropagation, 
                             AgglomerativeClustering, 
                             SpectralClustering,
                            #  HDBSCAN,
                             )

from sklearn.decomposition import (KernelPCA, NMF, SparsePCA, 
                                   PCA, FactorAnalysis, LatentDirichletAllocation,
                                   DictionaryLearning, 
                                   )

from graph_based_clustering import SpanTreeConnectedComponentsClustering
from sklearn.pipeline import make_pipeline
from graph_based_clustering import ConnectedComponentsClustering
from communities.algorithms import louvain_method, hierarchical_clustering, girvan_newman
from communities.visualization import draw_communities
from communities.algorithms import hierarchical_clustering
load_dotenv()

# import tsnecuda
# tsnecuda.test()
from tsnecuda import TSNE as TSNE_GPU

from dataclasses import dataclass, field
import pandas as pd
from typing import List, Dict, Callable
from adjustText import adjust_text

@dataclass
class CustomClustering:
    metadata_path: str = 'meta_data/TWB_ABD_expand_modified_gasex_21072022.csv'
    test_data_path: str = field(init=False)
    test_data_id: str = "09"
    corr_result_dir: str = "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/07-12-2024-03-57-58/"
    image_dir: str = "/home/liuusa_tw/data/cropped_images/"
    mi_id_type: str = "groundtruth-positive"
    mi_id_min_imgs: int = 20
    image_encoder_id: str = "densenet121"
    graph_encoder_id: str = "SETNET_GAT"
    num_classes: int = 2
    input_dim: int = 1024
    num_layers: int = 1
    device: torch.device = field(init=False)
    metadata: pd.DataFrame = field(init=False)
    test_data: pd.DataFrame = field(init=False)
    _image_transform: Callable = field(init=False)
    _image_encoder: Callable = field(init=False)
    _graph_encoder: Callable = field(init=False)
    _ckpt_name: str = field(init=False)
    _miid_to_y_dict: dict = field(init=False)
    _miid_to_img_ids_dict: dict = field(init=False)
    _miid_to_img_filepaths_dict: dict = field(init=False)
    _miid_to_corr_csv_dict: dict = field(init=False)
    mi_ids: list = field(init=False)
    verbose: bool = False

    def __post_init__(self):
        self.test_data_path = 'fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset'+str(self.test_data_id)+'/test_dataset'+str(self.test_data_id)+'.csv'
        self._ckpt_name = 'model_tl_twbabd'+ str(self.test_data_id)+'/best_results.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._strip_all_str()
        self._verify_input()
        
        self.metadata = pd.read_csv(self.metadata_path)
        self.test_data = pd.read_csv(self.test_data_path)
        
        metadata_req_cols = ["MI_ID", "liver_fatty", "IMG_ID_LIST"]
        test_data_req_cols = ["MI_ID"]
        
        if not all(col in self.metadata.columns for col in metadata_req_cols):
            raise ValueError(f"Missing required columns in metadata: {metadata_req_cols}" + 
                             f"Please check {self.metadata_path}")
        if not all(col in self.test_data.columns for col in test_data_req_cols):
            raise ValueError(f"Missing required columns in testdata: {test_data_req_cols}" + 
                             f"Please check {self.test_data_path}")
        
        self._image_transform = self._setup_image_transform()
        self._image_encoder = self._setup_image_encoder()
        # self._graph_encoder = self._setup_graph_encoder()
        self.get_all_subj_metadata()
        
    def _strip_all_str(self):
        self.test_data_path = self.test_data_path.strip()    
        self.metadata_path = self.metadata_path.strip()    
        self.corr_result_dir = self.corr_result_dir.strip()    
        self.image_dir = self.image_dir.strip()    
        self._ckpt_name = self._ckpt_name.strip()    
        self.image_encoder_id = self.image_encoder_id.strip()    
        self.graph_encoder_id = self.graph_encoder_id.strip()    
        self.mi_id_type = self.mi_id_type.strip()      
    
    def _verify_input(self):
        if not isinstance(self.test_data_path, str):
            raise TypeError(f"test_data_path must be a string but is {type(self.test_data_path)} instead")
        if not isinstance(self.metadata_path, str):
            raise TypeError(f"metadata_path must be a string but is {type(self.metadata_path)} instead")
        if not isinstance(self.image_dir, str):
            raise TypeError(f"image_dir must be a string but is {type(self.image_dir)} instead")
        if not isinstance(self.corr_result_dir, str):
            raise TypeError(f"corr_result_dir must be a string but is {type(self.corr_result_dir)} instead")
        if not isinstance(self._ckpt_name, str):
            raise TypeError(f"_ckpt_name must be a string but is {type(self._ckpt_name)} instead")
        if not isinstance(self.num_classes, int):
            raise TypeError(f"num_classes must be a int but is {type(self.num_classes)} instead")
        if not isinstance(self.input_dim, int):
            raise TypeError(f"input_dim must be a int but is {type(self.input_dim)} instead")
        if not isinstance(self.num_layers, int):
            raise TypeError(f"num_layers must be a int but is {type(self.num_layers)} instead")
        if not isinstance(self.mi_id_min_imgs, int):
            raise TypeError(f"mi_id_min_imgs must be a int but is {type(self.mi_id_min_imgs)} instead")
        if not isinstance(self.image_encoder_id, str):
            raise TypeError(f"image_encoder_id must be a string but is {type(self.image_encoder_id)} instead")
        if not isinstance(self.graph_encoder_id, str):
            raise TypeError(f"graph_encoder_id must be a string but is {type(self.graph_encoder_id)} instead")
        if not isinstance(self.test_data_path, str):
            raise TypeError(f"test_data_path must be a string but is {type(self.test_data_path)} instead")
        if not isinstance(self.mi_id_type, str):
            raise TypeError(f"mi_id_type must be a string but is {type(self.mi_id_type)} instead")
        
        if not os.path.exists(self.metadata_path): 
            raise ValueError(f"metadata_path {self.metadata_path} is not a valid filepath") 
        if not os.path.exists(self.test_data_path): 
            raise ValueError(f"image_dir {self.test_data_path} is not a valid filepath")
        if not os.path.exists(self.corr_result_dir): 
            raise ValueError(f"corr_result_dir {self.corr_result_dir} is not a valid filepath") 
        if not os.path.exists(self.image_dir): 
            raise ValueError(f"image_dir {self.image_dir} is not a valid filepath") 
        if not os.path.exists(self._ckpt_name) or self._ckpt_name.split('.')[-1].strip() != "ckpt": 
            raise ValueError(f"image_dir {self._ckpt_name} is not a valid checkpoint, please verify the model checkpoint") 
                
        if self.num_classes <= 0: 
            raise ValueError(f"num_classes {self.num_classes} is not a valid positive (non-zero) integer") 
        if self.input_dim <= 0: 
            raise ValueError(f"input_dim {self.input_dim} is not a valid positive (non-zero) integer") 
        if self.num_layers <= 0: 
            raise ValueError(f"num_layers {self.num_layers} is not a valid positive (non-zero) integer") 
        if self.mi_id_min_imgs <= 0: 
            raise ValueError(f"mi_id_min_imgs {self.mi_id_min_imgs} is not a valid positive integer") 

        if self.image_encoder_id not in ['resnet50', 'densenet121', 'vitl16in21k']:
            raise ValueError(f"image_encoder_id {self.image_encoder_id} is not a valid model id " + 
                             "\n Valid model ids: ['resnet50', 'densenet121', 'vitl16in21k']")
        # if self._graph_encoder not in ['resnet50', 'densenet121', 'vitl16in21k']:
        #     raise ValueError(f"_graph_encoder {self._graph_encoder} is not a valid model id " + 
                            #  "\n Valid model ids: ['SETNET_GAT',]")
        if self.mi_id_type not in ['groundtruth-positive', 'groundtruth-negative']:
            raise ValueError(f"mi_id_type {self.mi_id_type} is not a valid mi_id type " + 
                             "\n Valid input: ['groundtruth-positive', 'groundtruth-negative']")

    def _setup_image_transform(self):
        # Set up image transformation
        
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def _setup_image_encoder(self):
        # Set up image encoder
        _, pretrained_image_encoder = models.image_encoder_model(name=self.image_encoder_id, 
                                        pretrained=True, 
                                        num_classes=self.num_classes, 
                                        device=self.device)
        pretrained_image_encoder = pretrained_image_encoder.eval() 
        return pretrained_image_encoder

    def create_data_loader(self, mi_id: str, img_id_list: list, y: int):
        if not isinstance(img_id_list, list):
            raise TypeError(f"img_id_list must be a list but is {type(img_id_list)} instead")
        if not isinstance(mi_id, str):
            raise TypeError(f"mi_id must be a string but is {type(mi_id)} instead")
        if not isinstance(y, int):
            raise TypeError(f"y must be an integer but is {type(y)} instead")
        if y < 0:
            raise ValueError(f"y must a non-negative finite integer but is {y} instead")
        
        return datasets.single_data_loader(mi_id=mi_id, 
                                           img_id_list=img_id_list, 
                                           image_transform=self._image_transform, 
                                           pretrained_image_encoder=self._image_encoder, 
                                           y=y, 
                                           num_classes=self.num_classes, 
                                           device=self.device)
    
    def get_mi_ids_y_dict(self):
        mi_id_to_y_dict = {mi_id: self.metadata[self.metadata['MI_ID']==mi_id]['liver_fatty'].to_list()[0]
                for mi_id in self.test_data['MI_ID']}
        if self.num_classes == 2:
            mi_id_to_y_dict = {mi_id: 1 if y > 0 else y 
                               for mi_id, y in mi_id_to_y_dict.items()
                               }
        if self.num_classes == 3:
            mi_id_to_y_dict = {mi_id: 2 if y == 3 else y 
                               for mi_id, y in mi_id_to_y_dict.items()
                               }
        
        return mi_id_to_y_dict
        
    def get_mi_ids_img_ids_dict(self):
        return {mi_id: ast.literal_eval(self.metadata[self.metadata['MI_ID']==mi_id]['IMG_ID_LIST'].to_list()[0])
                for mi_id in self.test_data['MI_ID']}
    
    def get_mi_ids(self):
        mi_ids = self.test_data['MI_ID'].to_list().copy()
        self._miid_to_y_dict = self.get_mi_ids_y_dict()
        self._miid_to_img_ids_dict = self.get_mi_ids_img_ids_dict()

        if self.mi_id_type == 'groundtruth-positive': 
            mi_ids = [mi_id for mi_id in mi_ids if self._miid_to_y_dict[mi_id] > 0 ]
        elif self.mi_id_type == 'groundtruth-negative': 
            mi_ids = [mi_id for mi_id in mi_ids if self._miid_to_y_dict[mi_id] <= 0]
            
        mi_ids = [mi_id for mi_id in mi_ids 
                  if len(self._miid_to_img_ids_dict[mi_id]) >= self.mi_id_min_imgs]
    
        if self.verbose:
            print(self.mi_id_type)
            print(f"Num of subjects {len(mi_ids)}")
        
        return mi_ids
        
    def get_mi_ids_img_filepaths(self):
        if self._miid_to_img_ids_dict is None:
            self._miid_to_img_ids_dict = self.get_mi_ids_img_ids_dict()
        if self.mi_ids is None: 
            self.mi_ids = self.get_mi_ids()
            
        mi_ids_img_filepaths = {mi_id: 
            [self.image_dir + mi_id + '_' + str(img_id) + '.jpg' for img_id in img_id_list] 
            for mi_id, img_id_list in self._miid_to_img_ids_dict.items()
            }

        return mi_ids_img_filepaths
    
    def get_mi_ids_corr_csv_filepaths(self):
        if self.mi_ids is None: 
            self.mi_ids = self.get_mi_ids()
            
        mi_id_corr_path_dict = {mi_id: os.path.join(self.corr_result_dir, mi_id, "encoded_img_corr.csv")
                           for mi_id in self._miid_to_img_ids_dict
            }

        return mi_id_corr_path_dict

        
    # def get_all_subj_img_embeddings(self):
    #     assert self._miid_to_img_ids_dict is not None
    #     assert self.mi_ids is not None
        
    #     if self._miid_to_img_filepaths_dict is None:
    #         self._miid_to_img_filepaths_dict = self.get_mi_ids_img_filepaths()
            
    # def get_all_subj_corr(self):
    #     return
    #     assert self._miid_to_img_ids_dict is not None
    #     assert self.mi_ids is not None
        
    #     if self._miid_to_img_filepaths_dict is None:
    #         self._miid_to_img_filepaths_dict = self.get_mi_ids_img_filepaths()
            
       
    def get_all_subj_metadata(self):
        self.mi_ids = self.get_mi_ids() 
        self._miid_to_y_dict = self.get_mi_ids_y_dict()
        self._miid_to_img_ids_dict = self.get_mi_ids_img_ids_dict()
        self._miid_to_img_filepaths_dict = self.get_mi_ids_img_filepaths()
        self._miid_to_corr_csv_dict = self.get_mi_ids_corr_csv_filepaths()
    
    def get_subject_data(self, mi_id: str) -> Dict:
        # Extract subject data, including image list and ground truth
        img_id_list = self._miid_to_img_ids_dict.get(mi_id)
        y = self._miid_to_y_dict.get(mi_id)
        subj_data_loader = datasets.single_data_loader(mi_id=mi_id,
                                img_id_list=img_id_list,
                                image_transform=self._image_transform,
                                pretrained_image_encoder=self._image_encoder,
                                y=y,
                                num_classes=self.num_classes,
                                device=self.device)
        
        subj_img_embedding = subj_data_loader.x.to('cpu').numpy()
        subj_adjacency_matrix_filepath = self._miid_to_corr_csv_dict.get(mi_id)
        subj_adjacency_matrix = pd.read_csv(subj_adjacency_matrix_filepath, index_col = 0).to_numpy()
        
        return {'subj_img_embedding': subj_img_embedding,
                'subj_adjacency_matrix': subj_adjacency_matrix,
                'subj_data_loader': subj_data_loader, 
                'mi_id': mi_id,
                'y': y,
                'img_id_list': img_id_list,
                }
    def try_all_clustering_methods(self, subj_data: dict, min_n_clusters: int = 3, max_clustering_iter: int = 1000):
        if self.verbose: 
            print("Clustering starting...")
        clustering_start = time.perf_counter()
        subj_dict_cols = ['subj_img_embedding','subj_adjacency_matrix','subj_data_loader','mi_id', 'y','img_id_list',]
        assert(all(col in subj_dict_cols for col in subj_data.keys()))
        
        subj_img_embedding = subj_data.get('subj_img_embedding') 
        subj_adjacency_matrix = subj_data.get('subj_adjacency_matrix') 
        mi_id = subj_data.get('mi_id') 
        img_id_list = subj_data.get('img_id_list') 
        max_n_clusters = max((len(img_id_list) - 3), min_n_clusters)
        
        # clustering_result_dict = {
        #     "span_tree_connected_components": self.get_span_tree_connected_components_clusterings(adj_matrix=subj_adjacency_matrix,
        #                                                                                           img_embedding=subj_img_embedding,
        #                                                                                           min_n_clusters=min_n_clusters,
        #                                                                                           max_n_clusters=max_n_clusters,),
        #     "connected_components": self.get_connected_components_clusterings(adj_matrix=subj_adjacency_matrix,
        #                                                                       img_embedding=subj_img_embedding,
        #                                                                       min_n_clusters=min_n_clusters,
        #                                                                       max_n_clusters=max_n_clusters,
        #                                                                       max_iter=max_clustering_iter,),
        #     "louvain": self.get_louvain_clusterings(adj_matrix=subj_adjacency_matrix,
        #                                             img_embedding=subj_img_embedding,
        #                                             min_n_clusters=min_n_clusters,
        #                                             max_n_clusters=max_n_clusters,),
        #     "hierarchical": self.get_hierarchical_clusterings(adj_matrix=subj_adjacency_matrix,
        #                                                       img_embedding=subj_img_embedding,
        #                                                       min_n_clusters=min_n_clusters,
        #                                                       max_n_clusters=max_n_clusters,),
        #     "kmeans": self.get_k_means_clusterings(
        #         img_embedding=subj_img_embedding,
        #         min_n_clusters=min_n_clusters,
        #         max_n_clusters=max_n_clusters,
        #         max_iter=max_clustering_iter,),
        #     "hdbscan": self.get_hdbscan_clusterings(
        #                                                img_embedding=subj_img_embedding,
        #                                                min_n_clusters=min_n_clusters,
        #                                                max_n_clusters=max_n_clusters,
        #                                                max_iter=max_clustering_iter,),
        #     "affinity_propagation": self.get_affinity_propagation_clusterings(
        #         img_embedding=subj_img_embedding,
        #         min_n_clusters=min_n_clusters,
        #         max_n_clusters=max_n_clusters,
        #         max_iter=max_clustering_iter,),
        #     "agglomerative": self.get_agglomerative_clusterings(
        #         img_embedding=subj_img_embedding,
        #         min_n_clusters=min_n_clusters,
        #         max_n_clusters=max_n_clusters,
        #         max_iter=max_clustering_iter,),
        #     "spectral_embedding_based": self.get_spectral_embedding_based_clusterings(
        #         img_embedding=subj_img_embedding,
        #         min_n_clusters=min_n_clusters,
        #         max_n_clusters=max_n_clusters,
        #         max_iter=max_clustering_iter,),
        # }
        
        clustering_result_dict = dict.fromkeys([
            "best_silhouette_score",
            "best_clustering_algorithm",
            "span_tree_connected_components_clusterings",
            "span_tree_connected_components_n_clusters",
            "span_tree_connected_components_best_silhouette_score",
            "connected_components_clusterings",
            "connected_components_best_threshold",
            "connected_components_best_metric",
            "connected_components_best_silhouette_score", 
            "louvain_clusterings",
            "louvain_best_silhouette_score",
            "hierarchical_clusterings",
            "hierarchical_best_metric",
            "hierarchical_best_linkage_type",
            "hierarchical_best_silhouette_score",
            "spectral_clustering_graph_based_clusterings",
            "spectral_clustering_graph_based_n_clusters",
            "spectral_clustering_graph_based_best_silhouette_score",
            "kmeans_clusterings",
            "kmeans_n_clusters",
            "kmeans_best_silhouette_score", 
            "hdbscan_clusterings",
            "hdbscan_best_silhouette_score", 
            "hdbscan_best_cluster_selection_method"
            "affinity_propagation_clusterings",
            "affinity_propagation_best_silhouette_score", 
            "agglomerative_clusterings",
            "agglomerative_best_n_clusters", 
            "agglomerative_best_silhouette_score", 
            "spectral_embedding_based_clusterings",
            "spectral_embedding_based_best_n_clusters",
            "spectral_embedding_based_best_affinity_method", 
            "spectral_embedding_based_best_silhouette_score", 
            "mds_embeddings_results",
            "isomap_embeddings_results",
            "tsne_embeddings_results",
            "tsne_best_perplexity", 
            "tsne_best_kl_divergence",
            "subj_result_df",
        ])
        
        

        start = time.perf_counter()
        span_tree_connected_components_results = self.get_span_tree_connected_components_clusterings(adj_matrix=subj_adjacency_matrix,
                                                                                                     img_embedding=subj_img_embedding,
                                                                                                     min_n_clusters=min_n_clusters,
                                                                                                     max_n_clusters=max_n_clusters,)
        end = time.perf_counter()
        span_tree_connected_components_results['runtime'] = end - start
        if self.verbose: 
            print(f"span_tree clustering took {end - start} seconds")
        # if self.verbose: 
        #     start = time.perf_counter()
        # connected_components_results = self.get_connected_components_clusterings(adj_matrix=subj_adjacency_matrix,
        #                                                                          img_embedding=subj_img_embedding,
        #                                                                          min_n_clusters=min_n_clusters,
        #                                                                          max_n_clusters=max_n_clusters,
        #                                                                          max_iter=max_clustering_iter,)
        # if self.verbose: 
        #     end = time.perf_counter()
        #     print(f"connected components clustering took {end - start} seconds")
        start = time.perf_counter()
        louvain_results = self.get_louvain_clusterings(adj_matrix=subj_adjacency_matrix,
                                                       img_embedding=subj_img_embedding,
                                                       min_n_clusters=min_n_clusters,
                                                       max_n_clusters=max_n_clusters,)
        end = time.perf_counter()
        louvain_results['runtime'] = end - start
        if self.verbose: 
            print(f"louvain clustering took {end - start} seconds")
        start = time.perf_counter()
        hierarchical_results = self.get_hierarchical_clusterings(adj_matrix=subj_adjacency_matrix,
                                                                 img_embedding=subj_img_embedding,
                                                                 min_n_clusters=min_n_clusters,
                                                                 max_n_clusters=max_n_clusters,)
        end = time.perf_counter()
        hierarchical_results['runtime'] = end - start 
        if self.verbose: 
            print(f"hierarchical clustering took {end - start} seconds")

        start = time.perf_counter()
        k_means_results = self.get_k_means_clusterings(
                                                       img_embedding=subj_img_embedding,
                                                       min_n_clusters=min_n_clusters,
                                                       max_n_clusters=max_n_clusters,
                                                       max_iter=max_clustering_iter,)
        end = time.perf_counter()
        k_means_results['runtime'] = end - start
        if self.verbose: 
            print(f"k_means clustering took {end - start} seconds")
        # hdbscan_results = self.get_hdbscan_clusterings(
        #                                                img_embedding=subj_img_embedding,
        #                                                min_n_clusters=min_n_clusters,
        #                                                max_n_clusters=max_n_clusters,
        #                                                max_iter=max_clustering_iter,)
        start = time.perf_counter()
        affinity_propagation_results = self.get_affinity_propagation_clusterings(
                                                                                 img_embedding=subj_img_embedding,
                                                                                 min_n_clusters=min_n_clusters,
                                                                                 max_n_clusters=max_n_clusters,
                                                                                 max_iter=max_clustering_iter,)
        end = time.perf_counter()
        affinity_propagation_results['runtime'] = end - start 
        if self.verbose: 
            print(f"affinity_propagation clustering took {end - start} seconds")
        start = time.perf_counter()
        agglomerative_results = self.get_agglomerative_clusterings(
                                                                   img_embedding=subj_img_embedding,
                                                                   min_n_clusters=min_n_clusters,
                                                                   max_n_clusters=max_n_clusters,
                                                                   max_iter=max_clustering_iter,)
        end = time.perf_counter()
        agglomerative_results['runtime'] = end - start
        if self.verbose: 
            print(f"agglomerative clustering took {end - start} seconds")
        
        start = time.perf_counter()
        spectral_embedding_based_results = self.get_spectral_embedding_based_clusterings(
                                                                                         img_embedding=subj_img_embedding,
                                                                                         min_n_clusters=min_n_clusters,
                                                                                         max_n_clusters=max_n_clusters,
                                                                                         max_iter=max_clustering_iter,)
        end = time.perf_counter()
        spectral_embedding_based_results['runtime'] = end - start 
        if self.verbose: 
            print(f"Spectral clustering took {end - start} seconds")
            
        print(f"All clustering took {end - clustering_start} seconds")
        
        clustering_result_dict = {
            "span_tree_connected_components": span_tree_connected_components_results,
            # "connected_components": connected_components_results,
            "louvain": louvain_results,
            "hierarchical": hierarchical_results,
            "kmeans": k_means_results,
            # "hdbscan": hdbscan_results,
            "affinity_propagation": affinity_propagation_results,
            "agglomerative": agglomerative_results,
            "spectral_embedding_based": spectral_embedding_based_results,
        }
        
        clustering_result_dict["clustering_runtime"] = end - clustering_start
        dimensional_reduction_start = time.perf_counter()
        
        start = time.perf_counter()
            
        isomap_results = self.get_isomap_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        end = time.perf_counter()
        clustering_result_dict["isomap_runtime"] = end - start
        if self.verbose: 
            print(f"isomap took {end - start} seconds")
            
        start = time.perf_counter()
            
        mds_embedding_results = self.get_mds_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        end = time.perf_counter()
        clustering_result_dict["mds_runtime"] = end - start
        if self.verbose: 
            print(f"mds_embedding took {end - start} seconds")
            
        # start = time.perf_counter()
            
        # tsne_embedding_results = self.get_tsne_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
        # end = time.perf_counter()
        # clustering_result_dict["tsne_runtime"] = end - start
        # if self.verbose: 
        #     print(f"tsne_embedding took {end - start} seconds")
            
        start = time.perf_counter()
            
        kernel_pca_embedding_results = self.get_kernel_pca_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
        end = time.perf_counter()
        clustering_result_dict["kernel_pca_runtime"] = end - start
        if self.verbose: 
            print(f"kernel_pca_embedding took {end - start} seconds")
            
        start = time.perf_counter()
            
        locally_linear_embedding_results = self.get_locally_linear_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
        end = time.perf_counter()
        clustering_result_dict["locally_linear_runtime"] = end - start
        if self.verbose: 
            print(f"locally_linear_embedding took {end - start} seconds")
            
        # start = time.perf_counter()
            
        # nmf_embedding_results = self.get_nmf_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
    #     end = time.perf_counter()
    #     nmf_embedding_results["nmf_runtime"] = end - start
        # if self.verbose: 
        #     print(f"nmf_embedding took {end - start} seconds")
            
    #     start = time.perf_counter()
            
        # sparse_pca_embedding_results = self.get_sparse_pca_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
    #     end = time.perf_counter()
    #     clustering_result_dict["sparse_pca_runtime"] = end - start
        # if self.verbose: 
        #     print(f"sparse_pca_embedding took {end - start} seconds")
            
    #     start = time.perf_counter()
            
        # pca_embedding_results = self.get_pca_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
    #     end = time.perf_counter()
    #     clustering_result_dict["pca_runtime"] = end - start
        # if self.verbose: 
        #     print(f"pca_embedding took {end - start} seconds")
            
    #     start = time.perf_counter()
            
        # factor_analysis_embedding_results = self.get_factor_analysis_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
    #     end = time.perf_counter()
    #     clustering_result_dict["factor_analysis_runtime"] = end - start
        # if self.verbose: 
        #     print(f"factor_analysis_embedding took {end - start} seconds")
            
        start = time.perf_counter()
            
        lda_embedding_results = self.get_lda_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
        end = time.perf_counter()
        clustering_result_dict["lda_runtime"] = end - start
        if self.verbose: 
            print(f"lda_embedding took {end - start} seconds")
            
    #     start = time.perf_counter()
            
        # dictionary_learning_embedding_results = self.get_dictionary_learning_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
    #     end = time.perf_counter()
    #     clustering_result_dict["dictionary_learning_runtime"] = end - start
        # if self.verbose: 
        #     print(f"dictionary_learning_embedding took {end - start} seconds")
            
        start = time.perf_counter()
            
        umap_embedding_results = self.get_umap_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
        end = time.perf_counter()
        clustering_result_dict["umap_runtime"] = end - start
        if self.verbose: 
            print(f"umap_embedding took {end - start} seconds")
            
        start = time.perf_counter()
            
        densmap_embedding_results = self.get_densmap_embeddings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        
        end = time.perf_counter()
        clustering_result_dict["densmap_runtime"] = end - start
        if self.verbose: 
            print(f" densmap_embedding took {end - start} seconds")
            
        print(f"Entire dimensional reduction took {end - dimensional_reduction_start} seconds ")
        clustering_result_dict['dimenson_reduction_runtime'] = end - dimensional_reduction_start
        
        low_dimensional_embedding_results = {
            "isomap_results": isomap_results,
            "mds_embedding_results": mds_embedding_results,
            # "tsne_embedding_results": tsne_embedding_results["best_tsne_results"],
            "kernel_pca_embedding_results": kernel_pca_embedding_results,
            "locally_linear_embedding_results": locally_linear_embedding_results,
            # "nmf_embedding_results": nmf_embedding_results,
            # "sparse_pca_embedding_results": sparse_pca_embedding_results,
            # "pca_embedding_results": pca_embedding_results,
            # "factor_analysis_embedding_results": factor_analysis_embedding_results,
            "lda_embedding_results": lda_embedding_results,
            # "dictionary_learning_embedding_results": dictionary_learning_embedding_results,
            "umap_embedding_results": umap_embedding_results,
            "densmap_embedding_results": densmap_embedding_results,
        }

        clustering_result_dict["mi_id"] = mi_id
        clustering_result_dict["img_id_list"] = img_id_list
        clustering_result_dict["subj_img_embedding"] = subj_img_embedding
        clustering_result_dict["subj_adjacency_matrix"] = subj_adjacency_matrix
        clustering_result_dict["max_n_clusters"] = max_n_clusters
        clustering_result_dict["min_n_clusters"] = min_n_clusters
        # clustering_result_dict["span_tree_connected_components_clusterings"] = span_tree_connected_components_results["span_tree_best_cluster_labels"]
        # clustering_result_dict["span_tree_connected_components_n_clusters"] = span_tree_connected_components_results["span_tree_best_n_clusters"]
        # clustering_result_dict["span_tree_connected_components_best_silhouette_score"] = span_tree_connected_components_results["span_tree_best_silhouette_score"]
        # clustering_result_dict["connected_components_clusterings"] = connected_components_results["connected_components_best_cluster_labels"]
        # clustering_result_dict["connected_components_best_threshold"] = connected_components_results["connected_components_best_threshold"]
        # clustering_result_dict["connected_components_best_metric"] = connected_components_results["connected_components_best_metric"]
        # clustering_result_dict["connected_components_best_silhouette_score"] = connected_components_results["connected_components_best_silhouette_score"]
        # clustering_result_dict["louvain_clusterings"] = louvain_results["louvain_best_cluster_labels"]
        # clustering_result_dict["louvain_best_silhouette_score"] = louvain_results["louvain_best_silhouette_score"]
        # clustering_result_dict["hierarchical_clusterings"] = hierarchical_results["hierarchical_best_cluster_labels"]
        # clustering_result_dict["hierarchical_best_metric"] = hierarchical_results["hierarchical_best_metric"]
        # clustering_result_dict["hierarchical_best_linkage_type"] = hierarchical_results["hierarchical_best_linkage"]
        # clustering_result_dict["hierarchical_best_silhouette_score"] = hierarchical_results["hierarchical_best_silhouette_score"]
        # clustering_result_dict["kmeans_clusterings"] = k_means_results["k_means_best_cluster_labels"]
        # clustering_result_dict["kmeans_n_clusters"] = k_means_results["k_means_best_n_clusters"]
        # clustering_result_dict["kmeans_best_silhouette_score"] = k_means_results["k_means_best_silhouette_score"]
        # clustering_result_dict["hdbscan_clusterings"] = hdbscan_results["hdbscan_best_cluster_labels"]
        # clustering_result_dict["hdbscan_best_silhouette_score"] = hdbscan_results["hdbscan_best_silhouette_score"] 
        # clustering_result_dict["hdbscan_best_cluster_selection_method"] = hdbscan_results["hdbscan_best_cluster_selection_method"]        
        # clustering_result_dict["affinity_propagation_clusterings"] = affinity_propagation_results["affinity_propagation_best_cluster_labels"]
        # clustering_result_dict["affinity_propagation_best_silhouette_score"] = affinity_propagation_results["affinity_propagation_best_silhouette_score"] 
        # clustering_result_dict["agglomerative_clusterings"] = agglomerative_results["agglomerative_best_cluster_labels"]
        # clustering_result_dict["agglomerative_best_n_clusters"] = agglomerative_results["agglomerative_best_n_clusters"] 
        # clustering_result_dict["agglomerative_best_silhouette_score"] = agglomerative_results["agglomerative_best_silhouette_score"] 
        # clustering_result_dict["spectral_embedding_based_clusterings"] = spectral_embedding_based_results["spectral_embedding_based_best_cluster_labels"]
        # clustering_result_dict["spectral_embedding_based_best_affinity_method"] = spectral_embedding_based_results["spectral_embedding_based_best_affinity_method"] 
        # clustering_result_dict["spectral_embedding_based_best_n_clusters"] = spectral_embedding_based_results["spectral_embedding_based_best_n_clusters"] 
        # clustering_result_dict["spectral_embedding_based_best_silhouette_score"] = spectral_embedding_based_results["spectral_embedding_based_best_silhouette_score"] 
        
        # low_dimensional_embedding_results["tsne_best_perplexity"] =  tsne_embedding_results["best_perplexity"]
        # low_dimensional_embedding_results["tsne_best_kl_divergence"] = tsne_embedding_results["best_kl_divergence"]
        clustering_result_dict["low_dimensional_embedding_results"] = low_dimensional_embedding_results
        clustering_result_dict = clustering_result_dict | low_dimensional_embedding_results
        
        # clustering_result_dict["mds_embeddings_results"] = mds_embedding_results
        # clustering_result_dict["isomap_embeddings_results"] = isomap_results
        # clustering_result_dict["kernel_pca_embedding_results"] = kernel_pca_embedding_results
        # clustering_result_dict["locally_linear_embedding_results"] = locally_linear_embedding_results
        # clustering_result_dict["nmf_embedding_results"] = nmf_embedding_results
        # clustering_result_dict["sparse_pca_embedding_results"] = sparse_pca_embedding_results
        # clustering_result_dict["pca_embedding_results"] = pca_embedding_results
        # clustering_result_dict["factor_analysis_embedding_results"] = factor_analysis_embedding_results
        # clustering_result_dict["lda_embedding_results"] = lda_embedding_results
        # clustering_result_dict["dictionary_learning_embedding_results"] = dictionary_learning_embedding_results
        
        clustering_result_dict = self.get_best_clustering_algorithm(clustering_result_dict) | clustering_result_dict 
        # print(clustering_result_dict)
        
        subj_result_df_dict = {}
        for k, v in clustering_result_dict.items():
            if isinstance(v, (str, int, float, bool)):
                subj_result_df_dict[k] = v
            elif isinstance(v, np.floating):
                subj_result_df_dict[k] = float(v)
            elif isinstance(v, np.integer):
                subj_result_df_dict[k] = int(v)
                
            elif isinstance(v, (np.ndarray)):
                v = v.tolist()
            elif isinstance(v, (tuple, set)):
                v = list(v)
            if isinstance(v, (list)) and len(v) == len(img_id_list):
                subj_result_df_dict[k] = v
                
            if not isinstance(v, dict):
                continue
            for sub_key, sub_key_val in v.items():
                if isinstance(sub_key_val, (list, tuple)):
                    for indx, indx_val in enumerate(sub_key_val):
                        if not isinstance(indx_val, (int, float, np.floating, np.integer, bool, str)):
                            continue
                        subj_result_df_dict[f"{k}-{sub_key}-{indx}"] = [indx_val] * len(img_id_list)
                    continue 
                if not isinstance(sub_key_val, (int, float, np.floating, np.integer, bool, str)):
                    continue 
                subj_result_df_dict[f"{k}-{sub_key}"] = [sub_key_val] * len(img_id_list)
        
        subj_result_df_dict = subj_result_df_dict | {k: [v] * len (img_id_list) 
                                                     for k, v in subj_result_df_dict.items() if isinstance(v, (int, float, np.floating, np.integer, bool, str))}
        subj_result_df = pd.DataFrame(subj_result_df_dict)
        subj_result_df['mi_id'] = [mi_id] * len(img_id_list)
        subj_result_df['img_id'] = img_id_list
        subj_result_df['best_clustering_labels'] = clustering_result_dict["best_clustering_labels"]
        subj_result_df['best_cluster_labels'] = clustering_result_dict["best_cluster_labels"]
        # subj_result_df['best_clustering_algorithm'] = clustering_result_dict["best_clustering_algorithm"]
        # subj_result_df['best_silhouette_score'] = clustering_result_dict["best_silhouette_score"]
        # subj_result_df['worst_silhouette_score'] = clustering_result_dict["worst_silhouette_score"]
        # subj_result_df['best_clustering_labels'] = clustering_result_dict["best_clustering_labels"]

        low_dimensional_embedding_df = [pd.DataFrame(v,
                                                     columns = [
                                                         f"{k.split('_embedding')[0].replace('_','').upper()}1",
                                                         f"{k.split('_embedding')[0].replace('_','').upper()}2",
                                                         ]) for k, v in low_dimensional_embedding_results.items()]

        # tsne_df = pd.DataFrame(clustering_result_dict["tsne_embeddings_results"], columns=["TSNE1", "TSNE2"])
        # isomap_df = pd.DataFrame(isomap_results, columns=["ISOMAP1", "ISOMAP2"])
        # mds_df = pd.DataFrame(mds_embedding_results, columns=["MDS1", "MDS2"])
        
        assert(all(subj_result_df.shape[0] == df.shape[0] for df in low_dimensional_embedding_df))
        # assert(mds_df.shape[0] == isomap_df.shape[0] == tsne_df.shape[0] == subj_result_df.shape[0])
        
        low_dimensional_embedding_df = pd.concat(low_dimensional_embedding_df, axis = 1)
        subj_result_df = pd.concat([subj_result_df, low_dimensional_embedding_df], axis=1)
        # subj_result_df = pd.concat([subj_result_df, tsne_df, isomap_df, mds_df], axis=1)
        
        # subj_result_df["span_tree_connected_components_clusterings"] = clustering_result_dict["span_tree_connected_components_clusterings"]
        # subj_result_df["connected_components_clusterings"] = clustering_result_dict["connected_components_clusterings"]
        # subj_result_df["louvain_clusterings"] = clustering_result_dict["louvain_clusterings"]
        # subj_result_df["hierarchical_clusterings"] = clustering_result_dict["hierarchical_clusterings"]
        # subj_result_df["spectral_clustering_graph_based_clusterings"] = clustering_result_dict["spectral_clustering_graph_based_clusterings"]
        # subj_result_df["kmeans_clusterings"] = clustering_result_dict["kmeans_clusterings"]
        # subj_result_df["hdbscan_clusterings"] = clustering_result_dict["hdbscan_clusterings"]
        # subj_result_df["affinity_propagation_clusterings"] = clustering_result_dict["affinity_propagation_clusterings"]
        # subj_result_df["agglomerative_clusterings"] = clustering_result_dict["agglomerative_clusterings"]
        # subj_result_df["spectral_embedding_based_clusterings"] = clustering_result_dict["spectral_embedding_based_clusterings"]
        
        # subj_result_df["span_tree_connected_components_n_clusters"] = [clustering_result_dict["span_tree_connected_components_n_clusters"]] * len(img_id_list)
        # subj_result_df["span_tree_connected_components_best_silhouette_score"] = [clustering_result_dict["span_tree_connected_components_best_silhouette_score"]] * len(img_id_list)
        # subj_result_df["connected_components_best_threshold"] = [clustering_result_dict["connected_components_best_threshold"]] * len(img_id_list)
        # subj_result_df["connected_components_best_metric"] = [clustering_result_dict["connected_components_best_metric"]] * len(img_id_list)
        # subj_result_df["connected_components_best_silhouette_score"] = [clustering_result_dict["connected_components_best_silhouette_score"]] * len(img_id_list)
        # subj_result_df["louvain_best_silhouette_score"] = [clustering_result_dict["louvain_best_silhouette_score"]] * len(img_id_list)
        # subj_result_df["hierarchical_best_metric"] = [clustering_result_dict["hierarchical_best_metric"]] * len(img_id_list)
        # subj_result_df["hierarchical_best_linkage_type"] = [clustering_result_dict["hierarchical_best_linkage_type"]] * len(img_id_list)
        # subj_result_df["hierarchical_best_silhouette_score"] = [clustering_result_dict["hierarchical_best_silhouette_score"]] * len(img_id_list)
        # subj_result_df["spectral_clustering_graph_based_n_clusters"] = [clustering_result_dict["spectral_clustering_graph_based_n_clusters"]] * len(img_id_list)
        # subj_result_df["spectral_clustering_graph_based_best_silhouette_score"] = [clustering_result_dict["spectral_clustering_graph_based_best_silhouette_score"]] * len(img_id_list)
        # subj_result_df["kmeans_n_clusters"] = [clustering_result_dict["kmeans_n_clusters"]] * len(img_id_list)
        # subj_result_df["kmeans_best_silhouette_score"] = [clustering_result_dict["kmeans_best_silhouette_score"]] * len(img_id_list)
        # subj_result_df["hdbscan_best_silhouette_score"] = [clustering_result_dict["hdbscan_best_silhouette_score"]] * len(img_id_list)
        # subj_result_df["hdbscan_best_cluster_selection_method"] = [clustering_result_dict["hdbscan_best_cluster_selection_method"]] * len(img_id_list)
        # subj_result_df["affinity_propagation_best_silhouette_score"] = [clustering_result_dict["affinity_propagation_best_silhouette_score"]] * len(img_id_list)
        # subj_result_df["agglomerative_best_n_clusters"] = [clustering_result_dict["agglomerative_best_n_clusters"]] * len(img_id_list)
        # subj_result_df["agglomerative_best_silhouette_score"] = [clustering_result_dict["agglomerative_best_silhouette_score"]] * len(img_id_list)
        # subj_result_df["spectral_embedding_based_best_affinity_method"] = [clustering_result_dict["spectral_embedding_based_best_affinity_method"]] * len(img_id_list)
        # subj_result_df["spectral_embedding_based_best_n_clusters"] = [clustering_result_dict["spectral_embedding_based_best_n_clusters"]] * len(img_id_list)
        # subj_result_df["spectral_embedding_based_best_silhouette_score"] = [clustering_result_dict["spectral_embedding_based_best_silhouette_score"]] * len(img_id_list)
        
        # subj_result_df["tsne_best_perplexity"] =  [clustering_result_dict["tsne_best_perplexity"]] * len(img_id_list)
        # subj_result_df["tsne_best_kl_divergence"] = [clustering_result_dict["tsne_best_kl_divergence"]] * len(img_id_list)
        
        clustering_result_dict["subj_result_df"] = subj_result_df
        if len(subj_result_df.columns) != len(set(subj_result_df.columns)):
            print(subj_result_df.columns)
        
        subj_result_df.set_index("img_id")
        
        return clustering_result_dict
        
        
    def convert_sklearn_labels_to_communities(self, labels: np.ndarray):
        # Determine the number of unique communities
        num_communities = len(np.unique(labels))

        # Initialize a list of empty sets, one for each community
        communities = [set() for _ in range(num_communities)]

        # Populate the communities
        for indx, group in enumerate(labels):
            communities[group].add(indx)

        return communities
    
    def convert_communities_output_to_labels(self, communities):
        labels_pred = [None for community in communities for _ in community]
        for community_indx, community in enumerate(communities):
            for node in community:
                labels_pred[node] = community_indx
        return labels_pred
    
    def get_span_tree_connected_components_clusterings(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_cluster_labels = None
        best_n_clusters = None
        best_silhouette_score = -100
        n_clusters = [i + 1 for i in range(min_n_clusters, max_n_clusters)]
        clusterings = [SpanTreeConnectedComponentsClustering(
            n_clusters=i,
            metric="euclidean",
            n_jobs=-1,
                ) for i in n_clusters]
        
        adj_matrix_labels_pred_scores = [0] * len(clusterings)
        img_embedding_labels_pred_scores = [0] * len(clusterings)
        
        
        for indx, clustering in enumerate(clusterings):
            labels_pred = clustering.fit_predict(adj_matrix)
            labels_pred_score = silhouette_score(img_embedding, labels_pred)
            adj_matrix_labels_pred_scores[indx] = labels_pred_score
            
            if labels_pred_score > best_silhouette_score:
                best_silhouette_score = labels_pred_score
                best_n_clusters = n_clusters[indx] 
                best_cluster_labels = labels_pred
                
            labels_pred = clustering.fit_predict(img_embedding)
            labels_pred_score = silhouette_score(img_embedding, labels_pred)
            img_embedding_labels_pred_scores[indx] = labels_pred_score
            
            if labels_pred_score > best_silhouette_score:
                best_silhouette_score = labels_pred_score
                best_n_clusters = n_clusters[indx] 
                best_cluster_labels = labels_pred
            
        return {
            "best_n_clusters": best_n_clusters, 
            "best_cluster_labels": best_cluster_labels, 
            "best_silhouette_score": best_silhouette_score,
            "n_clusters": n_clusters,
            "silhouette_scores": adj_matrix_labels_pred_scores + img_embedding_labels_pred_scores,
            "clusterings": clusterings,
            "n_clusters": n_clusters + n_clusters, 
            }
    
    def get_connected_components_clusterings(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int = 1000):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
            
        best_cluster_labels = None
        best_threshold = None
        best_metric = None
        best_silhouette_score = -100
        thresholds = np.linspace(0.15, 0.97, num = int(max_iter))
        metrics = ["euclidean", "cosine"]
        
        clusterings = [ConnectedComponentsClustering(
            threshold=threshold,
            metric=metric,
            n_jobs=-1,)
                                 for threshold in thresholds 
                                 for metric in metrics]
        adj_matrix_labels_pred_scores = [0] * len(clusterings)
        img_embedding_labels_pred_scores = [0] * len(clusterings)
        adj_matrix_n_clusters = [None] * len(clusterings)
        img_embedding_n_clusters = [None] * len(clusterings)
        
        for indx, clustering in enumerate(clusterings):
            labels_pred = clustering.fit_predict(adj_matrix)
            num_clusters = len(set(labels_pred))
            adj_matrix_n_clusters[indx] = num_clusters
            
            if min_n_clusters <=num_clusters <= max_n_clusters: 
                labels_pred_score = silhouette_score(img_embedding, labels_pred)
                adj_matrix_labels_pred_scores[indx] = labels_pred_score
                
                if labels_pred_score > best_silhouette_score:
                    best_silhouette_score = labels_pred_score
                    best_cluster_labels = labels_pred
                    best_metric = clustering.metric
                    best_threshold = clustering.threshold
                
            labels_pred = clustering.fit_predict(img_embedding)
            num_clusters = len(set(labels_pred))
            img_embedding_n_clusters[indx] = num_clusters
            
            if min_n_clusters <=num_clusters <= max_n_clusters: 
                labels_pred_score = silhouette_score(img_embedding, labels_pred)
                img_embedding_labels_pred_scores[indx] = labels_pred_score
            
                if labels_pred_score > best_silhouette_score:
                    best_silhouette_score = labels_pred_score
                    best_cluster_labels = labels_pred
                    best_metric = clustering.metric
                    best_threshold = clustering.threshold

        return {
            "best_threshold": best_threshold, 
            "best_cluster_labels": best_cluster_labels, 
            "best_silhouette_score": best_silhouette_score,
            "best_metric": best_metric,
            "clusterings": clusterings, 
            "thresholds": thresholds + thresholds,
            "silhouette_scores": adj_matrix_labels_pred_scores + img_embedding_labels_pred_scores,
            "n_clusters": adj_matrix_n_clusters + img_embedding_n_clusters,
            }    
    
    def get_louvain_clusterings(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        communities, _ = louvain_method(adj_matrix=adj_matrix)
        labels_pred = self.convert_communities_output_to_labels(communities)
        
        if min_n_clusters <=len(set(labels_pred)) <= max_n_clusters: 
            best_silhouette_score = silhouette_score(img_embedding, labels_pred)
        else:
            best_silhouette_score = None 
        
        return {
            "best_cluster_labels": labels_pred,
            "best_silhouette_score": best_silhouette_score,
            "communities": communities, 
            "n_clusters": [len(set(labels_pred))],
            "silhouette_scores": [best_silhouette_score],
        }
        
    def get_hierarchical_clusterings(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        metrics = ['euclidean', 'cosine']
        linkages = ['single', 'mean', 'complete']
        best_silhouette_score = -100
        best_linkage = None
        best_metric = None
        best_cluster_labels = None
        params = [{"linkage": linkage, "metric": metric, } for linkage in linkages for metric in metrics]
        clustering_communities = [hierarchical_clustering(adj_matrix=adj_matrix, metric=i["metric"], linkage=i["linkage"])
                                  for i in params]
        clustering_pred = [self.convert_communities_output_to_labels(i) for i in clustering_communities]
        hierarchical_silhouette_scores = [0] * len(clustering_communities)
        n_clusters = [0] * len(clustering_communities)
        
        for indx, labels_pred in enumerate(clustering_pred):
            i_n_clusters = len(set(labels_pred))
            n_clusters[indx] = i_n_clusters
            
            if not (min_n_clusters <= i_n_clusters <= max_n_clusters):
                continue
                
            labels_pred_score = silhouette_score(img_embedding, labels_pred)
            hierarchical_silhouette_scores[indx] = labels_pred_score
            
            if labels_pred_score <= best_silhouette_score: 
                continue
            
            best_silhouette_score = labels_pred_score
            best_cluster_labels = labels_pred
            best_metric = params[indx]["metric"]
            best_linkage = params[indx]["linkage"]

        return {
            "best_cluster_labels": best_cluster_labels,
            "best_silhouette_score": best_silhouette_score,
            "best_linkage": best_linkage,
            "best_metric": best_metric,
            "communities": clustering_communities,
            "silhouette_scores": hierarchical_silhouette_scores,
            "n_clusters": n_clusters,
        }
    
    def get_k_means_clusterings(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_n_clusters = None
        best_silhouette_score = -100
        best_cluster_labels = None
        num_clusters = [i+ 1 for i in range(min_n_clusters, max_n_clusters)]
        labels_predictions = [KMeans(n_clusters=n,  max_iter=max_iter).fit_predict(img_embedding) 
                              for n in num_clusters]
        k_means_silhouette_scores = [silhouette_score(img_embedding, labels_pred) for labels_pred in labels_predictions]
        
        for indx, labels_pred_score in enumerate(k_means_silhouette_scores):
            if labels_pred_score <= best_silhouette_score:
                continue
            
            best_silhouette_score = labels_pred_score
            best_cluster_labels = labels_predictions[indx]
            best_n_clusters = num_clusters[indx]
        
        return {
            "best_cluster_labels": best_cluster_labels,
            "best_silhouette_score": best_silhouette_score,
            "best_n_clusters": best_n_clusters,
            "silhouette_scores": k_means_silhouette_scores,
            "n_clusters": num_clusters, 
        }
   
    # def get_hdbscan_clusterings(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
    #     if self.verbose:
    #         print(inspect.currentframe().f_code.co_name)
    #     best_silhouette_score = -100
    #     best_cluster_labels = None
    #     best_n_clusters = None
    #     min_cluster_size = int(len(img_embedding)/ max_n_clusters)
    #     min_cluster_size = max(min_cluster_size, 2)
    #     max_cluster_size = int(len(img_embedding)/ min_n_clusters)
    #     cluster_selection_methods = ["eom", "leaf"]
    #     best_cluster_selection_method = None
    #     label_predictions = [HDBSCAN(min_cluster_size=min_cluster_size,
    #                                  max_cluster_size=max_cluster_size,
    #                                  cluster_selection_method=cluster_selection_method,
    #                                  n_jobs = -1).fit_predict(img_embedding) 
    #                          for cluster_selection_method in cluster_selection_methods]
    #     hdbscan_silhouette_scores = [0] * len(label_predictions)
    #     num_clusters = [0] * len(label_predictions)
        
        # for indx, labels_pred in enumerate(label_predictions):
            
        #     indx_num_clusters = len(set(labels_pred))
        #     num_clusters[indx] = indx_num_clusters
        #     if not (min_n_clusters <= len(set(labels_pred)) <= max_n_clusters): 
        #         continue
            
        #     labels_pred_score = silhouette_score(img_embedding, labels_pred)
        #     hdbscan_silhouette_scores[indx] = labels_pred_score

        #     if labels_pred_score <= best_silhouette_score:
        #         continue
            
        #     best_silhouette_score = labels_pred_score
        #     best_cluster_labels = labels_pred
        #     best_cluster_selection_method = cluster_selection_methods[indx]
        #     best_n_clusters = indx_num_clusters

        # return {
        #     "best_cluster_labels": best_cluster_labels,
        #     "best_silhouette_score": best_silhouette_score,
        #     "best_n_clusters": best_n_clusters,
        #     "best_cluster_selection_method": best_cluster_selection_method,
        #     "silhouette_scores": hdbscan_silhouette_scores,
        #     "n_clusters": num_clusters,
        # }
             
             
    def get_affinity_propagation_clusterings(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_silhouette_score = -100
        best_cluster_labels = None
        best_n_clusters = None
        
        
        labels_pred = AffinityPropagation( max_iter=max_iter).fit_predict(img_embedding)
        n_clusters = len(set(labels_pred))

        try: 
            if min_n_clusters <= n_clusters <= max_n_clusters:
                labels_pred_score = silhouette_score(img_embedding, labels_pred)
                best_silhouette_score = labels_pred_score
                best_cluster_labels = labels_pred
                best_n_clusters = n_clusters
        except ValueError as e:
                print("affinity proagation error: ",e)

        return {
            "best_cluster_labels": best_cluster_labels,
            "best_silhouette_score": best_silhouette_score,
            "best_n_clusters": best_n_clusters,
            "n_clusters": [n_clusters],
            "silhouette_scores": [best_silhouette_score]
        }
     
    def get_agglomerative_clusterings(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_silhouette_score = -100
        best_cluster_labels = None
        best_n_clusters = None
        best_linkage = None
        
        num_clusters = [i + 1 for i in range(min_n_clusters, max_n_clusters)]
        linkages = ["ward", "complete", "average", "single"]
        params = [{"n": n, "linkage": linkage} for n in num_clusters for linkage in linkages]
        del num_clusters
        
        num_clusters = [i['n'] for i in params]
        label_predictions = [AgglomerativeClustering(n_clusters=i["n"],
                                                     linkage=i["linkage"]).fit_predict(img_embedding) for i in params]
        agglomerative_silhouette_scores = [0] * len(label_predictions)
        
        for indx, labels_pred in enumerate(label_predictions):
            if not (min_n_clusters <= len(set(labels_pred)) <= max_n_clusters):
                continue
            
            labels_pred_score = silhouette_score(img_embedding, labels_pred)
            agglomerative_silhouette_scores[indx] = labels_pred_score
            if labels_pred_score > best_silhouette_score:
                best_silhouette_score = labels_pred_score
                best_cluster_labels = labels_pred
                best_n_clusters = num_clusters[indx]
                best_linkage = params[indx]["linkage"]

        return {
            "best_cluster_labels": best_cluster_labels,
            "best_silhouette_score": best_silhouette_score,
            "best_n_clusters": best_n_clusters,
            "best_linkage": best_linkage,
            "silhouette_scores": agglomerative_silhouette_scores,
            "n_clusters": num_clusters,
        }
    def get_spectral_embedding_based_clusterings(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_silhouette_score = -100
        best_cluster_labels = None
        best_n_clusters = None
        best_affinity_method = None
        affinity_methods = ["nearest_neighbors", "rbf"]
        num_clusters = [i + 1 for i in range(min_n_clusters, max_n_clusters)]
        params = [{"n": n, "affinity_method": affinity_method} for n in num_clusters for affinity_method in affinity_methods]
        del num_clusters
        num_clusters = [i["n"] for i in params]
        label_predictions = [
            SpectralClustering(n_clusters=i["n"],
                               n_init=10,
                               n_jobs=-1,
                               n_neighbors=5,
                               affinity = i["affinity_method"],
                               random_state=0).fit_predict(img_embedding) for i in params]
        spectral_clustering_silhouette_scores = [0] * len(label_predictions)
        
        for indx, labels_pred in enumerate(label_predictions):
            if not (min_n_clusters <= len(set(labels_pred)) <= max_n_clusters):
                continue
            
            labels_pred_score = silhouette_score(img_embedding, labels_pred)
            spectral_clustering_silhouette_scores[indx] = labels_pred_score
            if labels_pred_score <= best_silhouette_score:
                continue
            
            best_silhouette_score = labels_pred_score
            best_cluster_labels = labels_pred
            best_params = params[indx]
            best_n_clusters = num_clusters[indx]
            best_affinity_method = best_params["affinity_method"]
        
        return {
            "best_cluster_labels": best_cluster_labels,
            "best_silhouette_score": best_silhouette_score,
            "best_n_clusters": best_n_clusters,
            "best_affinity_method": best_affinity_method,
            "n_clusters": num_clusters,
            "silhouette_scores": spectral_clustering_silhouette_scores,
        }
        
    def get_span_tree_connected_components_clusterings2(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_cluster_labels = None
        best_n_clusters = None
        best_silhouette_score = -100
        for i in range(min_n_clusters, max_n_clusters):
            i = i + 1
            clustering = SpanTreeConnectedComponentsClustering(
                    n_clusters=i,
                    metric="euclidean",
                    n_jobs=-1,
                )
            
            labels_pred = clustering.fit_predict(adj_matrix)
            labels_pred_score = silhouette_score(img_embedding, labels_pred)
            
            if labels_pred_score > best_silhouette_score:
                best_silhouette_score = labels_pred_score
                best_n_clusters = i 
                best_cluster_labels = labels_pred
                
            labels_pred = clustering.fit_predict(img_embedding)
            labels_pred_score = silhouette_score(img_embedding, labels_pred)
            
            if labels_pred_score > best_silhouette_score:
                best_silhouette_score = labels_pred_score
                best_n_clusters = i 
                best_cluster_labels = labels_pred
            
            
        return {
            "span_tree_best_n_clusters": best_n_clusters, 
            "span_tree_best_cluster_labels": best_cluster_labels, 
            "span_tree_best_silhouette_score": best_silhouette_score,
            }
        
    def get_connected_components_clusterings2(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int = 1000):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_cluster_labels = None
        best_threshold = None
        best_metric = None
        best_silhouette_score = -100
        thresholds = np.linspace(0.15, 0.97, num = max_iter)
        metrics = ["euclidean", "cosine"]
        
        clusterings = [ConnectedComponentsClustering(
            threshold=threshold,
            metric=metric,
            n_jobs=-1,)
                                 for threshold in thresholds 
                                 for metric in metrics]
        
        for _, clustering in enumerate(clusterings):
            labels_pred = clustering.fit_predict(adj_matrix)
            
            if min_n_clusters <= len(set(labels_pred)) <= max_n_clusters:
                labels_pred_score = silhouette_score(img_embedding, labels_pred)
                if labels_pred_score > best_silhouette_score:
                    best_silhouette_score = labels_pred_score
                    best_cluster_labels = labels_pred
                    best_metric = clustering.metric
                    best_threshold = clustering.threshold
                
            labels_pred = clustering.fit_predict(img_embedding)
            
            if min_n_clusters <= len(set(labels_pred)) <= max_n_clusters:
                labels_pred_score = silhouette_score(img_embedding, labels_pred)
                if labels_pred_score > best_silhouette_score:
                    best_silhouette_score = labels_pred_score
                    best_cluster_labels = labels_pred
                    best_metric = clustering.metric
                    best_threshold = clustering.threshold
        
        # for threshold in thresholds:
        #     metric = "euclidean"
        #     clustering = ConnectedComponentsClustering(
        #         threshold=threshold,
        #         metric=metric,
        #         n_jobs=-1,
        #     )
            
        #     labels_pred = clustering.fit_predict(adj_matrix)
            
        #     if min_n_clusters <=len(set(labels_pred)) <= max_n_clusters and labels_pred_score > best_silhouette_score:
        #         labels_pred_score = silhouette_score(img_embedding, labels_pred)
        #         best_silhouette_score = labels_pred_score
        #         best_cluster_labels = labels_pred
        #         best_metric = metric
        #         best_threshold = threshold
                
        #     labels_pred = clustering.fit_predict(img_embedding)
            
        #     if min_n_clusters <=len(set(labels_pred)) <= max_n_clusters and labels_pred_score > best_silhouette_score:
        #         labels_pred_score = silhouette_score(img_embedding, labels_pred)
        #         best_silhouette_score = labels_pred_score
        #         best_cluster_labels = labels_pred
        #         best_metric = metric
        #         best_threshold = threshold
                
        #     metric = "cosine"
        #     clustering = ConnectedComponentsClustering(
        #         threshold=threshold,
        #         metric=metric,
        #         n_jobs=-1,
        #     )
            
        #     labels_pred = clustering.fit_predict(adj_matrix)
            
        #     if min_n_clusters <=len(set(labels_pred)) <= max_n_clusters and labels_pred_score > best_silhouette_score:
        #         labels_pred_score = silhouette_score(img_embedding, labels_pred)
        #         best_silhouette_score = labels_pred_score
        #         best_cluster_labels = labels_pred
        #         best_metric = metric
        #         best_threshold = threshold
                
        #     labels_pred = clustering.fit_predict(img_embedding)
            
        #     if min_n_clusters <=len(set(labels_pred)) <= max_n_clusters and labels_pred_score > best_silhouette_score:
        #         labels_pred_score = silhouette_score(img_embedding, labels_pred)
        #         best_silhouette_score = labels_pred_score
        #         best_cluster_labels = labels_pred
        #         best_metric = metric
        #         best_threshold = threshold
            
            
        return {
            "connected_components_best_threshold": best_threshold, 
            "connected_components_best_cluster_labels": best_cluster_labels, 
            "connected_components_best_silhouette_score": best_silhouette_score,
            "connected_components_best_metric": best_metric,
            }
    
    def get_louvain_clusterings2(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int):
        communities, _ = louvain_method(adj_matrix=adj_matrix)
        labels_pred = self.convert_communities_output_to_labels(communities)
        
        if min_n_clusters <=len(set(labels_pred)) <= max_n_clusters: 
            best_silhouette_score = silhouette_score(img_embedding, labels_pred)
        else:
            best_silhouette_score = None 
        
        return {
            "louvain_best_cluster_labels": labels_pred,
            "louvain_best_silhouette_score": best_silhouette_score,
        }
        
    def get_hierarchical_clusterings2(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        metrics = ['euclidean', 'cosine']
        linkages = ['single', 'mean', 'complete']
        best_silhouette_score = -100
        best_linkage = None
        best_metric = None
        best_cluster_labels = None
        for linkage in linkages:
            for metric in metrics: 
                communities = hierarchical_clustering(adj_matrix=adj_matrix, metric=metric, linkage=linkage)
                labels_pred = self.convert_communities_output_to_labels(communities)
                labels_pred_score = silhouette_score(img_embedding, labels_pred)
                
                if labels_pred_score > best_silhouette_score: 
                    best_metric = metric
                    best_linkage = linkage
                    best_cluster_labels = labels_pred
                else:
                    best_silhouette_score = None 
        
        return {
            "hierarchical_best_cluster_labels": best_cluster_labels,
            "hierarchical_best_silhouette_score": best_silhouette_score,
            "hierarchical_best_linkage": best_linkage,
            "hierarchical_best_metric": best_metric,
        }
        
    def get_k_means_clusterings2(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_n_clusters = None
        best_silhouette_score = -100
        best_cluster_labels = None
        for i in range(min_n_clusters, max_n_clusters): 
            i = i + 1
            labels_pred = KMeans(n_clusters=i,  max_iter=max_iter).fit_predict(img_embedding)
            labels_pred_score = silhouette_score(img_embedding, labels_pred)
            
            if labels_pred_score > best_silhouette_score:
                best_silhouette_score = labels_pred_score
                best_cluster_labels = labels_pred
                best_n_clusters = i

        return {
            "k_means_best_cluster_labels": best_cluster_labels,
            "k_means_best_silhouette_score": best_silhouette_score,
            "k_means_best_n_clusters": best_n_clusters,
        }
        
    # def get_hdbscan_clusterings2(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
    #     if self.verbose:
    #         print(inspect.currentframe().f_code.co_name)
    #     best_silhouette_score = -100
    #     best_cluster_labels = None
    #     min_cluster_size = int(len(img_embedding)/ max_n_clusters)
    #     min_cluster_size = max(min_cluster_size, 2)
    #     max_cluster_size = int(len(img_embedding)/ min_n_clusters)
    #     best_cluster_selection_method = None
        
    #     for cluster_selection_method in ["eom", "leaf"]:
        
    #         labels_pred = HDBSCAN(min_cluster_size=min_cluster_size,
    #                                 max_cluster_size=max_cluster_size,
    #                                 cluster_selection_method=cluster_selection_method,
    #                                 n_jobs = -1).fit_predict(img_embedding)
    #         try: 
    #             labels_pred_score = silhouette_score(img_embedding, labels_pred)

    #             if min_n_clusters <=len(set(labels_pred)) <= max_n_clusters and labels_pred_score > best_silhouette_score:
    #                 best_silhouette_score = labels_pred_score
    #                 best_cluster_labels = labels_pred
    #                 best_cluster_selection_method = cluster_selection_method
    #         except ValueError as e:
    #             print(cluster_selection_method, e)

    #     return {
    #         "hdbscan_best_cluster_labels": best_cluster_labels,
    #         "hdbscan_best_silhouette_score": best_silhouette_score,
    #         "hdbscan_best_cluster_selection_method": best_cluster_selection_method,
    #     }
        
    def get_affinity_propagation_clusterings2(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_silhouette_score = -100
        best_cluster_labels = None
        
        
        labels_pred = AffinityPropagation( max_iter=max_iter).fit_predict(img_embedding)
        try: 
            labels_pred_score = silhouette_score(img_embedding, labels_pred)

            if min_n_clusters <=len(set(labels_pred)) <= max_n_clusters:
                best_silhouette_score = labels_pred_score
                best_cluster_labels = labels_pred
        except ValueError as e:
                print("affinity proagation error: ",e)

        return {
            "affinity_propagation_best_cluster_labels": best_cluster_labels,
            "affinity_propagation_best_silhouette_score": best_silhouette_score,
        }
        
    def get_agglomerative_clusterings2(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_silhouette_score = -100
        best_cluster_labels = None
        best_n_clusters = None
        
        for i in range(min_n_clusters, max_n_clusters):
            i = i + 1
        
            labels_pred = AgglomerativeClustering(n_clusters=i,).fit_predict(img_embedding)
            labels_pred_score = silhouette_score(img_embedding, labels_pred)

            if labels_pred_score > best_silhouette_score:
                best_silhouette_score = labels_pred_score
                best_cluster_labels = labels_pred
                best_n_clusters = i

        return {
            "agglomerative_best_cluster_labels": best_cluster_labels,
            "agglomerative_best_silhouette_score": best_silhouette_score,
            "agglomerative_best_n_clusters": best_n_clusters,
        }
        
    def get_spectral_embedding_based_clusterings2(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        best_silhouette_score = -100
        best_cluster_labels = None
        best_n_clusters = None
        best_affinity_method = None
        affinity_methods = ["nearest_neighbors", "rbf"]
        
        for i in range(min_n_clusters, max_n_clusters):
            i = i + 1
            
            for affinity_method in affinity_methods:
        
                labels_pred = SpectralClustering(n_clusters=i,
                                                n_jobs=-1,
                                                affinity = affinity_method,
                                                random_state=0).fit_predict(img_embedding)
                labels_pred_score = silhouette_score(img_embedding, labels_pred)

                if labels_pred_score > best_silhouette_score:
                    best_silhouette_score = labels_pred_score
                    best_cluster_labels = labels_pred
                    best_n_clusters = i
                    best_affinity_method = affinity_method

        return {
            "spectral_embedding_based_best_cluster_labels": best_cluster_labels,
            "spectral_embedding_based_best_silhouette_score": best_silhouette_score,
            "spectral_embedding_based_best_n_clusters": best_n_clusters,
            "spectral_embedding_based_best_affinity_method": best_affinity_method,
        }
        
    # def get_nca_embeddings(self, img_embedding: np.ndarray, max_iter: int):
    #     try: 
    #         result = NeighborhoodComponentsAnalysis(
    #             n_components=2,
    #             init = "pca",
    #             max_iter = max_iter,
    #             ).fit_transform(img_embedding)
    #         return result
    #     except Exception as e:
    #         print(e)
    #         return np.zeros((img_embedding.shape[0], 2))
        
    def get_mds_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return MDS(
            n_components=2,
            # n_jobs = -1,
            max_iter = max_iter,
        ).fit_transform(img_embedding)
        
    def get_isomap_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return Isomap(
            n_components=2,
            n_jobs = -1,
            max_iter = max_iter,
        ).fit_transform(img_embedding)
        
    def get_kernel_pca_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return KernelPCA(n_components=2, 
                         kernel = "rbf",
                         max_iter=max_iter,
                         n_jobs=-1,).fit_transform(img_embedding)
        
    def get_locally_linear_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return LocallyLinearEmbedding(n_components=2, 
                         method = "modified",
                         n_jobs=-1,).fit_transform(img_embedding)
    def get_nmf_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return NMF(n_components=2,).fit_transform(img_embedding)
    
    def get_sparse_pca_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return SparsePCA(n_components=2,
                         max_iter=max_iter,
                         n_jobs=-1).fit_transform(img_embedding)
    
    def get_pca_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return PCA(n_components=2,).fit_transform(img_embedding)
    
    def get_factor_analysis_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return FactorAnalysis(n_components=2,).fit_transform(img_embedding)
    
    def get_lda_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return LatentDirichletAllocation(n_components=2,).fit_transform(img_embedding)
    
    def get_dictionary_learning_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return DictionaryLearning(n_components=2,
                                  n_jobs=-1).fit_transform(img_embedding)
        
    def get_umap_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return umap.UMAP(densmap=False).fit_transform(img_embedding)
    
    def get_densmap_embeddings(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
        return umap.UMAP(densmap=True).fit_transform(img_embedding)
        
    def get_tsne_embeddings2(self, img_embedding: np.ndarray, max_iter: int):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
            
            
        min_perplexity = min(5, img_embedding.shape[0])
        max_perplexity = min(50, img_embedding.shape[0])
        best_tsne_results = None
        best_perplexity = None
        best_kl_divergence = 1000
        
        for perplexity in range(min_perplexity, max_perplexity):
        
            _tsne = TSNE(
                n_components=2,
                n_jobs = -1,
                learning_rate = 'auto',
                init='pca',
                n_iter_without_progress=500,
                perplexity=perplexity,
            )
            tsne_results = _tsne.fit_transform(img_embedding)
            if _tsne.kl_divergence_ < best_kl_divergence:
                best_kl_divergence = _tsne.kl_divergence_
                best_perplexity = perplexity
                best_tsne_results = tsne_results
        
        return {
            "best_tsne_results": best_tsne_results, 
            "best_perplexity": best_perplexity, 
            "best_kl_divergence": best_kl_divergence,
            }
        
    def get_tsne_embeddings(self, img_embedding: np.ndarray, max_iter: int, cuda_tsne = False, sklearn_tsne_bool = True):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)
            
            
        if cuda_tsne and torch.cuda.is_available():
            perplexity = 17
            sample_size = min(img_embedding.shape)
            early_exaggeration = 12 
            learning_rate = max(sample_size/ early_exaggeration / 4, 50)
            _tsne = TSNE_GPU(
                n_components=2,
                learning_rate = learning_rate,
                n_iter_without_progress=1000,  
                perplexity=perplexity,
                )
            
            tsne_result = _tsne.fit_transform(img_embedding)
        
            return {
                "best_tsne_results": tsne_result, 
                "best_perplexity": 17, 
                # "best_kl_divergence": _tsne.kl_divergence_,
                }
           
        if not sklearn_tsne_bool:  
            _tsne = open_TSNE(
                    n_components=2,
                    n_jobs = mp.cpu_count() -1,
                    learning_rate = 'auto',
                    initialization='pca',
                    perplexity=17,
                    verbose = True,
                    # n_iter=50,
                    n_iter=max_iter,
                    )
            tsne_fit = _tsne.fit(img_embedding)
            tsne_result = tsne_fit.transform(img_embedding)
            
            return {
                "best_tsne_results": tsne_result, 
                "best_perplexity": 17, 
                # "best_kl_divergence": _tsne.kl_divergence_,
            }            

        ######## SKLEARN
        # _tsne = TSNE(
        #         n_components=2,
        #         n_jobs = -1,
        #         learning_rate = 'auto',
        #         init='pca',
        #         n_iter_without_progress=300,
        #         perplexity=17,
        #         verbose = 2,
        #         n_iter=max_iter,
        #         )
        # tsne_result = _tsne.fit_transform(img_embedding)
        
        # return {
        #     "best_tsne_results": tsne_result, 
        #     "best_perplexity": 17, 
        #     "best_kl_divergence": _tsne.kl_divergence_,
        # }
        
        min_perplexity = min(5, img_embedding.shape[0])
        max_perplexity = min(50, img_embedding.shape[0])
        best_tsne_results = None
        best_perplexity = None
        best_kl_divergence = 1000
        perplexities = [perplexity for perplexity in range(min_perplexity, max_perplexity, 2)]
        if 17 not in perplexities:
            perplexities.append(17)
        
        for perplexity in tqdm(perplexities):
        
            _tsne = TSNE(
                n_components=2,
                n_jobs = -1,
                learning_rate = 'auto',
                init='pca',
                n_iter_without_progress=500,
                perplexity=perplexity,
            )
            tsne_results = _tsne.fit_transform(img_embedding)
            if _tsne.kl_divergence_ < best_kl_divergence:
                best_kl_divergence = _tsne.kl_divergence_
                best_perplexity = perplexity
                best_tsne_results = tsne_results
        
        return {
            "best_tsne_results": best_tsne_results, 
            "best_perplexity": best_perplexity, 
            "best_kl_divergence": best_kl_divergence,
            }
            
        min_perplexity = min(5, img_embedding.shape[0])
        max_perplexity = min(50, img_embedding.shape[0])
        best_tsne_results = None
        best_perplexity = None
        best_kl_divergence = 1000
        perplexities = [perplexity for perplexity in range(min_perplexity, max_perplexity, 5)]
        if 17 not in perplexities:
            perplexities.append(17)
        # perplexities = perplexities[:10]
            
        _tsnes = [
            TSNE(
                n_components=2,
                init='pca',
                learning_rate = 'auto',
                n_jobs = -1,
                n_iter_without_progress=300,
                perplexity=perplexity,
                ) for perplexity in perplexities
            ]

        tsne_results = [_tsne.fit_transform(img_embedding) for _tsne in tqdm(_tsnes)]
        
        for _tsne, tsne_result, perplexity in zip(_tsnes, tsne_results, perplexities):
            if _tsne.kl_divergence_ < best_kl_divergence:
                    best_kl_divergence = _tsne.kl_divergence_
                    best_perplexity = perplexity
                    best_tsne_results = tsne_result
        
        return {
            "best_tsne_results": best_tsne_results, 
            "best_perplexity": best_perplexity, 
            "best_kl_divergence": best_kl_divergence,
            }
        
    def get_best_clustering_algorithm(self, clustering_result_dict: dict):
        if self.verbose:
            print(inspect.currentframe().f_code.co_name)

        # clustering_results = [[k, k["best_silhouette_score"]] for k in clustering_result_dict]
        clustering_algo_dict = {k: v for k,v  in clustering_result_dict.items() 
                                if isinstance(v, dict)}
        
        clustering_algo_dict = {k: v.get("best_silhouette_score") for k,v  in clustering_algo_dict.items() 
                                if v.get("best_silhouette_score", None) is not None}
        
        all_algos, all_silhouette_scores = clustering_algo_dict.keys(), clustering_algo_dict.values()            
        
        assert(len(all_algos) == len(all_silhouette_scores))

        clustering_algo_dict = {key: val for key, val in clustering_algo_dict.items() 
                                if isinstance(val, (int, float, np.floating, np.integer)) }
        
        try: 
            best_clustering_algorithm = max(clustering_algo_dict, key=clustering_algo_dict.get)
        except Exception as e:
            print(e)
            print(clustering_algo_dict)
            raise Exception(e)
        
        best_silhouette_score = clustering_algo_dict.get(best_clustering_algorithm)
        print(f"Best clustering algorithm {best_clustering_algorithm}, best silhouette score {best_silhouette_score}")
        best_clustering_labels = clustering_result_dict.get(best_clustering_algorithm).get("best_cluster_labels")
        
        worst_clustering_algorithm = min(clustering_algo_dict, key=clustering_algo_dict.get)
        worst_silhouette_score = clustering_algo_dict.get(worst_clustering_algorithm)
        
        return {
            "best_clustering_algorithm": best_clustering_algorithm,
            "best_silhouette_score": best_silhouette_score,
            "worst_clustering_algorithm": worst_clustering_algorithm,
            "worst_silhouette_score": worst_silhouette_score,
            "best_clustering_labels": best_clustering_labels,
            "best_cluster_labels": best_clustering_labels,
        }
    
    def get_best_clustering_algorithm2(self, clustering_result_dict: dict):
        all_silhouette_scores = [
            clustering_result_dict.get("span_tree_connected_components_best_silhouette_score"), 
            clustering_result_dict.get("connected_components_best_silhouette_score"), 
            clustering_result_dict.get("louvain_best_silhouette_score"), 
            clustering_result_dict.get("hierarchical_best_silhouette_score"), 
            clustering_result_dict.get("spectral_clustering_graph_based_best_silhouette_score"), 
            clustering_result_dict.get("kmeans_best_silhouette_score"), 
            # clustering_result_dict.get("hdbscan_best_silhouette_score"), 
            clustering_result_dict.get("affinity_propagation_best_silhouette_score"), 
            clustering_result_dict.get("agglomerative_best_silhouette_score"), 
            clustering_result_dict.get("spectral_embedding_based_best_silhouette_score"), 
        ]
        all_clustering_algos = [
            "span_tree_connected_components",
            "connected_components",
            "louvain",
            "hierarchical",
            "spectral_clustering_graph_based",
            "kmeans",
            # "hdbscan",
            "affinity_propagation",
            "agglomerative",
            "spectral_embedding_based",
        ]
        
        assert(len(all_clustering_algos) == len(all_silhouette_scores))
        
        clustering_algo_dict = {all_clustering_algos[i]: all_silhouette_scores[i]
                                for i in range(len(all_clustering_algos))}

        clustering_algo_dict = {key: val for key, val in clustering_algo_dict.items() 
                                if val is not None and isinstance(val, (int, float, np.floating, np.integer)) }
        
        best_clustering_algorithm = max(clustering_algo_dict, key=clustering_algo_dict.get)
        best_silhouette_score = clustering_algo_dict.get(best_clustering_algorithm)
        print(f"Best clustering algorithm {best_clustering_algorithm}, best silhouette score {best_silhouette_score}")
        best_clustering_labels = clustering_result_dict.get(f"{best_clustering_algorithm}_clusterings")
        print(f"{best_clustering_algorithm}_clusterings", best_clustering_labels, )
        
        worst_clustering_algorithm = min(clustering_algo_dict, key=clustering_algo_dict.get)
        worst_silhouette_score = clustering_algo_dict.get(worst_clustering_algorithm)
        
        return {
            "best_clustering_algorithm": best_clustering_algorithm,
            "best_silhouette_score": best_silhouette_score,
            "worst_clustering_algorithm": worst_clustering_algorithm,
            "worst_silhouette_score": worst_silhouette_score,
            "best_clustering_labels": best_clustering_labels,
        }
        
    def combine_pre_computed_low_dimensional_embeddings(self, clustering_result_dict: dict):
        pass
        
    def _convert_iterable(self, iterable_to_be_converted):
        assert(isinstance(iterable_to_be_converted, (list, set, tuple)))
        original_iterable = iterable_to_be_converted.copy()
        if isinstance(iterable_to_be_converted, (set,tuple)):
            iterable_to_be_converted = list(iterable_to_be_converted)
            
        for indx, item in enumerate(iterable_to_be_converted):
            if isinstance(item, np.floating):
                iterable_to_be_converted[indx] = float(item)
            elif isinstance(item, np.integer):
                iterable_to_be_converted[indx] = int(item)
            elif isinstance(item, pd.DataFrame):
                iterable_to_be_converted[indx] = item.to_dict(orient="records")
            elif isinstance(item, np.ndarray):
                iterable_to_be_converted[indx] = item.tolist()
            elif isinstance(item, dict):
                iterable_to_be_converted[indx] = self._convert_dict(item)
            elif isinstance(item, (list, tuple, set)):
                iterable_to_be_converted[indx] = self._convert_iterable(item)
            elif isinstance(item, (int, float, str, bool)):
                continue
            elif isinstance(item, torch.Tensor):
                iterable_to_be_converted[indx] = item.cpu().detach().numpy().tolist()
            else:
                iterable_to_be_converted = [""] * len(iterable_to_be_converted)
                break
                print(f"List was made empty because it contained an invalid type {type(item)}")
        
        # if isinstance(original_iterable, set):
        #     iterable_to_be_converted = set(iterable_to_be_converted)
        if isinstance(original_iterable, tuple):
            iterable_to_be_converted = tuple(iterable_to_be_converted)
        
        return iterable_to_be_converted
            
        
        
    def _convert_dict(self, clustering_result_save_ver_dict: dict):
        for k, v in clustering_result_save_ver_dict.items():
            if isinstance(v, (int, float, str, bool)):
                continue
            if isinstance(v, pd.DataFrame):
                clustering_result_save_ver_dict[k] = v.to_dict(orient="records")
            elif isinstance(v, np.ndarray):
                clustering_result_save_ver_dict[k] = v.tolist()
            elif isinstance(v, dict):
                clustering_result_save_ver_dict[k] = self._convert_dict(v)
            elif isinstance(v, torch.Tensor):
                clustering_result_save_ver_dict[k] = v.cpu().detach().numpy().tolist()
            elif isinstance(v, (list, tuple, set)):
                clustering_result_save_ver_dict[k] = self._convert_iterable(v)
                # for indx, iterable_val in enumerate(v):
                #     if isinstance(iterable_val, pd.DataFrame):
                #         clustering_result_save_ver_dict[k][indx] = sub_v.to_dict(orient="records")
                #     elif isinstance(iterable_val, np.ndarray):
                #         clustering_result_save_ver_dict[k][indx] = sub_v.tolist() 
                #     elif isinstance(v, dict):
                #         clustering_result_save_ver_dict[k][indx] = self._convert_dict(v)
                #     elif isinstance(iterable_val, (np.integer)):
                #         clustering_result_save_ver_dict[k][indx] = int(iterable_val)
                #     elif isinstance(iterable_val, (np.floating)):
                #         clustering_result_save_ver_dict[k][indx] = float(iterable_val)
                #     elif isinstance(iterable_val, (list, tuple)):
                #         if len(iterable_val) == 0: continue
                #         if isinstance(iterable_val[0], dict):
                #             clustering_result_save_ver_dict[k][indx] = [self._convert_dict(d) for d in iterable_val]
                #         if not isinstance(iterable_val[0], (str, float, int)):
                #             pass
                #         clustering_result_save_ver_dict[k][indx] = float(iterable_val)
                #     elif not isinstance(iterable_val, (str, float, int, list, set, tuple)):
                #         print(f"Removed {k} because it contained an invalid type {type(iterable_val)}")
            elif isinstance(v, (np.floating)):
                clustering_result_save_ver_dict[k] = float(v) 
            elif isinstance(v, np.integer):
                clustering_result_save_ver_dict[k] = int(v)          
            elif isinstance(v, (int, float, bool)):
                continue
            else:
                clustering_result_save_ver_dict[k] = None
                print(f"Removed {k} because it was of invalid type {type(v)}")
            
        return clustering_result_save_ver_dict
        
    def _label_point(self, x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        ax = [ax.annotate( str(point['val']), (point['x']+.02, point['y']), fontsize = 8) for _, point in a.iterrows()]
        adjust_text(ax)
    
    def get_silhouette_plot_data(self,clustering_result_dict: dict):

        cluster_nums, cluster_nums_silhouette_scores, algo_type = [], [], []
        for k,v in clustering_result_dict.items():
            
            if not isinstance(v, dict):
                continue
            
            if "n_clusters" not in v and "silhouette_scores" not in v:
                continue
            
            if len(v["n_clusters"]) != len(v["silhouette_scores"]):
                print(k)
                continue
            
            cluster_nums = cluster_nums + v["n_clusters"]
            cluster_nums_silhouette_scores = cluster_nums_silhouette_scores + v["silhouette_scores"]
            
            if k == "agglomerative":
                linkage_types = ["ward", "complete", "average", "single"]
                algo_type = algo_type + [f"{k}-{linkage_types[i % len(linkage_types)]}" for i in range(len(v["n_clusters"])) ]
                continue
                
            if len(v["n_clusters"]) != len(set(v["n_clusters"])) and (len(v["n_clusters"])/ len(set(v["n_clusters"]))).is_integer():
                num_categories = len(v["n_clusters"])/ len(set(v["n_clusters"]))
                num_categories = int(num_categories)
                categories = list(range((num_categories)))
                algo_type = algo_type + [f"{k}-{categories[i % len(categories)]}" for i in range(len(v["n_clusters"])) ]
                continue
                
            algo_type = algo_type + [k] * len(v["n_clusters"])
            
        return {
            "cluster_nums": cluster_nums,
            "cluster_nums_silhouette_scores": cluster_nums_silhouette_scores,
            "algo_type": algo_type,
        }

    def save_clustering_results(self, clustering_result_dict: dict, result_parent_dir: str):
        assert(os.path.exists(result_parent_dir))
        mi_id = clustering_result_dict.get("mi_id")
        result_dir = os.path.join(result_parent_dir, mi_id)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
            
        subj_img_embedding = clustering_result_dict.get("subj_img_embedding", None)
        if subj_img_embedding is not None:
            subj_img_embedding_df = pd.DataFrame(subj_img_embedding.T, columns = clustering_result_dict["img_id_list"])
            subj_img_embedding_df.to_csv(os.path.join(result_dir, f"{mi_id}-img-embedding.csv"))
            
        result_df = clustering_result_dict["subj_result_df"]
        if len(result_df.columns) != len(set(result_df.columns)):
            print(result_df.columns)

        result_df = result_df.loc[:,~result_df.columns.duplicated()].copy()
        result_df.to_csv(os.path.join(result_dir, "clustering_results.csv"))
        clustering_result_save_ver_dict = clustering_result_dict.copy()
        clustering_result_save_ver_dict = self._convert_dict(clustering_result_save_ver_dict)
                
        with open(os.path.join(result_dir, "clustering_result_dict.json"), "w") as outfile:
            json.dump(clustering_result_save_ver_dict, outfile)
        
        num_lower_embedding_methods = len([k for k, v in clustering_result_dict["low_dimensional_embedding_results"].items() 
                                           if isinstance(v,(np.ndarray, list, pd.DataFrame))])
        num_clustering_methods = len([k for k, v in clustering_result_dict.items() 
                                      if isinstance(v,dict) and v.get("best_cluster_labels", None) is not None])
        plot_clustering_dict = {k: v for k, v in clustering_result_dict.items() 
                                      if isinstance(v,dict) and v.get("best_cluster_labels", None) is not None
                                      and v.get("best_silhouette_score", None) is not None}
        
        fig, axes = plt.subplots(num_lower_embedding_methods, num_clustering_methods, figsize=(16, 10), subplot_kw={"xticks":(), "yticks": ()})
        for i, (projection_suffix, projection_result) in enumerate(clustering_result_dict["low_dimensional_embedding_results"].items()):
            projection_suffix = projection_suffix.split('_embedding')[0].replace('_','').upper()
            if not isinstance(projection_result, (np.ndarray, list, pd.DataFrame)):
                continue
            
            for j, (clustering_algo, clustering_result) in enumerate(plot_clustering_dict.items()):
                if not isinstance(clustering_result, dict):
                    continue 
                
                if clustering_result.get("best_silhouette_score", None) is None or clustering_result.get("best_cluster_labels", None) is None:
                    continue
                    
                ax = axes[i,j]
                ax.scatter(result_df[f"{projection_suffix}1"], result_df[f"{projection_suffix}2"], c=plt.cm.tab10(clustering_result.get("best_cluster_labels")), alpha=.7)
                if i == 0:
                    if clustering_algo == clustering_result_dict.get("best_clustering_algorithm"):
                        ax.set_title(clustering_algo.upper(), size = 6, weight="bold")
                    else:
                        ax.set_title(clustering_algo, size = 6)
                if j == 0:
                    ax.set_ylabel(projection_suffix, fontsize = 6)
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_aspect("equal")
                
        # plt.tight_layout()
        plt.savefig(os.path.join(result_dir,f"all_clustering_embeddings.png"), dpi = 600, bbox_inches='tight')
        plt.show()
        plt.clf()
        
        silhouette_plot_dict = self.get_silhouette_plot_data(clustering_result_dict=clustering_result_dict)
        cluster_nums, cluster_nums_silhouette_scores, algo_type = silhouette_plot_dict.get("cluster_nums"), silhouette_plot_dict.get("cluster_nums_silhouette_scores"), silhouette_plot_dict.get("algo_type"), 
        ax = sns.lineplot(x=cluster_nums, y=cluster_nums_silhouette_scores,
                  hue=algo_type, errorbar=None, palette='colorblind')
        # ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.xlabel("Cluster Number")
        plt.ylabel("Silhouette Score")
        plt.savefig(os.path.join(result_dir,f"cluster_num_vs_silhouette_score.png"), bbox_inches='tight')
        plt.show()   
        plt.clf() 
        
        for projection_suffix, projection_result in clustering_result_dict["low_dimensional_embedding_results"].items():
            projection_suffix = projection_suffix.split('_embedding')[0].replace('_','').upper()
            if not isinstance(projection_result, (np.ndarray, list, pd.DataFrame)):
                continue
            fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
            sns.set_style(None, {"grid.color": ".6", "grid.linestyle": ":"})
            sns.scatterplot(data=result_df, x=f'{projection_suffix}1', y=f'{projection_suffix}2', hue='best_cluster_labels', palette='colorblind')
            ax.collections[0].set_sizes([100])
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.title(f'Scatter plot of images using {projection_suffix}');
            plt.xlabel(f'{projection_suffix}1');
            plt.ylabel(f'{projection_suffix}2');
            plt.axis('equal')
            plt.savefig(os.path.join(result_dir,f"{projection_suffix}_best_clustering_results_no_img_label.png"), bbox_inches='tight')
            plt.show()
            plt.clf()
            
            fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
            sns.set_style(None, {"grid.color": ".6", "grid.linestyle": ":"})
            sns.scatterplot(data=result_df, x=f'{projection_suffix}1', y=f'{projection_suffix}2', hue='best_cluster_labels', palette='colorblind')
            ax.collections[0].set_sizes([100])
            self._label_point(result_df[f'{projection_suffix}1'], result_df[f'{projection_suffix}2'], result_df["img_id"], ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.title(f'Scatter plot of images using {projection_suffix}');
            plt.xlabel(f'{projection_suffix}1');
            plt.ylabel(f'{projection_suffix}2');
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir,f"{projection_suffix}_best_clustering_results.png"), bbox_inches='tight')
            plt.show()
            plt.clf()
        
        # for projection_suffix, projection_result in clustering_result_dict["low_dimensional_embedding_results"].items():
        #     projection_suffix = projection_suffix.split('_embedding')[0].replace('_','').upper()
        #     if not isinstance(projection_result, (np.ndarray, list, pd.DataFrame)):
        #         continue
        #     fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
        #     sns.set_style(None, {"grid.color": ".6", "grid.linestyle": ":"})
        #     sns.scatterplot(data=result_df, x=f'{projection_suffix}1', y=f'{projection_suffix}2', hue='best_cluster_labels', palette='hls')
        #     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        #     plt.title(f'Scatter plot of images using {projection_suffix}');
        #     plt.xlabel(f'{projection_suffix}1');
        #     plt.ylabel(f'{projection_suffix}2');
        #     plt.axis('equal')
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(result_dir,f"{projection_suffix}_best_clustering_results_no_img_label.png"), bbox_inches='tight')
        #     plt.show()
        #     plt.clf()
            
        #     fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
        #     sns.set_style(None, {"grid.color": ".6", "grid.linestyle": ":"})
        #     sns.scatterplot(data=result_df, x=f'{projection_suffix}1', y=f'{projection_suffix}2', hue='best_cluster_labels', palette='hls')
        #     self._label_point(result_df[f'{projection_suffix}1'], result_df[f'{projection_suffix}2'], result_df["img_id"], ax)
        #     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        #     plt.title(f'Scatter plot of images using {projection_suffix}');
        #     plt.xlabel(f'{projection_suffix}1');
        #     plt.ylabel(f'{projection_suffix}2');
        #     plt.axis('equal')
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(result_dir,f"{projection_suffix}_best_clustering_results.png"), bbox_inches='tight')
        #     plt.show()
        #     plt.clf()
        
        # for projection_suffix in ["ISOMAP", "MDS", "TSNE"]:
        #     fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
        #     sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
        #     sns.scatterplot(data=result_df, x=f'{projection_suffix}1', y=f'{projection_suffix}2', hue='best_clustering_labels', palette='hls')
        #     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        #     plt.title(f'Scatter plot of images using {projection_suffix}');
        #     plt.xlabel(f'{projection_suffix}1');
        #     plt.ylabel(f'{projection_suffix}2');
        #     plt.axis('equal')
        #     plt.savefig(os.path.join(result_dir,f"{projection_suffix}_best_clustering_results_no_img_label.png"), bbox_inches='tight')
        #     plt.show()
        #     plt.clf()
            
        #     fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
        #     sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
        #     sns.scatterplot(data=result_df, x=f'{projection_suffix}1', y=f'{projection_suffix}2', hue='best_clustering_labels', palette='hls')
        #     self._label_point(result_df[f'{projection_suffix}1'], result_df[f'{projection_suffix}2'], result_df["img_id"], ax)
        #     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        #     plt.title(f'Scatter plot of images using {projection_suffix}');
        #     plt.xlabel(f'{projection_suffix}1');
        #     plt.ylabel(f'{projection_suffix}2');
        #     plt.axis('equal')
        #     plt.savefig(os.path.join(result_dir,f"{projection_suffix}_best_clustering_results.png"), bbox_inches='tight')
        #     plt.show()
        #     plt.clf()
        try:  
            communities = self.convert_sklearn_labels_to_communities(result_df["best_cluster_labels"])
        except IndexError as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(e, exc_type, fname, exc_tb.tb_lineno)
            return 
                
        try:
            draw_communities(clustering_result_dict.get("subj_adjacency_matrix"), communities)
        except AttributeError as e:
            print(e)
            
        plt.savefig(os.path.join(result_dir,f"network_diagram_best_clustering_results.png"), bbox_inches='tight')
        plt.show()
        plt.clf()
       
    def combine_pre_computed_dimensionality_reductions(self, csv_path: str, clustering_result_dict: dict):
        clustering_result_dict = clustering_result_dict.copy()
        df = pd.read_csv(csv_path, index_col = 0)
        embedding_columns = {i for i in df.columns if i.endswith('2') or i.endswith('1')}
        embedding_names = set([i[:-1] for i in embedding_columns])
        subj_result_df = clustering_result_dict["subj_result_df"].copy()
        other_columns = set(df.columns) - embedding_columns
        other_columns = other_columns - {'mi_id'}
        original_low_dimensional_embedding_results = clustering_result_dict["low_dimensional_embedding_results"].copy()
        
        
        for embedding_name in embedding_names:
            embedding_col1 = f"{embedding_name}1"
            embedding_col2 = f"{embedding_name}2"
            
            if embedding_col1 not in embedding_columns and embedding_col2 not in embedding_columns:
                print(embedding_col1, embedding_col2)
                print(embedding_columns)
                print(embedding_name, " was not in DF??")
                
            clustering_result_dict[f"{embedding_name}_embedding_results"] = df[[embedding_col1, embedding_col2]].to_numpy()
            clustering_result_dict["low_dimensional_embedding_results"][f"{embedding_name}_embedding_results"] = df[[embedding_col1, embedding_col2]].to_numpy()

        if len(other_columns) <= len(embedding_columns) / 2:
            for col in other_columns:
                val = df[[col]].to_numpy().tolist()
                if isinstance(val[0], (int, str, float, np.floating, np.integer, bool)):
                    if len(set(val)) == 1:
                        val = val[0]
                clustering_result_dict[col] = df[[col]].to_numpy().tolist()
        else:
            clustering_result_dict[f"other_metadata"] = {}
            for col in other_columns:
                val = df[[col]].to_numpy().tolist()
                if isinstance(val[0], (int, str, float, np.floating, np.integer, bool)):
                    if len(set(val)) == 1:
                        val = val[0]
                clustering_result_dict[f"other_metadata"][col] = val
                
        new_low_dimensional_embedding_results = {k: v for k, v in clustering_result_dict["low_dimensional_embedding_results"].items()
                                                 if k not in original_low_dimensional_embedding_results.keys()}
        new_low_dimensional_embedding_df = [pd.DataFrame(v,
                                                     columns = [
                                                         f"{k.split('_embedding')[0].replace('_','').upper()}1",
                                                         f"{k.split('_embedding')[0].replace('_','').upper()}2",
                                                         ]) for k, v in new_low_dimensional_embedding_results.items()]
        
        assert(all(subj_result_df.shape[0] == df.shape[0] for df in new_low_dimensional_embedding_df))
        new_low_dimensional_embedding_df = pd.concat(new_low_dimensional_embedding_df, axis = 1)
                
        new_subj_result_df_dict = {}
        for k, v in clustering_result_dict.items():
            if k in subj_result_df.columns:
                continue
            
            if isinstance(v, (str, int, float, bool)):
                new_subj_result_df_dict[k] = v
            elif isinstance(v, np.floating):
                new_subj_result_df_dict[k] = float(v)
            elif isinstance(v, np.integer):
                new_subj_result_df_dict[k] = int(v)
                
            elif isinstance(v, (np.ndarray)):
                v = v.tolist()
            elif isinstance(v, (tuple, set)):
                v = list(v)
            if isinstance(v, (list)) and len(v) == len(clustering_result_dict["img_id_list"]):
                new_subj_result_df_dict[k] = v
                
            if not isinstance(v, dict):
                continue
            for sub_key, sub_key_val in v.items():
                if isinstance(sub_key_val, (list, tuple)):
                    for indx, indx_val in enumerate(sub_key_val):
                        if not isinstance(indx_val, (int, float, np.floating, np.integer, bool, str)):
                            continue
                        new_subj_result_df_dict[f"{k}-{sub_key}-{indx}"] = [indx_val] * len(clustering_result_dict["img_id_list"])
                    continue 
                if not isinstance(sub_key_val, (int, float, np.floating, np.integer, bool, str)):
                    continue 
                new_subj_result_df_dict[f"{k}-{sub_key}"] = [sub_key_val] * len(clustering_result_dict["img_id_list"])
                
        new_subj_result_df = pd.DataFrame(new_subj_result_df_dict)
        assert(len(clustering_result_dict["img_id_list"]) == subj_result_df.shape[0] == new_subj_result_df.shape[0])
        new_subj_result_df = pd.concat([subj_result_df, new_subj_result_df, new_low_dimensional_embedding_df], axis = 1)
                
        new_subj_result_df = new_subj_result_df.loc[:,~new_subj_result_df.columns.duplicated()].copy()
        clustering_result_dict['subj_result_df'] = new_subj_result_df
        return clustering_result_dict 
    
    def process_pre_computed_results(self,pre_computed_results_dirs: list):
        pre_computed_results_filepaths = [None] * len(pre_computed_results_dirs)
        for indx, pre_compute_dir in enumerate(pre_computed_results_dirs):
            assert(os.path.exists(pre_compute_dir))
            
            if pre_compute_dir[-1] == '/':
                csv_filepaths = glob.glob(f"{pre_compute_dir}*.csv")
            else: 
                csv_filepaths = glob.glob(f"{pre_compute_dir}/*.csv")
                
            assert(all(os.path.exists(i) for i in csv_filepaths))
            csv_filepaths = {os.path.split(i)[-1].split(".csv")[0]: i for i in csv_filepaths}
            pre_computed_results_filepaths[indx] = csv_filepaths
                
        return pre_computed_results_filepaths

    def all_subj_clustering_pipeline(self, result_parent_dir: str, pre_computed_results_dirs: list = None):
        assert(os.path.exists(result_parent_dir))
        
        self.get_all_subj_metadata()
        
        num_subj = len(self.mi_ids)
        all_subj_best_results_df = [pd.DataFrame()] * num_subj
        result_timestamp = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        result_dir = os.path.join("/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results", f"clustering-{result_timestamp}")
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
            
        if pre_computed_results_dirs is not None and len(pre_computed_results_dirs) != 0:
            pre_computed_results_filepaths = self.process_pre_computed_results(pre_computed_results_dirs)
        else: 
            pre_computed_results_filepaths = None
        
        for indx, mi_id in tqdm(enumerate(self.mi_ids), total = num_subj):
            if indx % 1000 == 0:
                print(indx, mi_id)
            
            # if mi_id != "P0018571" and mi_id != "P0012750": 
            #     continue

            subj_data = self.get_subject_data(mi_id=mi_id)
            
            clustering_result_dict = self.try_all_clustering_methods(subj_data)
            
            if pre_computed_results_filepaths is not None:
                for j, pre_computed_csv_dict in enumerate(pre_computed_results_filepaths):
                    pre_computed_csv_path = pre_computed_csv_dict.get(mi_id, None)
                    if pre_computed_csv_path is None:
                        print(f"No pre-computed embedding found for {mi_id} in {pre_computed_results_dirs[j]}")
                        continue
                    
                    clustering_result_dict = self.combine_pre_computed_dimensionality_reductions(csv_path = pre_computed_csv_path,
                                                                                                 clustering_result_dict = clustering_result_dict,
                                                                                          )                
                    
            self.save_clustering_results(clustering_result_dict, result_dir)
            subj_best_results_df = clustering_result_dict["subj_result_df"].copy()
            subj_best_results_df = subj_best_results_df[["mi_id", "max_n_clusters",
                                                            "min_n_clusters", "best_clustering_algorithm",
                                                            "best_silhouette_score", "worst_clustering_algorithm",
                                                            "worst_silhouette_score",
                                                            ]]
            subj_best_results_df.drop_duplicates(inplace = True)
            all_subj_best_results_df[indx] = subj_best_results_df
            
            # try:
            #     clustering_result_dict = self.try_all_clustering_methods(subj_data)
            #     self.save_clustering_results(clustering_result_dict, result_dir)
            # except KeyboardInterrupt as e:
            #     all_subj_best_results_df = [i for i in all_subj_best_results_df if i is not None and isinstance(i, pd.DataFrame) and not i.empty]
            #     all_subj_best_results_df = pd.concat(all_subj_best_results_df)
            #     self.all_subj_best_results_df = all_subj_best_results_df
            #     all_subj_best_results_df.to_csv(os.path.join(result_dir, "all_subj_best_results.csv"), index=None)
            #     print(e)
            #     sys.exit(1)
            # except Exception as e:
            #     all_subj_best_results_df[indx] = pd.DataFrame()
            #     print(f"Error occurred while processing MI_ID: {mi_id}")
            #     print(e)
            # else:
            #     subj_best_results_df = clustering_result_dict["subj_result_df"].copy()
            #     subj_best_results_df = subj_best_results_df[["mi_id", "max_n_clusters",
            #                                                  "min_n_clusters", "best_clustering_algorithm",
            #                                                  "best_silhouette_score", "worst_clustering_algorithm",
            #                                                  "worst_silhouette_score", "img_id_list",
            #                                                  ]]
            #     subj_best_results_df.drop_duplicates(inplace = True)
                
                # subj_best_results_df["mi_id"] = clustering_result_dict["mi_id"]
                # subj_best_results_df["max_n_clusters"] = clustering_result_dict["max_n_clusters"]
                # subj_best_results_df["min_n_clusters"] = clustering_result_dict["min_n_clusters"]
                # subj_best_results_df["best_clustering_algorithm"] = clustering_result_dict["best_clustering_algorithm"]
                # subj_best_results_df["best_silhouette_score"] = clustering_result_dict["best_silhouette_score"]
                # subj_best_results_df["worst_clustering_algorithm"] = clustering_result_dict["worst_clustering_algorithm"]
                # subj_best_results_df["worst_silhouette_score"] = clustering_result_dict["worst_silhouette_score"]
                
                # all_subj_best_results_df[indx] = subj_best_results_df
                
        # all_subj_best_results_df = [i for i in all_subj_best_results_df if i is not None and isinstance(i, pd.DataFrame) and not i.empty]
        all_subj_best_results_df = pd.concat(all_subj_best_results_df, axis = 0 )
        self.all_subj_best_results_df = all_subj_best_results_df
        all_subj_best_results_df.to_csv(os.path.join(result_dir, "all_subj_best_results.csv"), index=None)
        
        print(f"All clustering complete. Results saved to {result_dir}")
                
            


