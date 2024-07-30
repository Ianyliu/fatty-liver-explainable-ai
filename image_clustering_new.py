import sys 
from tqdm import tqdm
import ast
import pandas as pd 
import datetime
import glob
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv
import seaborn as sns
from usflc_xai import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,

)
from communities.algorithms import spectral_clustering
from communities.algorithms import girvan_newman

from sklearn.cluster import KMeans
import time 
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

from graph_based_clustering import SpanTreeConnectedComponentsClustering
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from graph_based_clustering import ConnectedComponentsClustering
from communities.algorithms import louvain_method, hierarchical_clustering, girvan_newman
from communities.visualization import draw_communities
from communities.algorithms import hierarchical_clustering
load_dotenv()

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from typing import List, Dict, Callable
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NeighborhoodComponentsAnalysis
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
        
        subj_img_embedding = subj_data_loader.x.to('cpu')
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
        print("Clustering starting...")
        subj_dict_cols = ['subj_img_embedding','subj_adjacency_matrix','subj_data_loader','mi_id', 'y','img_id_list',]
        assert(all(col in subj_dict_cols for col in subj_data.keys()))
        
        subj_img_embedding = subj_data.get('subj_img_embedding') 
        subj_adjacency_matrix = subj_data.get('subj_adjacency_matrix') 
        mi_id = subj_data.get('mi_id') 
        img_id_list = subj_data.get('img_id_list') 
        max_n_clusters = len(img_id_list) - 3
        
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
            # "nca_embeddings_results",
            "mds_embeddings_results",
            "isomap_embeddings_results",
            "tsne_embeddings_results",
            "tsne_best_perplexity", 
            "tsne_best_kl_divergence",
            "subj_result_df",
        ])
        

        span_tree_connected_components_results = self.get_span_tree_connected_components_clusterings(adj_matrix=subj_adjacency_matrix,
                                                                                                     img_embedding=subj_img_embedding,
                                                                                                     min_n_clusters=min_n_clusters,
                                                                                                     max_n_clusters=max_n_clusters,)
        connected_components_results = self.get_connected_components_clusterings(adj_matrix=subj_adjacency_matrix,
                                                                                 img_embedding=subj_img_embedding,
                                                                                 min_n_clusters=min_n_clusters,
                                                                                 max_n_clusters=max_n_clusters,
                                                                                 max_iter=max_clustering_iter,)
        louvain_results = self.get_louvain_clusterings(adj_matrix=subj_adjacency_matrix,
                                                       img_embedding=subj_img_embedding,
                                                       min_n_clusters=min_n_clusters,
                                                       max_n_clusters=max_n_clusters,)
        hierarchical_results = self.get_hierarchical_clusterings(adj_matrix=subj_adjacency_matrix,
                                                                 img_embedding=subj_img_embedding,
                                                                 min_n_clusters=min_n_clusters,
                                                                 max_n_clusters=max_n_clusters,)
        graph_based_spectral_results = self.get_graph_based_spectral_clusterings(adj_matrix=subj_adjacency_matrix,
                                                                                 img_embedding=subj_img_embedding,
                                                                                 min_n_clusters=min_n_clusters,
                                                                                 max_n_clusters=max_n_clusters,)
        k_means_results = self.get_k_means_clusterings(
                                                       img_embedding=subj_img_embedding,
                                                       min_n_clusters=min_n_clusters,
                                                       max_n_clusters=max_n_clusters,
                                                       max_iter=max_clustering_iter,)
        hdbscan_results = self.get_hdbscan_clusterings(
                                                       img_embedding=subj_img_embedding,
                                                       min_n_clusters=min_n_clusters,
                                                       max_n_clusters=max_n_clusters,
                                                       max_iter=max_clustering_iter,)
        affinity_propagation_results = self.get_affinity_propagation_clusterings(
                                                                                 img_embedding=subj_img_embedding,
                                                                                 min_n_clusters=min_n_clusters,
                                                                                 max_n_clusters=max_n_clusters,
                                                                                 max_iter=max_clustering_iter,)
        agglomerative_results = self.get_agglomerative_clusterings(
                                                                   img_embedding=subj_img_embedding,
                                                                   min_n_clusters=min_n_clusters,
                                                                   max_n_clusters=max_n_clusters,
                                                                   max_iter=max_clustering_iter,)
        spectral_embedding_based_results = self.get_spectral_embedding_based_clusterings(
                                                                                         img_embedding=subj_img_embedding,
                                                                                         min_n_clusters=min_n_clusters,
                                                                                         max_n_clusters=max_n_clusters,
                                                                                         max_iter=max_clustering_iter,)
        # nca_embedding_results = self.get_nca_embedings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        isomap_results = self.get_isomap_embedings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        mds_embedding_results = self.get_mds_embedings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)
        tsne_embedding_results = self.get_tsne_embedings(img_embedding=subj_img_embedding, max_iter=max_clustering_iter)

        clustering_result_dict["mi_id"] = mi_id
        clustering_result_dict["img_id_list"] = img_id_list
        clustering_result_dict["subj_img_embedding"] = subj_img_embedding
        clustering_result_dict["subj_adjacency_matrix"] = subj_adjacency_matrix
        clustering_result_dict["max_n_clusters"] = max_n_clusters
        clustering_result_dict["min_n_clusters"] = min_n_clusters
        clustering_result_dict["span_tree_connected_components_clusterings"] = span_tree_connected_components_results["span_tree_best_cluster_labels"]
        clustering_result_dict["span_tree_connected_components_n_clusters"] = span_tree_connected_components_results["span_tree_best_n_clusters"]
        clustering_result_dict["span_tree_connected_components_best_silhouette_score"] = span_tree_connected_components_results["span_tree_best_silhouette_score"]
        clustering_result_dict["connected_components_clusterings"] = connected_components_results["connected_components_best_cluster_labels"]
        clustering_result_dict["connected_components_best_threshold"] = connected_components_results["connected_components_best_threshold"]
        clustering_result_dict["connected_components_best_metric"] = connected_components_results["connected_components_best_metric"]
        clustering_result_dict["connected_components_best_silhouette_score"] = connected_components_results["connected_components_best_silhouette_score"]
        clustering_result_dict["louvain_clusterings"] = louvain_results["louvain_best_cluster_labels"]
        clustering_result_dict["louvain_best_silhouette_score"] = louvain_results["louvain_best_silhouette_score"]
        clustering_result_dict["hierarchical_clusterings"] = hierarchical_results["hierarchical_best_cluster_labels"]
        clustering_result_dict["hierarchical_best_metric"] = hierarchical_results["hierarchical_best_metric"]
        clustering_result_dict["hierarchical_best_linkage_type"] = hierarchical_results["hierarchical_best_linkage"]
        clustering_result_dict["hierarchical_best_silhouette_score"] = hierarchical_results["hierarchical_best_silhouette_score"]
        clustering_result_dict["spectral_clustering_graph_based_clusterings"] = graph_based_spectral_results["graph_based_spectral_best_cluster_labels"]
        clustering_result_dict["spectral_clustering_graph_based_n_clusters"] = graph_based_spectral_results["graph_based_spectral_best_n_clusters"]
        clustering_result_dict["spectral_clustering_graph_based_best_silhouette_score"] = graph_based_spectral_results["graph_based_spectral_best_silhouette_score"]
        clustering_result_dict["kmeans_clusterings"] = k_means_results["k_means_best_cluster_labels"]
        clustering_result_dict["kmeans_n_clusters"] = k_means_results["k_means_best_n_clusters"]
        clustering_result_dict["kmeans_best_silhouette_score"] = k_means_results["k_means_best_silhouette_score"]
        clustering_result_dict["hdbscan_clusterings"] = hdbscan_results["hdbscan_best_cluster_labels"]
        clustering_result_dict["hdbscan_best_silhouette_score"] = hdbscan_results["hdbscan_best_silhouette_score"] 
        clustering_result_dict["hdbscan_best_cluster_selection_method"] = hdbscan_results["hdbscan_best_cluster_selection_method"]        
        clustering_result_dict["affinity_propagation_clusterings"] = affinity_propagation_results["affinity_propagation_best_cluster_labels"]
        clustering_result_dict["affinity_propagation_best_silhouette_score"] = affinity_propagation_results["affinity_propagation_best_silhouette_score"] 
        clustering_result_dict["agglomerative_clusterings"] = agglomerative_results["agglomerative_best_cluster_labels"]
        clustering_result_dict["agglomerative_best_n_clusters"] = agglomerative_results["agglomerative_best_n_clusters"] 
        clustering_result_dict["agglomerative_best_silhouette_score"] = agglomerative_results["agglomerative_best_silhouette_score"] 
        clustering_result_dict["spectral_embedding_based_clusterings"] = spectral_embedding_based_results["spectral_embedding_based_best_cluster_labels"]
        clustering_result_dict["spectral_embedding_based_best_affinity_method"] = spectral_embedding_based_results["spectral_embedding_based_best_affinity_method"] 
        clustering_result_dict["spectral_embedding_based_best_n_clusters"] = spectral_embedding_based_results["spectral_embedding_based_best_n_clusters"] 
        clustering_result_dict["spectral_embedding_based_best_silhouette_score"] = spectral_embedding_based_results["spectral_embedding_based_best_silhouette_score"] 
        # clustering_result_dict["nca_embeddings_results"] = nca_embedding_results
        clustering_result_dict["mds_embeddings_results"] = mds_embedding_results
        clustering_result_dict["isomap_embeddings_results"] = isomap_results
        clustering_result_dict["tsne_embeddings_results"] = tsne_embedding_results["best_tsne_results"]
        clustering_result_dict["tsne_best_perplexity"] =  tsne_embedding_results["best_perplexity"]
        clustering_result_dict["tsne_best_kl_divergence"] = tsne_embedding_results["best_kl_divergence"]
        
        clustering_result_dict = clustering_result_dict | self.get_best_clustering_algorithm(clustering_result_dict)
        
        subj_result_df = pd.DataFrame()
        subj_result_df['mi_id'] = [mi_id] * len(img_id_list)
        subj_result_df['img_id'] = img_id_list
        subj_result_df['best_clustering_algorithm'] = clustering_result_dict["best_clustering_algorithm"]
        subj_result_df['best_silhouette_score'] = clustering_result_dict["best_silhouette_score"]
        subj_result_df['worst_silhouette_score'] = clustering_result_dict["worst_silhouette_score"]
        subj_result_df['best_clustering_labels'] = clustering_result_dict["best_clustering_labels"]

        tsne_df = pd.DataFrame(clustering_result_dict["tsne_embeddings_results"], columns=["TSNE1", "TSNE2"])
        # nca_df = pd.DataFrame(nca_embedding_results, columns=["NCA1", "NCA2"])
        isomap_df = pd.DataFrame(isomap_results, columns=["ISOMAP1", "ISOMAP2"])
        mds_df = pd.DataFrame(mds_embedding_results, columns=["MDS1", "MDS2"])
        
        # assert(mds_df.shape[0] == isomap_df.shape[0] == nca_df.shape[0] == tsne_df.shape[0] == subj_result_df.shape[0])
        assert(mds_df.shape[0] == isomap_df.shape[0] == tsne_df.shape[0] == subj_result_df.shape[0])
        
        # subj_result_df = pd.concat([subj_result_df, tsne_df, nca_df, isomap_df, mds_df], axis=1)
        subj_result_df = pd.concat([subj_result_df, tsne_df, isomap_df, mds_df], axis=1)
        
        subj_result_df["span_tree_connected_components_clusterings"] = clustering_result_dict["span_tree_connected_components_clusterings"]
        subj_result_df["connected_components_clusterings"] = clustering_result_dict["connected_components_clusterings"]
        subj_result_df["louvain_clusterings"] = clustering_result_dict["louvain_clusterings"]
        subj_result_df["hierarchical_clusterings"] = clustering_result_dict["hierarchical_clusterings"]
        subj_result_df["spectral_clustering_graph_based_clusterings"] = clustering_result_dict["spectral_clustering_graph_based_clusterings"]
        subj_result_df["kmeans_clusterings"] = clustering_result_dict["kmeans_clusterings"]
        subj_result_df["hdbscan_clusterings"] = clustering_result_dict["hdbscan_clusterings"]
        subj_result_df["affinity_propagation_clusterings"] = clustering_result_dict["affinity_propagation_clusterings"]
        subj_result_df["agglomerative_clusterings"] = clustering_result_dict["agglomerative_clusterings"]
        subj_result_df["spectral_embedding_based_clusterings"] = clustering_result_dict["spectral_embedding_based_clusterings"]
        
        subj_result_df["span_tree_connected_components_n_clusters"] = [clustering_result_dict["span_tree_connected_components_n_clusters"]] * len(img_id_list)
        subj_result_df["span_tree_connected_components_best_silhouette_score"] = [clustering_result_dict["span_tree_connected_components_best_silhouette_score"]] * len(img_id_list)
        subj_result_df["connected_components_best_threshold"] = [clustering_result_dict["connected_components_best_threshold"]] * len(img_id_list)
        subj_result_df["connected_components_best_metric"] = [clustering_result_dict["connected_components_best_metric"]] * len(img_id_list)
        subj_result_df["connected_components_best_silhouette_score"] = [clustering_result_dict["connected_components_best_silhouette_score"]] * len(img_id_list)
        subj_result_df["louvain_best_silhouette_score"] = [clustering_result_dict["louvain_best_silhouette_score"]] * len(img_id_list)
        subj_result_df["hierarchical_best_metric"] = [clustering_result_dict["hierarchical_best_metric"]] * len(img_id_list)
        subj_result_df["hierarchical_best_linkage_type"] = [clustering_result_dict["hierarchical_best_linkage_type"]] * len(img_id_list)
        subj_result_df["hierarchical_best_silhouette_score"] = [clustering_result_dict["hierarchical_best_silhouette_score"]] * len(img_id_list)
        subj_result_df["spectral_clustering_graph_based_n_clusters"] = [clustering_result_dict["spectral_clustering_graph_based_n_clusters"]] * len(img_id_list)
        subj_result_df["spectral_clustering_graph_based_best_silhouette_score"] = [clustering_result_dict["spectral_clustering_graph_based_best_silhouette_score"]] * len(img_id_list)
        subj_result_df["kmeans_n_clusters"] = [clustering_result_dict["kmeans_n_clusters"]] * len(img_id_list)
        subj_result_df["kmeans_best_silhouette_score"] = [clustering_result_dict["kmeans_best_silhouette_score"]] * len(img_id_list)
        subj_result_df["hdbscan_best_silhouette_score"] = [clustering_result_dict["hdbscan_best_silhouette_score"]] * len(img_id_list)
        subj_result_df["hdbscan_best_cluster_selection_method"] = [clustering_result_dict["hdbscan_best_cluster_selection_method"]] * len(img_id_list)
        subj_result_df["affinity_propagation_best_silhouette_score"] = [clustering_result_dict["affinity_propagation_best_silhouette_score"]] * len(img_id_list)
        subj_result_df["agglomerative_best_n_clusters"] = [clustering_result_dict["agglomerative_best_n_clusters"]] * len(img_id_list)
        subj_result_df["agglomerative_best_silhouette_score"] = [clustering_result_dict["agglomerative_best_silhouette_score"]] * len(img_id_list)
        subj_result_df["spectral_embedding_based_best_affinity_method"] = [clustering_result_dict["spectral_embedding_based_best_affinity_method"]] * len(img_id_list)
        subj_result_df["spectral_embedding_based_best_n_clusters"] = [clustering_result_dict["spectral_embedding_based_best_n_clusters"]] * len(img_id_list)
        subj_result_df["spectral_embedding_based_best_silhouette_score"] = [clustering_result_dict["spectral_embedding_based_best_silhouette_score"]] * len(img_id_list)
        
        subj_result_df["tsne_best_perplexity"] =  [clustering_result_dict["tsne_best_perplexity"]] * len(img_id_list)
        subj_result_df["tsne_best_kl_divergence"] = [clustering_result_dict["tsne_best_kl_divergence"]] * len(img_id_list)
        
        clustering_result_dict["subj_result_df"] = subj_result_df
        if len(subj_result_df.columns) != len(set(subj_result_df.columns)):
            print(subj_result_df.head())
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
        
    def get_connected_components_clusterings(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int = 1000):
        best_cluster_labels = None
        best_threshold = None
        best_metric = None
        best_silhouette_score = -100
        thresholds = np.linspace(0.15, 0.97, num = max_iter)
        for threshold in thresholds:
            metric = "euclidean"
            clustering = ConnectedComponentsClustering(
                threshold=threshold,
                metric=metric,
                n_jobs=-1,
            )
            
            labels_pred = clustering.fit_predict(adj_matrix)
            
            if min_n_clusters < len(labels_pred) < max_n_clusters and labels_pred_score > best_silhouette_score:
                labels_pred_score = silhouette_score(img_embedding, labels_pred)
                best_silhouette_score = labels_pred_score
                best_cluster_labels = labels_pred
                best_metric = metric
                best_threshold = threshold
                
            labels_pred = clustering.fit_predict(img_embedding)
            
            if min_n_clusters < len(labels_pred) < max_n_clusters and labels_pred_score > best_silhouette_score:
                labels_pred_score = silhouette_score(img_embedding, labels_pred)
                best_silhouette_score = labels_pred_score
                best_cluster_labels = labels_pred
                best_metric = metric
                best_threshold = threshold
                
            metric = "cosine"
            clustering = ConnectedComponentsClustering(
                threshold=threshold,
                metric=metric,
                n_jobs=-1,
            )
            
            labels_pred = clustering.fit_predict(adj_matrix)
            
            if min_n_clusters < len(labels_pred) < max_n_clusters and labels_pred_score > best_silhouette_score:
                labels_pred_score = silhouette_score(img_embedding, labels_pred)
                best_silhouette_score = labels_pred_score
                best_cluster_labels = labels_pred
                best_metric = metric
                best_threshold = threshold
                
            labels_pred = clustering.fit_predict(img_embedding)
            
            if min_n_clusters < len(labels_pred) < max_n_clusters and labels_pred_score > best_silhouette_score:
                labels_pred_score = silhouette_score(img_embedding, labels_pred)
                best_silhouette_score = labels_pred_score
                best_cluster_labels = labels_pred
                best_metric = metric
                best_threshold = threshold
            
            
        return {
            "connected_components_best_threshold": best_threshold, 
            "connected_components_best_cluster_labels": best_cluster_labels, 
            "connected_components_best_silhouette_score": best_silhouette_score,
            "connected_components_best_metric": best_metric,
            }
    
    def get_louvain_clusterings(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int):
        communities, _ = louvain_method(adj_matrix=adj_matrix)
        labels_pred = self.convert_communities_output_to_labels(communities)
        
        if min_n_clusters < len(set(labels_pred)) < max_n_clusters: 
            best_silhouette_score = silhouette_score(img_embedding, labels_pred)
        else:
            best_silhouette_score = None 
        
        return {
            "louvain_best_cluster_labels": labels_pred,
            "louvain_best_silhouette_score": best_silhouette_score,
        }
        
    def get_hierarchical_clusterings(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int):
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
        
    def get_graph_based_spectral_clusterings(self, adj_matrix: np.ndarray, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int):
        best_n_clusters = None
        best_silhouette_score = -100
        best_cluster_labels = None
        for i in range(min_n_clusters, max_n_clusters): 
            i = i + 1
            communities = spectral_clustering(adj_matrix=adj_matrix, k = i)
            labels_pred = self.convert_communities_output_to_labels(communities)
            try: 
                if min_n_clusters < len(set(labels_pred)) < max_n_clusters:
                    labels_pred_score = silhouette_score(img_embedding, labels_pred)
                    if labels_pred_score > best_silhouette_score:
                        best_silhouette_score = silhouette_score(img_embedding, labels_pred)
                        best_cluster_labels = labels_pred
                        best_n_clusters = i
            except ValueError as e:
                print("graph spectral clustering error: ", i, e)
            

        return {
            "graph_based_spectral_best_cluster_labels": best_cluster_labels,
            "graph_based_spectral_best_silhouette_score": best_silhouette_score,
            "graph_based_spectral_best_n_clusters": best_n_clusters,
        }
        
    def get_k_means_clusterings(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
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
        
    def get_hdbscan_clusterings(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
        best_silhouette_score = -100
        best_cluster_labels = None
        min_cluster_size = int(len(img_embedding)/ max_n_clusters)
        min_cluster_size = max(min_cluster_size, 2)
        max_cluster_size = int(len(img_embedding)/ min_n_clusters)
        best_cluster_selection_method = None
        
        for cluster_selection_method in ["eom", "leaf"]:
        
            labels_pred = HDBSCAN(min_cluster_size=min_cluster_size,
                                    max_cluster_size=max_cluster_size,
                                    cluster_selection_method=cluster_selection_method,).fit_predict(img_embedding)
            try: 
                labels_pred_score = silhouette_score(img_embedding, labels_pred)

                if min_n_clusters < len(set(labels_pred)) < max_n_clusters and labels_pred_score > best_silhouette_score:
                    best_silhouette_score = labels_pred_score
                    best_cluster_labels = labels_pred
                    best_cluster_selection_method = cluster_selection_method
            except ValueError as e:
                print(cluster_selection_method, e)

        return {
            "hdbscan_best_cluster_labels": best_cluster_labels,
            "hdbscan_best_silhouette_score": best_silhouette_score,
            "hdbscan_best_cluster_selection_method": best_cluster_selection_method,
        }
        
    def get_affinity_propagation_clusterings(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
        best_silhouette_score = -100
        best_cluster_labels = None
        
        
        labels_pred = AffinityPropagation( max_iter=max_iter).fit_predict(img_embedding)
        try: 
            labels_pred_score = silhouette_score(img_embedding, labels_pred)

            if min_n_clusters < len(set(labels_pred)) < max_n_clusters:
                best_silhouette_score = labels_pred_score
                best_cluster_labels = labels_pred
        except ValueError as e:
                print("affinity proagation error: ",e)

        return {
            "affinity_propagation_best_cluster_labels": best_cluster_labels,
            "affinity_propagation_best_silhouette_score": best_silhouette_score,
        }
        
    def get_agglomerative_clusterings(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
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
        
    def get_spectral_embedding_based_clusterings(self, img_embedding: np.ndarray, min_n_clusters: int, max_n_clusters: int, max_iter: int):
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
        
    def get_nca_embedings(self, img_embedding: np.ndarray, max_iter: int):
        try: 
            result = NeighborhoodComponentsAnalysis(
                n_components=2,
                init = "pca",
                max_iter = max_iter,
                ).fit_transform(img_embedding)
            return result
        except Exception as e:
            print(e)
            return np.zeros((img_embedding.shape[0], 2))
        
    def get_mds_embedings(self, img_embedding: np.ndarray, max_iter: int):
        return MDS(
            n_components=2,
            n_jobs = -1,
            max_iter = max_iter,
        ).fit_transform(img_embedding)
        
    def get_isomap_embedings(self, img_embedding: np.ndarray, max_iter: int):
        return Isomap(
            n_components=2,
            n_jobs = -1,
            max_iter = max_iter,
        ).fit_transform(img_embedding)
        
    def get_tsne_embedings(self, img_embedding: np.ndarray, max_iter: int):
        min_perplexity = min(5, img_embedding.shape[0])
        max_perplexity = min(50, img_embedding.shape[0])
        best_tsne_results = None
        best_perplexity = None
        best_kl_divergence = 1000
        
        for perplexity in range(min_perplexity, max_perplexity):
        
            _tsne = TSNE(
                n_components=2,
                n_jobs = -1,
                max_iter = max_iter,
                n_iter_without_progress=1000,
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
        
    def get_best_clustering_algorithm(self, clustering_result_dict: dict):
        all_silhouette_scores = [
            clustering_result_dict.get("span_tree_connected_components_best_silhouette_score"), 
            clustering_result_dict.get("connected_components_best_silhouette_score"), 
            clustering_result_dict.get("louvain_best_silhouette_score"), 
            clustering_result_dict.get("hierarchical_best_silhouette_score"), 
            clustering_result_dict.get("spectral_clustering_graph_based_best_silhouette_score"), 
            clustering_result_dict.get("kmeans_best_silhouette_score"), 
            clustering_result_dict.get("hdbscan_best_silhouette_score"), 
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
            "hdbscan",
            "affinity_propagation",
            "agglomerative",
            "spectral_embedding_based",
        ]
        
        assert(len(all_clustering_algos) == len(all_silhouette_scores))
        
        clustering_algo_dict = {all_clustering_algos[i]: all_silhouette_scores[i]
                                for i in range(len(all_clustering_algos))}
        
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
        
    def _label_point(self, x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        ax = [ax.annotate( str(point['val']), (point['x']+.02, point['y']), fontsize = 8) for _, point in a.iterrows()]
        adjust_text(ax)
    
    def save_clustering_results(self, clustering_result_dict: dict, result_parent_dir: str):
        assert(os.path.exists(result_parent_dir))
        mi_id = clustering_result_dict.get("mi_id")
        result_dir = os.path.join(result_parent_dir, mi_id)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        result_df = clustering_result_dict["subj_result_df"]
        if len(result_df.columns) != len(set(result_df.columns)):
            print(result_df.head())
            print(result_df.columns)
        result_df = result_df.loc[:,~result_df.columns.duplicated()].copy()
        result_df.to_csv(os.path.join(result_dir, f"clustering_results.csv"))
        
        # for projection_suffix in ["NCA", "ISOMAP", "MDS", "TSNE"]:
        for projection_suffix in ["ISOMAP", "MDS", "TSNE"]:
            fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
            sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
            sns.scatterplot(data=result_df, x=f'{projection_suffix}1', y=f'{projection_suffix}2', hue='best_clustering_labels', palette='hls')
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.title(f'Scatter plot of images using {projection_suffix}');
            plt.xlabel(f'{projection_suffix}1');
            plt.ylabel(f'{projection_suffix}2');
            plt.axis('equal')
            plt.savefig(os.path.join(result_dir,f"{projection_suffix}_best_clustering_results_no_img_label.png"), bbox_inches='tight')
            plt.show()
            plt.clf()
            
            fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
            sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
            sns.scatterplot(data=result_df, x=f'{projection_suffix}1', y=f'{projection_suffix}2', hue='best_clustering_labels', palette='hls')
            self._label_point(result_df[f'{projection_suffix}1'], result_df[f'{projection_suffix}2'], result_df["img_id"], ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.title(f'Scatter plot of images using {projection_suffix}');
            plt.xlabel(f'{projection_suffix}1');
            plt.ylabel(f'{projection_suffix}2');
            plt.axis('equal')
            plt.savefig(os.path.join(result_dir,f"{projection_suffix}_best_clustering_results.png"), bbox_inches='tight')
            plt.show()
            plt.clf()
            
        communities = self.convert_sklearn_labels_to_communities(result_df["best_clustering_labels"])
        
        try:
            draw_communities(clustering_result_dict.get("subj_adjacency_matrix"), communities)
        except AttributeError as e:
            print(e)
            
        plt.savefig(os.path.join(result_dir,f"network_diagram_best_clustering_results.png"), bbox_inches='tight')
        plt.show()
        plt.clf()


    def all_subj_clustering_pipeline(self, result_parent_dir: str):
        assert(os.path.exists(result_parent_dir))
        
        all_subj_best_results_df = [None] * len(self.mi_ids)
        result_timestamp = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        result_dir = os.path.join("/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results", f"clustering-{result_timestamp}")
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        
        self.get_all_subj_metadata()
        for indx, mi_id in tqdm(enumerate(self.mi_ids), total = len(self.mi_ids)):

            subj_data = self.get_subject_data(mi_id=mi_id)
            try:
                clustering_result_dict = self.try_all_clustering_methods(subj_data)
                self.save_clustering_results(clustering_result_dict, result_dir)
                
                subj_best_results_df = pd.DataFrame()
                subj_best_results_df["mi_id"] = clustering_result_dict["mi_id"]
                subj_best_results_df["max_n_clusters"] = clustering_result_dict["max_n_clusters"]
                subj_best_results_df["min_n_clusters"] = clustering_result_dict["min_n_clusters"]
                subj_best_results_df["tsne_best_perplexity"] = clustering_result_dict["tsne_best_perplexity"]
                subj_best_results_df["best_clustering_algorithm"] = clustering_result_dict["best_clustering_algorithm"]
                subj_best_results_df["best_silhouette_score"] = clustering_result_dict["best_silhouette_score"]
                subj_best_results_df["worst_clustering_algorithm"] = clustering_result_dict["worst_clustering_algorithm"]
                subj_best_results_df["worst_silhouette_score"] = clustering_result_dict["worst_silhouette_score"]
                subj_best_results_df["tsne_best_kl_divergence"] = clustering_result_dict["tsne_best_kl_divergence"]

                all_subj_best_results_df[indx] = subj_best_results_df
            except KeyboardInterrupt:
                sys.exit(1)
            except Exception as e:
                print(f"Error occurred while processing MI_ID: {mi_id}")
                print(e)
                
        all_subj_best_results_df = pd.concat(all_subj_best_results_df)
        all_subj_best_results_df.to_csv(os.path.join(result_dir, "all_subj_best_results.csv"), index=None)

# CROP_IMAGE_DIR = os.getenv('CROP_IMAGE_DIR_PATH')
# test_data_id = '09'
# metadata_name = 'meta_data/TWB_ABD_expand_modified_gasex_21072022.csv'
# test_data_list_name = 'fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset'+str(test_data_id)+'/test_dataset'+str(test_data_id)+'.csv'
# test_data_list = pd.read_csv(test_data_list_name)
# meta_data = pd.read_csv(metadata_name, sep=",")

# image_transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.Resize([224, 224]),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ckpt_name = 'model_tl_twbabd'+str(test_data_id)+'/best_results.ckpt'
# image_encoder_id = 'densenet121'
# graph_encoder_id = 'SETNET_GAT'
# num_classes = 2
# input_dim = 1024
# num_layers = 1
# transform = transforms.Compose([
# transforms.Grayscale(num_output_channels=3),
# transforms.Resize([224, 224]),
# transforms.ToTensor(),
# transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# ## Call pretrained image encoder ###
# _, pretrained_image_encoder = models.image_encoder_model(name=image_encoder_id, 
#                                         pretrained=True, 
#                                         num_classes=num_classes, 
#                                         device=device)
# pretrained_image_encoder = pretrained_image_encoder.eval() 


# ## Subject id in the test data ##
# mi_id = test_data_list['MI_ID'][0]
# mi_id = "P0012750"
# ## Obtain image ids and label y for each subject. ###
# img_id_list = ast.literal_eval(meta_data[meta_data['MI_ID']==mi_id]['IMG_ID_LIST'].to_list()[0])
# print(mi_id, len(img_id_list), img_id_list)
# y = 1
# imgs_arr_y = np.arange(0, 25)
# img_id_to_indx = {img_id_list[i]: imgs_arr_y[i] 
#                   for i in range(len(img_id_list))}
# mydata=datasets.single_data_loader(mi_id=mi_id,
#                                 img_id_list=img_id_list,
#                                 image_transform=transform,
#                                 pretrained_image_encoder=pretrained_image_encoder,
#                                 y=y,
#                                 num_classes=num_classes,
#                                 device=device)
# ### Classification ###
# x = mydata.x.to('cpu')


# graph_df = pd.read_csv("/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/07-12-2024-03-57-58/P0012750/encoded_img_corr.csv", index_col= 0)
# graph_np_arr = graph_df.to_numpy()


# communities, frames = louvain_method(graph_np_arr)

# try:
#     draw_communities(graph_np_arr, communities)
# except AttributeError as e:
#     print(e)
    
# def convert_communities_output_to_labels(communities):
#     labels_pred = [None for community in communities for _ in community]
#     for community_indx, community in enumerate(communities):
#         for node in community:
#             labels_pred[node] = community_indx
#     return labels_pred

# def convert_sklearn_labels_to_communities(labels: np.ndarray):
#     # Determine the number of unique communities
#     num_communities = len(np.unique(labels))

#     # Initialize a list of empty sets, one for each community
#     communities = [set() for _ in range(num_communities)]

#     # Populate the communities
#     for indx, group in enumerate(labels):
#         communities[group].add(indx)

#     return communities

      
# nca_results = NeighborhoodComponentsAnalysis(
#     n_components=2,
#     
#     init = "pca",
#     max_iter = 1000,
# ).fit_transform(x, imgs_arr_y)
# df_tsne = pd.DataFrame(nca_results, columns=['nca1', 'nca2'])
# df_tsne['Cluster'] = labels_pred # Add labels column from df_train to df_tsne
# fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
# sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
# sns.scatterplot(data=df_tsne, x='nca1', y='nca2', hue='Cluster', palette='hls')
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
# plt.title(f'Scatter plot of images using Neighborhood Component Analysis');
# plt.xlabel('nca1');
# plt.ylabel('nca2');
# plt.axis('equal')




# embeddings = {
#     "MDS embedding": MDS(n_components=2, n_init=1, max_iter=1000, n_jobs=2),
#     "t-SNE embedding": TSNE(
#         n_components=2,
#         max_iter=10000,
#         n_iter_without_progress=1000,
#         n_jobs=2,
#         
#         perplexity=15,
#     ),
#     "NCA embedding": NeighborhoodComponentsAnalysis(
#         n_components=2, init="pca", random_state=0
#     ),
# }


# projections, timing = {}, {}
# for name, transformer in embeddings.items():

#     data = x

#     print(f"Computing {name}...")
#     start_time = time.perf_counter()
#     projections[name] = transformer.fit_transform(data, imgs_arr_y)
#     timing[name] = time.perf_counter() - start_time
    
# for projection_name, projection in projections.items():
#     x_axis_lab = f'{projection}1'
#     y_axis_lab = f'{projection}2'
#     projection_df = pd.DataFrame(projection, columns=[f'{projection}1', f'{projection}2'])
#     projection_df['Img'] = img_id_list # Add labels column from df_train to df_tsne
#     fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
#     sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
#     sns.scatterplot(data=projection_df, x=x_axis_lab, y=y_axis_lab, hue='Img', palette='hls')
#     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#     plt.title(f'Scatter plot of images using {projection_name}');
#     plt.xlabel(f'{projection_name} 1');
#     plt.ylabel(f'{projection_name} 2');
#     plt.axis('equal')
#     plt.show()
    
# for i in range(5, len(img_id_list)-3):
#     tsne_results = TSNE(
#             n_components=2,
#             max_iter=10000,
#             n_iter_without_progress=1000,
#             n_jobs=2,
#             
#             perplexity=i,
#         ).fit_transform(x)
#     df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
#     df_tsne['Img'] = img_id_list # Add labels column from df_train to df_tsne
#     fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
#     sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
#     sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Img', palette='hls')
#     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#     plt.title(f'Scatter plot of images using t-SNE with perplexity {i}');
#     plt.xlabel('TSNE1');
#     plt.ylabel('TSNE2');
#     plt.axis('equal')

# tsne_results = TSNE(
#         n_components=2,
#         max_iter=10000,
#         n_iter_without_progress=1000,
#         n_jobs=2,
#         
#         perplexity=16,
#     ).fit_transform(x)
# df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
# df_tsne['Img'] = img_id_list # Add labels column from df_train to df_tsne
# fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
# sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
# sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Img', palette='hls')
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
# plt.title(f'Scatter plot of images using t-SNE with perplexity {i}');
# plt.xlabel('TSNE1');
# plt.ylabel('TSNE2');
# plt.axis('equal')


# for i in range(2, 8):
#     kmeans_model = KMeans(n_clusters=i).fit(x)
#     kmeans_labels = kmeans_model.fit_predict(x)
#     df_tsne['Cluster'] = kmeans_labels
#     fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
#     sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
#     sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Cluster', palette='magma')
#     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#     plt.title('Scatter plot of images using KMeans Clustering');
#     plt.xlabel('TSNE1');
#     plt.ylabel('TSNE2');
#     plt.axis('equal')
#     plt.show()
#     plt.close()
#     print("Silhouette score: ", silhouette_score(x, kmeans_labels))
    
    
# from adjustText import adjust_text
# def label_point(x, y, val, ax):
#     a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
#     ax = [ax.annotate( str(point['val']), (point['x']+.02, point['y'])) for _, point in a.iterrows()]
#     adjust_text(ax)
    
# _tsne = TSNE(
#             n_components=2,
#             max_iter=10000,
#             n_iter_without_progress=1000,
#             n_jobs=2,
#             
#             perplexity=20,
#         )
# print(_tsne.kl_divergence_)
# tsne_results = _tsne.fit_transform(x)

# df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
# df_tsne["img_id"] = img_id_list
# df_tsne['Cluster'] = labels_pred # Add labels column from df_train to df_tsne
# fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
# sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
# sns.scatterplot(data=df_tsne, x='tsne1', y='tsne2', hue='Cluster', palette='hls')
# label_point(df_tsne['tsne1'], df_tsne['tsne2'], df_tsne["img_id"], ax)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
# plt.title(f'Scatter plot of images using t-SNE');
# plt.xlabel('tsne1');
# plt.ylabel('tsne2');
# plt.axis('equal')

# # STEPS
# # for every ground truth subj id == 1 and img length > 25
# # obtain correlation matrix and create adjacency numpy matrix 
# # obtain image embedding matrix  
# # perform clustering using 2 ways 
# ## 1 scikit learn image embedding 
# ## 2 graph-based methods using adjacency matrix 
# # record the highest silhouette score and display it using TSNE and other low-dimensional projection methods
# # output the best results into a CSV
# # CSV has columns: mi_id, img_list, best clustering algorithm, best silhouette score, each clustering algorithm_silhouette_score
# ## for algorithms with clusternumber only choose the version with the best cluster 
# ## for each algorithm add a column with a suffix _cluster_num and here we have the number of clusters
# ## output a folder for each subject containing 
# ### best clusterting algorithm results, best clustering algorithm low-dimesional projection method results,
# ### all clustering algorithms results (CSV indicating which images are in which cluster)
# ### all clustering algorithm visualizations 
# ### separate algorithm results by folder perhaps? 


# #don't have spearate df for low-dimensional embeddings, like tsne_Df and mds-df,
# # instead, just have one single df with different columns indicating which method