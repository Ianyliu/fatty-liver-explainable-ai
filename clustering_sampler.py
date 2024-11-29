import ast
import datetime
import gc
import glob
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import islice
from multiprocessing.pool import Pool
from typing import Callable, Dict, Mapping, Optional, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries
from sklearn.metrics import confusion_matrix
from torch_geometric.data import Data
from tqdm import tqdm

from usflc_xai import datasets, models


# pylint: disable=E1136

class MI_ID_INFO(TypedDict):
    clusterings: Union[list[Union[int, str]], tuple[Union[int, str]]]
    img_id_list: Union[list[str], tuple[str]]
    y: int
    img_filepaths: Union[list[str], tuple[str]]
    
class MI_IDS_DICT(TypedDict):
    mi_id: MI_ID_INFO

@dataclass
class Sampler:
    test_data_id: str
    result_dir: str = field(init=False, default=None)
    samples_per_subj: int = 10000
    batch_size: int = 10
    clustering_result_dir: str = "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/clustering-07-29-2024-06-05-39/"
    img_dir: str = field(init=False)
    image_encoder_id: str = 'densenet121'
    graph_encoder_id: str = 'SETNET_GAT'
    mi_id_type: str = "groundtruth-positive"
    mi_id_min_imgs: int = 20
    num_classes: int = 2
    clustering_algorithm: str = "agglomerative"
    num_layers: int = 1
    test_data_dir: str = "/home/liuusa_tw/data/cropped_images/"
    metadata_path: str = '/home/liuusa_tw/twbabd_image_xai_20062024/meta_data/TWB_ABD_expand_modified_gasex_21072022.csv'
    verbose: bool = True
    _pretrained_image_encoder: Callable = field(init=False, default= None)
    _graph_encoder: Callable = field(init=False, default= None)
    input_dim: int = field(init=False, default= None)
    _ckpt_name: str = field(init=False, default= None)
    device: torch.device = field(init=False, default= None)
    checkpoint: dict = field(init=False, default= None)
    _transform: Callable = field(init=False, default= None)
    test_data_path: str = field(init=False, default= None)
    metadata: pd.DataFrame = field(init=False, default= None)
    test_data_list: pd.DataFrame = field(init=False, default= None)
    mi_id_dict: MI_IDS_DICT = field(init=False, default= None)
    mi_ids: list = field(init=False, default= None)
    
    def __post_init__(self):
        load_dotenv()
        self.img_dir = os.getenv('CROP_IMAGE_DIR_PATH')
        if self.result_dir is None: 
            result_timestamp = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
            self.result_dir = os.path.join("/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results",
                                           f"stratified-sampling-{result_timestamp}")
            if not os.path.exists(self.result_dir):
                os.mkdir(self.result_dir)
            print(f"results will be stored in {self.result_dir}")
                
        if self._ckpt_name is None:
            self._ckpt_name = f'/home/liuusa_tw/twbabd_image_xai_20062024/model_tl_twbabd{self.test_data_id}/best_results.ckpt'
            
        if self.test_data_path is None:
            self.test_data_path = f'fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset{self.test_data_id}/test_dataset{self.test_data_id}.csv'
        if self.image_encoder_id == 'resnet50':
            self.input_dim = 2048
        elif self.image_encoder_id == 'vitl16in21k':
            self.input_dim = 768
        elif self.image_encoder_id == 'densenet121':
            self.input_dim = 1024

        self._verify_input()
        
        self._load_data()
        self._load_model()
        self.get_clustering_results()

    def _verify_input(self):
        
        if not isinstance(self.test_data_id, str):
            raise TypeError(f"test_data_id must be type string but is {type(self.test_data_id)} instead")
        if not isinstance(self.result_dir, str):
            raise TypeError(f"result_dir must be type string but is {type(self.result_dir)} instead")
        if not isinstance(self.clustering_result_dir, str):
            raise TypeError(f"clustering_result_dir must be type string but is {type(self.clustering_result_dir)} instead")
        if not isinstance(self._ckpt_name, str):
            raise TypeError(f"_ckpt_name must be type string but is {type(self._ckpt_name)} instead")
        if not isinstance(self.image_encoder_id, str):
            raise TypeError(f"image_encoder_id must be type string but is {type(self.image_encoder_id)} instead")
        if not isinstance(self.graph_encoder_id, str):
            raise TypeError(f"graph_encoder_id must be type string but is {type(self.graph_encoder_id)} instead")
        if not isinstance(self.graph_encoder_id, str):
            raise TypeError(f"graph_encoder_id must be type string but is {type(self.graph_encoder_id)} instead")
        if not isinstance(self.test_data_dir, str):
            raise TypeError(f"test_data_dir must be type string but is {type(self.test_data_dir)} instead")
        if not isinstance(self.metadata_path, str):
            raise TypeError(f"metadata_path must be type string but is {type(self.metadata_path)} instead")
        if not isinstance(self.img_dir, str):
            raise TypeError(f"img_dir must be type string but is {type(self.img_dir)} instead")

        if self.test_data_id not in ['01', '02', '03', '04',
                                      '05', '06', '07', '08', '09', '10']:
            raise ValueError(f"invalid test_data_id {self.test_data_id} is not in" + 
                             "'01', '02', '03', '04', '05', '06', '07', '08', '09', '10'")
        if not os.path.exists(self.result_dir):
            raise ValueError(f"{self.result_dir} does not exist")
        if not os.path.exists(self.clustering_result_dir):
            raise ValueError(f"{self.clustering_result_dir} does not exist")
        if not os.path.exists(self.test_data_dir):
            raise ValueError(f"{self.test_data_dir} does not exist")
        if not os.path.exists(self.img_dir):
            raise ValueError(f"{self.img_dir} does not exist")
        if not os.path.exists(self._ckpt_name) or self._ckpt_name.split('.')[-1].strip() != "ckpt": 
            raise ValueError(f"image_dir {self._ckpt_name} is not a valid checkpoint, please verify the model checkpoint") 
        
        if not isinstance(self.samples_per_subj, int):
            raise TypeError(f"samples_per_subj must be a positive integer but is {type(self.samples_per_subj)} instead")
        if not isinstance(self.batch_size, int):
            raise TypeError(f"batch_size must be a positive integer but is {type(self.batch_size)} instead")
        if not isinstance(self.num_classes, int):
            raise TypeError(f"num_classes must be a positive integer but is {type(self.num_classes)} instead")
        if not isinstance(self.num_layers, int):
            raise TypeError(f"num_layers must be a positive integer but is {type(self.num_layers)} instead")
        if not isinstance(self.mi_id_min_imgs, int):
            raise TypeError(f"mi_id_min_imgs must be a positive integer but is {type(self.mi_id_min_imgs)} instead")
        if not isinstance(self.input_dim, int):
            raise TypeError(f"input_dim must be a positive integer but is {type(self.input_dim)} instead")
        
        if self.image_encoder_id not in ['resnet50', 'densenet121', 'vitl16in21k']:
            raise ValueError(f"image_encoder_id must be following: 'resnet50', 'densenet121', 'vitl16in21k'  \n but is {self.image_encoder_id} instead")
        if self.graph_encoder_id not in ['SETNET_GAT']:
            raise ValueError(f"graph_encoder_id must be SETNET_GAT but is {self.graph_encoder_id} instead")
        if self.mi_id_type not in ['groundtruth-positive', 'groundtruth-negative']:
            raise ValueError(f"mi_id_type {self.mi_id_type} is not a valid mi_id type " + 
                             "\n Valid input: ['groundtruth-positive', 'groundtruth-negative']")
        
        if self.samples_per_subj < 0:
            raise ValueError(f"samples_per_subj must be a positive integer but is {self.samples_per_subj} instead")
        if self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer but is {self.batch_size} instead")
        if self.num_classes < 0:
            raise ValueError(f"num_classes must be a positive integer but is {self.num_classes} instead")
        if self.num_layers < 0:
            raise ValueError(f"num_layers must be a positive integer but is {self.num_layers} instead")
        if self.mi_id_min_imgs < 0:
            raise ValueError(f"mi_id_min_imgs must be a positive integer but is {self.mi_id_min_imgs} instead")
        if self.input_dim < 0:
            raise ValueError(f"input_dim must be a positive integer but is {self.input_dim} instead")
   
    def _setup_image_transform(self):
        # Set up image transformation
        
        self._transform =  transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
    def _load_data(self):
        assert (os.path.exists(self.metadata_path))
        assert (os.path.exists(self.test_data_path))

        self.metadata = pd.read_csv(self.metadata_path, sep=",")
        self.test_data_list = pd.read_csv(self.test_data_path)
        
        metadata_req_cols = ["MI_ID", "liver_fatty", "IMG_ID_LIST"]
        test_data_req_cols = ["MI_ID"]
        
        if not all(col in self.metadata.columns for col in metadata_req_cols):
            raise ValueError(f"Missing required columns in metadata: {metadata_req_cols}" + 
                             f"Please check {self.metadata_path}")
        if not all(col in self.test_data_list.columns for col in test_data_req_cols):
            raise ValueError(f"Missing required columns in testdata: {test_data_req_cols}" + 
                             f"Please check {self.test_data_path}")
            
        if self.verbose:
            print(
                f'There are {len(self.metadata)} subjects in the meta data list.')
            print(
                f'There are {len(self.test_data_list)} subjects in the test dataset{self.test_data_id}.')
            
        self.get_clustering_results()
        
        if self.verbose: 
            print(f"Based on requirements of {self.mi_id_min_imgs} min. images and {self.mi_id_type}," + 
                  f"there are {len(self.mi_ids)} subjects")

    def _load_model(self):
        """Loads model
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert (os.path.exists(self._ckpt_name))

        self._setup_image_transform()

        self._setup_image_encoder()


        ### Call graph encoder ###
        self._graph_encoder, _ = models.encoder_model(name=self.graph_encoder_id,
                                                     input_dim=self.input_dim,
                                                     num_layers=self.num_layers,
                                                     num_classes=self.num_classes,
                                                     device=self.device)
        ### Load trained weights ###
        self.checkpoint = torch.load(self._ckpt_name)
        self._graph_encoder.load_state_dict(self.checkpoint['model_state_dict'])
        self._graph_encoder = self._graph_encoder.eval()
  
    def get_clustering_results(self):
        if self.clustering_result_dir.endswith('/'):
            all_jsons = self.clustering_result_dir + "*/clustering_result_dict.json"
        if self.clustering_result_dir.endswith("\\"):
            all_jsons = self.clustering_result_dir + r"*\clustering_result_dict.json"
        else:
            all_jsons = self.clustering_result_dir + "/*/clustering_result_dict.json"
            
        json_paths = glob.glob(all_jsons)
        all_clustering_result_dict = [self._custom_load_json(json_path) for json_path in json_paths]
        clusterings = [i.get(self.clustering_algorithm).get("best_cluster_labels") for i in all_clustering_result_dict]
        all_img_id_list = [i.get("img_id_list") for i in all_clustering_result_dict]

        mi_ids = [i.get("mi_id") for i in all_clustering_result_dict if i.get("mi_id") is not None]
        ys = [self._get_y(mi_id) for mi_id in mi_ids]
        
        self.mi_id_dict = {mi_id: {
            "img_id_list": all_img_id_list[i],
            "clusterings": clusterings[i],
            "y": ys[i], 
            "img_filepaths": all_img_id_list[i],
            }
            for i, mi_id in enumerate(mi_ids) 
        }
        
        get_img_filepath = lambda mi_id, img_id: os.path.join(self.img_dir, f"{mi_id}_{img_id}.jpg")
        for mi_id, v in self.mi_id_dict.items():
            img_id_list = v.get("img_id_list")
            v["img_filepaths"] = [get_img_filepath(mi_id=mi_id, img_id=img_id) for img_id in img_id_list]
        
        assert all([os.path.exists(filepath) for filepath in v["img_filepaths"] for v in self.mi_id_dict.values()])
        self.mi_ids = mi_ids
        
    def _get_y(self, mi_id: str):
        y = self.metadata[self.metadata['MI_ID']==mi_id]['liver_fatty'].to_list()[0]
        if self.num_classes == 2:
            if y > 0:
                y = 1
        if self.num_classes == 3:
            if y == 3: 
                y = 2
        return y
        

    def _setup_image_encoder(self):
        # Set up image encoder
        _, pretrained_image_encoder = models.image_encoder_model(name=self.image_encoder_id, 
                                        pretrained=True, 
                                        num_classes=self.num_classes, 
                                        device=self.device)
        self._pretrained_image_encoder = pretrained_image_encoder.eval() 
    def _custom_load_json(self, filepath: str):
        with open(filepath) as file_reader:
            file_contents = file_reader.read()
        parsed_json = json.loads(file_contents)
        return parsed_json

    def single_prediction(self, mi_id: str, img_id_list: Union[list[str], tuple[str]], y: int) -> tuple[int]:

        mydata = datasets.single_data_loader(mi_id=mi_id,
                                            img_id_list=img_id_list,
                                            image_transform=self._transform,
                                            pretrained_image_encoder=self._pretrained_image_encoder,
                                            y=y,
                                            num_classes=self.num_classes,
                                            device=self.device)


        x = mydata.x.to(self.device)
        y = mydata.y
        A = mydata.edge_index_corr.to(self.device)
        b = torch.zeros(x.shape[0], dtype=torch.int64).to(self.device)
        train_mask = 1

        h = self._graph_encoder(x, A, b, train_mask)
        _, y_hat = torch.max(h, dim=1)
        y_hat = y_hat.data.to('cpu').numpy()[0]
        
        return y, y_hat

    # def get_cluster_proportions(self, clusterings: Union[list[Union[int, str]], tuple[Union[int, str]]]) -> Dict[Union[int, str]: float]:  
    def get_cluster_proportions(self, clusterings: list[int]):  
        cluster_counts = Counter(clusterings)
        total_images = len(clusterings)
        cluster_proportions = {cluster: count / total_images for cluster, count in cluster_counts.items()}
        return cluster_proportions

    # def calculate_min_sample_size(self, cluster_proportions: Optional[Mapping[int: float]] = None, clusterings: Optional[Union[list[Union[int, str]], tuple[Union[int, str]]]] = None) -> int:
    def calculate_min_sample_size(self, cluster_proportions = None, clusterings = None) -> int:
        if (cluster_proportions is None and clusterings is None) or (cluster_proportions is not None and clusterings is not None):
            raise ValueError("Exactly one of 'cluster_proportions' or 'clusterings' must be provided")
        
        if cluster_proportions is None: 
            cluster_proportions = self.get_cluster_proportions(clusterings)
        
        min_sample_size = math.ceil(1 / min(cluster_proportions.values()))
        return min_sample_size

    # def get_cluster_images_dict(self, clusterings: Union[list[Union[int, str]], tuple[Union[int, str]]], img_id_list: Union[list[str], tuple[str]]) -> Dict[Union[int, str]: str]:
    def get_cluster_images_dict(self, clusterings, img_id_list):
        
        if len(clusterings) != len(img_id_list):
            raise ValueError(f"length of clusterings ({len(clusterings)}) must be equal to length of img_id_list ({len(img_id_list)})")
        
        # Create a dictionary to store image IDs for each cluster
        cluster_images = defaultdict(list)

        for img_id, cluster in zip(img_id_list, clusterings):
            cluster_images[cluster].append(img_id)
            
        return dict(cluster_images)

    # def _get_cluster_sample_size(self, cluster: Union[int, str], proportion: float, sample_size: int, cluster_images_dict: Dict[Union[int, str]: str]) -> int:
    def _get_cluster_sample_size(self, cluster, proportion: float, sample_size: int, cluster_images_dict) -> int:
        return min(
            round(sample_size * proportion), 
            len(cluster_images_dict[cluster])
                )

    # def _sample_from_clusters(self, cluster_sample_sizes: Dict[Union[int, str]: int], cluster_images_dict: Dict[Union[int, str]: str]) -> list[str]:
    def _sample_from_clusters(self, cluster_sample_sizes, cluster_images_dict):
        sample = []
        for cluster, sample_size in cluster_sample_sizes.items():
            sample.extend(
                random.sample(cluster_images_dict[cluster], sample_size)
                )
        return sample

    # def get_clustering_based_samples_for_single_subj(self, clusterings: Union[list[Union[int, str]], tuple[Union[int, str]]], img_id_list: Union[list[str], tuple[str]], num_samples: Optional[int] = None) -> list[list[str]]:
    def get_clustering_based_samples_for_single_subj(self, clusterings,img_id_list, num_samples: Optional[int] = None):
        """Stratified sampling based on clustering results. Samples preserve proportions of ocurrences of clusters. 

        Parameters
        ----------
        num_samples : int
        clusterings : Union[list[Union[int, str]], tuple[Union[int, str]]]
            A list of integers or strings representing which cluster the ith image is at. Each index of the img_id_list should correspond to the index of the clusterings. In other words, at img_id_list[i] belongs to the cluster at clusterings[i].
        img_id_list : Union[list[str], tuple[str]]
            List of image IDs, must be same length as clusterings. Each index of the img_id_list should correspond to the index of the clusterings. In other words, at img_id_list[i] belongs to the cluster at clusterings[i].

        Returns
        -------
        list[list[str]]
            List of samples with image IDs

        Raises
        ------
        ValueError
            If length of clusterings do not match length of image IDs
        ValueError
            If num_samples is a negative integer
        """    
        if num_samples is None:
            num_samples = self.samples_per_subj
            
        if len(clusterings) != len(img_id_list):
            raise ValueError(f"length of clusterings ({len(clusterings)}) must be equal to length of img_id_list ({len(img_id_list)})")
        
        if num_samples < 0:
            raise ValueError(f"'num_samples' must be a non-negative integer, but is {num_samples} instead")
        
        samples = [None] * num_samples
        cluster_proportions = self.get_cluster_proportions(clusterings)
        cluster_images_dict = self.get_cluster_images_dict(clusterings=clusterings,
                                                    img_id_list=img_id_list,)
        min_sample_size = self.calculate_min_sample_size(cluster_proportions)
        max_sample_size = len(clusterings)
        sample_sizes = np.random.randint(min_sample_size, max_sample_size + 1, num_samples)
        
        
        for i, sample_size in enumerate(sample_sizes):
            
            sample = []
            
            cluster_sample_sizes = {cluster: self._get_cluster_sample_size(cluster, proportion, sample_size, cluster_images_dict) 
                                        for cluster, proportion in cluster_proportions.items()}
            
            sample = self._sample_from_clusters(cluster_sample_sizes, cluster_images_dict)
            
            samples[i] = sample
        return samples
        
    # def get_clustering_based_samples_all_subj(self, mi_id_dict: MI_IDS_DICT, num_samples: Optional[int] = None, use_mp: bool = True) -> Dict[str: list[list[str]]]:
    def get_clustering_based_samples_all_subj(self, mi_id_dict: MI_IDS_DICT, num_samples: Optional[int] = None, use_mp: bool = True):
        all_subj_samples = {}
        use_mp = False
        if num_samples is None:
            num_samples = self.samples_per_subj
            
        if self.result_dir is not None:
            completed_subj = [f.name for f in os.scandir(self.result_dir) if f.is_dir()]
            
        if not use_mp:
            for mi_id, v in tqdm(mi_id_dict.items()):
                if mi_id in completed_subj:
                    continue
                clusterings = v.get("clusterings")
                img_id_list = v.get("img_id_list")
                all_subj_samples[mi_id] = self.get_clustering_based_samples_for_single_subj(num_samples=num_samples,
                                                                                    clusterings=clusterings,
                                                                                    img_id_list=img_id_list,
                                                                                    )
            return all_subj_samples

            
        input_items = [(num_samples, v.get("clusterings"), v.get("img_id_list")) for _, v in sorted(mi_id_dict.items())]
        mi_ids = [mi_id for mi_id, _ in mi_id_dict.items()]
        mi_ids = [mi_id for mi_id in mi_ids if mi_id not in completed_subj]
        with Pool(mp.cpu_count() - 1) as pool:
            sampling_results = pool.starmap(self.get_clustering_based_samples_for_single_subj, input_items)
            for indx, result in enumerate(sampling_results):
                mi_id = mi_ids[indx]
                all_subj_samples[mi_id] = result
            
        return all_subj_samples
    
    def predict_on_all_samples(self, mi_id: str, samples: list[list[str]], y: int) -> list[int]:
        predictions = [None] * len(samples)
        for i, sample in enumerate(samples):
            _, y_hat = self.single_prediction(mi_id= mi_id, 
                                   img_id_list=sample, 
                                   y=y)
            predictions[i] = y_hat
            
        return predictions
    
    # def predict_on_all_subj(self, mi_id_dict: MI_IDS_DICT, all_subj_samples: Dict[str: list[list[str]]]) -> Dict[str: Dict[str: list[Union[int, str]]]]:
    def predict_on_all_subj(self, mi_id_dict: MI_IDS_DICT, all_subj_samples):
        results_dict = {mi_id:
            {
                "samples": sample,
                "y_hat": None,
             } for mi_id, sample in all_subj_samples.items()
            }
            
        for i, (mi_id, samples) in tqdm(enumerate(all_subj_samples.items()), total = len(all_subj_samples.keys())):

            predictions = self.predict_on_all_samples(mi_id=mi_id, 
                                               samples=samples, 
                                               y=mi_id_dict[mi_id].get("y"))
            results_dict[mi_id]["y_hat"] = predictions
            
        return results_dict
    
    def get_result_df(self, mi_id: str, img_id_list: list[str], y_hat: int, y: int, samples: list[list[str]]) -> pd.DataFrame:
        num_samples = len(samples)
        num_img = len(img_id_list)
        img_to_indx = {img_id: indx for indx, img_id in enumerate(sorted(img_id_list))}
        indx_to_img = {v: k for k, v in img_to_indx.items()}
        subj_result_dir = os.path.join(self.result_dir, mi_id)
        if not os.path.exists(subj_result_dir):
            os.mkdir(subj_result_dir)
        
        design_matrix = np.zeros((num_samples, num_img), dtype = int)
        for row_indx, sample in enumerate(samples):
            for img_id in sample:
                col_idx = img_to_indx[img_id]
                design_matrix[row_indx, col_idx] = 1
                
        # design_matrix = [tuple(row) for row in design_matrix]
        # design_matrix = np.unique(design_matrix, axis = 0)
                
        df_colnames = [indx_to_img[i] for i in range(design_matrix.shape[1])]
        result_df = pd.DataFrame(design_matrix, columns=df_colnames)
        result_df.to_csv(os.path.join(subj_result_dir,"design_matrix.csv"), index=None)
        result_df['yhat'] = y_hat
        result_df['y'] = np.full(len(y_hat), y)
        result_df.to_csv(os.path.join(subj_result_dir,"pred_results.csv"), index=None)
        return result_df
        
    # def save_results(self, mi_id_dict: MI_IDS_DICT, pred_result_dict: Dict[str: Dict[str: list[Union[int, str]]]], use_mp: bool = True):
    def save_results(self, mi_id_dict: MI_IDS_DICT, pred_result_dict, use_mp: bool = True):

        all_results_df = {}
        
        if use_mp:
            input_items = [(mi_id, 
                           v.get("img_id_list"), 
                           pred_result_dict[mi_id].get("y_hat"), 
                           v.get("y"),
                           pred_result_dict[mi_id].get("samples"),)
                           for mi_id, v in mi_id_dict.items()]

            with Pool(mp.cpu_count() - 1) as pool:
                all_result_df = pool.starmap(self.get_result_df, input_items)
                for indx, result in enumerate(all_result_df):
                    continue
                    
            return 
        
        
        for mi_id, v in tqdm(pred_result_dict.items(), total = len(pred_result_dict.keys())):
            mi_id_dict_v = mi_id_dict.get(mi_id)
            img_id_list = mi_id_dict_v.get("img_id_list")
            # img_filepaths = v.get("img_filepaths")
            samples = v.get("samples")
            y_hat = v.get("y_hat")
            y = mi_id_dict_v.get("y")
            result_df = self.get_result_df(mi_id=mi_id,
                                           img_id_list=img_id_list,
                                           y_hat=y_hat,
                                           y=y,
                                           samples=samples)
        
        return 
        
    def _split_dict_to_chunks(self, data, SIZE=10000):
        it = iter(data)
        for i in range(0, len(data), SIZE):
            yield {k:data[k] for k in islice(it, SIZE)}
        
        
    def sample_predict_pipeline(self, use_mp: bool = True):
        all_subj_samples = self.get_clustering_based_samples_all_subj(mi_id_dict=self.mi_id_dict,
                                                                      num_samples=self.samples_per_subj,
                                                                      use_mp=use_mp)
        for batch in self._split_dict_to_chunks(all_subj_samples, self.batch_size):
            pred_result_dict = self.predict_on_all_subj(mi_id_dict=self.mi_id_dict,
                                                        all_subj_samples=batch,
                                                        )
            self.save_results(mi_id_dict=self.mi_id_dict,
                            pred_result_dict=pred_result_dict,
                            use_mp=use_mp)