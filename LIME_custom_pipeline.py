import ast
import datetime
import glob
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gc
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from PIL import Image
from skimage.segmentation import mark_boundaries
from sklearn.linear_model import HuberRegressor, SGDRegressor
from sklearn.metrics import confusion_matrix
import time
from torch_geometric.data import Data
from tqdm import tqdm
from usflc_xai import datasets, models
import inspect
from typing import Union, Callable
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cbook as cbook
import seaborn as sns

@dataclass
class LIME_subj_pipeline:
    test_data_id: str
    img_list: set
    y: bool
    mi_id: str
    img_dir: str
    pred_func: Callable[[list], int]
    result_parent_dir: str
    image_based: bool = True
    verbose: bool = False

    def __post_init__(self):
        assert (self.test_data_id in ['01', '02', '03', '04',
                                      '05', '06', '07', '08', '09', '10'])
        # assert (isinstance(self.result_dir, str)
        #         and os.path.exists(self.result_dir))
        # assert (isinstance(self.cuda_device_no, int))
        # assert (isinstance(self.image_encoder_id, str) and self.image_encoder_id in [
                # 'resnet50', 'densenet121', 'vitl16in21k'])
        # assert (isinstance(self.graph_encoder_id, str)
                # and self.graph_encoder_id in ['SETNET_GAT'])
        # assert (isinstance(self.cuda_device_no, int)
                # and self.cuda_device_no > -1)
        # assert (isinstance(self.num_classes, int) and self.num_classes > 0)
        # assert (isinstance(self.num_layers, int) and self.num_layers > 0)

        self.__verify_input()
        self.__verify_pred_function_signature()

        self.__load_data()
        self.sample_pred_results = []
        self.samples = []
        self.bootstrap_sample = []
        self.img_list = sorted(self.img_list)
        self.all_img_abs_filepaths = sorted(self.all_img_abs_filepaths)
        self.img_to_indx = {img: indx for indx, img in enumerate(self.img_list)}
        self.indx_to_img = {indx: img for indx, img in enumerate(self.img_list)}
        self.indx_to_abs_filepath = {indx: img for indx, img in enumerate(self.all_img_abs_filepaths)}
        self.result_dir = os.path.join(self.result_parent_dir, f"{self.mi_id}")
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    def __verify_input(self):
        # check test data id is string
        if not isinstance(self.test_data_id, str):
            raise TypeError(f"Test data id must be a string but is {type(self.test_data_id)} instead")

        # check test data id exists

        # check y is int
        if not isinstance(self.y, int):
            raise TypeError(f"Y must be an integer but is {type(self.y)} instead")

        # check type of img_list 
        if not isinstance(self.img_list, list):
            raise TypeError(f"img_list must be a list but is {type(self.img_list)} instead")

        #check if img_list is unique, length is not 0, path exists
        assert len(self.img_list) != 0, "img_list must have length > 0"

        try:
            assert len(self.img_list) == len(set(self.img_list)), "Image list is not unique!"
        except AssertionError as e:
            print(e)
            print("Converting img_list to unique img list")
            self.img_list = list(set(self.img_list))

        # img_dir = os.path.join()
        self.all_img_abs_filepaths = [os.path.join(os.getenv('CROP_IMAGE_DIR_PATH'), str(self.mi_id) + '_' + str(i)+'.jpg') for i in self.img_list]
        assert all([os.path.exists(filepath) for filepath in self.all_img_abs_filepaths])
        
        assert os.path.exists(self.result_parent_dir), f"{self.result_parent_dir} does not exist!"

    def __verify_sampling_input(self, n_samples: int, target_positive_proportion: float, min_sample_prop: float = None, min_sample_size: int = None, max_sample_prop: float = None, max_sample_size: int = None, ):
        num_imgs = len(self.img_list)

        if len(self.negative_img_pool) == 0 and len(self.positive_img_pool) != 0:
            print("All images resulted in correct predictions, sampling may lead to predictable (good) results.")
        
        if len(self.positive_img_pool) == 0 and len(self.negative_img_pool) != 0:
            print("All images resulted in incorrect predictions, sampling may lead to predictable (bad) results.")

        if len(self.negative_img_pool) == 0 and len(self.positive_img_pool) == 0:
            raise ValueError("No images in sampling pool, please run self.__generate_sampling_pool() first")

        min_sample_prop_none_bool = min_sample_prop is not None
        max_sample_prop_none_bool = max_sample_prop is not None
        min_sample_size_none_bool = min_sample_size is not None
        max_sample_size_none_bool = max_sample_size is not None
        
        if min_sample_size_none_bool and min_sample_prop_none_bool:
            raise ValueError("Please only provide either min_sample_size or min_sample_prop to avoid confusion.")
        
        if max_sample_size_none_bool and max_sample_prop_none_bool:
            raise ValueError("Please only provide either max_sample_size or max_sample_prop to avoid confusion.")
        
        if not isinstance(n_samples, int):
            raise TypeError(f"n_samples must be an integer but is instead type {type(n_samples)}")
        
        if not isinstance(target_positive_proportion, float):
            raise TypeError(f"target_positive_proportion must be type float but is instead type {type(target_positive_proportion)}")
        
        if min_sample_prop_none_bool and not isinstance(min_sample_prop, float):
            raise TypeError(f"min_sample_prop must be type float but is instead type {type(min_sample_prop)}")
        
        if max_sample_prop_none_bool and not isinstance(max_sample_prop, float):
            raise TypeError(f"max_sample_prop must be type float but is instead type {type(max_sample_prop)}")
        
        if min_sample_size_none_bool and not isinstance(min_sample_size, int):
            raise TypeError(f"min_sample_size must be type integer but is instead type {type(min_sample_size)}")
        
        if max_sample_size_none_bool and not isinstance(max_sample_size, int):
            raise TypeError(f"max_sample_size must be type integer but is instead type {type(max_sample_size)}")

        if min_sample_prop_none_bool and (min_sample_prop < 0.0 or min_sample_prop > 1.0):
            raise ValueError(f"Minimum sample proportion {min_sample_prop} must be between 0.0 and 1.0")
        
        if min_sample_prop_none_bool and (max_sample_prop < 0.0 or max_sample_prop > 1.0):
            raise ValueError(f"Maximum sample proportion {max_sample_prop} must be between 0.0 and 1.0")
        
        if  min_sample_size_none_bool and (min_sample_size < 0 or min_sample_size > num_imgs):
            raise ValueError(f"Minimum sample size {min_sample_size} must be between 0 and {num_imgs}")

        if  max_sample_size_none_bool and (max_sample_size < 0 or max_sample_size > num_imgs):
            raise ValueError(f"Maximum sample size {max_sample_size} must be between 0 and {num_imgs}")

        if min_sample_prop_none_bool and max_sample_prop_none_bool and min_sample_prop > max_sample_prop: 
            raise ValueError(f"Specified minimum sample proportion {min_sample_prop} must be less than or equal to specified maximum sample proportion {max_sample_prop}")
        if min_sample_size_none_bool and max_sample_size_none_bool and min_sample_size > max_sample_size: 
            raise ValueError(f"Specified minimum sample size {min_sample_size} must be less than or equal to specified maximum sample size {max_sample_size}")
        
    def __load_data(self):
        pass

    def __verify_pred_function_signature(self):
        sig = inspect.signature(self.pred_func)
        parameters = sig.parameters
        num_req_params = 0
        desired_num_of_req_params = 1
        
        for param in parameters.values():
            if param.default == inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_POSITIONAL and param.kind != inspect.Parameter.VAR_KEYWORD:
                num_req_params += 1
                

        if desired_num_of_req_params != num_req_params:
            print(f"WARNING: pred_func # of required parameters {num_req_params} out of {parameters} total parameters do not match expected number of required parameters {desired_num_of_req_params}")
        
        # Test the function with all images
        test_output = self.pred_func(self.img_list)

        # Verify function output type 
        if not isinstance(test_output, (int)):
            raise TypeError(f"test_output must be type int but is {type(test_output)} instead")


    def predict_on_random_samples_until_convergence(self, n_samples: int, target_positive_proportion: float, min_sample_prop: float = None, min_sample_size: int = None, max_sample_prop: float = None, max_sample_size: int = None, max_iter: int = 100000, append_to_original: bool = True):
        self.__predict_on_all_single_img()
        # pos_img_pool_none_bool = self.positive_img_pool is None or len(self.positive_img_pool) == 0 or not isinstance(self.positive_img_pool, list)
        # neg_img_pool_none_bool = self.negative_img_pool is None or len(self.negative_img_pool) == 0 or not isinstance(self.negative_img_pool, list)
        # if pos_img_pool_none_bool or neg_img_pool_none_bool:
        #     self.__generate_sampling_pool()
        self.__generate_sampling_pool()

        num_imgs = len(self.img_list)
        all_imgs_pos_bool = len(self.positive_img_pool) == num_imgs and len(self.negative_img_pool) == 0
        all_imgs_neg_bool = len(self.positive_img_pool) == 0 and len(self.negative_img_pool) == num_imgs
        resample_bool = False
        if all_imgs_pos_bool or all_imgs_neg_bool:
            print("Class balance cannot be guaranteed due to all single images belonging to one class")
            resample_bool = True

        self.__verify_sampling_input(n_samples=n_samples, min_sample_prop=min_sample_prop, min_sample_size=min_sample_size, max_sample_prop=max_sample_prop, max_sample_size=max_sample_size, target_positive_proportion=target_positive_proportion)

        if not isinstance(target_positive_proportion, float):
            raise TypeError(f"target_positive_proportion must be a float but got {type(target_positive_proportion)}")
        
        if not isinstance(max_iter, int):
            raise TypeError(f"max_iter must be a int but got {type(max_iter)}")
        
        if target_positive_proportion < 0.0 or target_positive_proportion > 1.0:
            raise ValueError(f"target_positive_proportion must be between 0 and 1 but is {target_positive_proportion}")
        
        
        if max_iter <= 0:
            raise ValueError(f"max_iter must be greater than 0 but is {max_iter}")

        if n_samples == 0:
            raise ValueError(f"n_samples must be greater than 0 but got {n_samples}")
        
        if n_samples >= max_iter:
            raise ValueError(f"n_samples must be less than max_iter but got {n_samples} samples and {max_iter} iterations as parameters. (Each iteration generates one sample)")

         # Calculate target positive count
        target_positive_count = int(n_samples * target_positive_proportion)
        target_negative_count = n_samples - target_positive_count
        current_positive_count = 0
        current_negative_count = 0
        if self.samples is None:
            self.samples = []
        if self.sample_pred_results is None:
            self.sample_pred_results = []
        
        if min_sample_size is None and min_sample_prop is not None:
            min_sample_size = int(num_imgs * min_sample_prop)

        if max_sample_size is None and max_sample_prop is not None:
            max_sample_size = int(num_imgs * max_sample_prop)
            
        iterations = 0
        # Randomly choose sample sizes between min and max
        sample_sizes = np.random.randint(min_sample_size, max_sample_size + 1, size = n_samples)
        # Random sampling from each pool based on sample sizes
        samples = [np.random.choice(self.img_list, sample_size, replace=False) for sample_size in sample_sizes]
        
        while iterations < max_iter and current_positive_count < target_positive_count and current_negative_count < target_negative_count:
            sample = samples[iterations]
            y_hat = self.pred_func(sample)
            if y_hat != self.y:
                current_negative_count += 1
            else:
                current_positive_count += 1
                
            self.samples.append(sample)
            self.sample_pred_results.append(y_hat)

            iterations += 1

            if self.verbose and iterations % 200 == 0:
                print(f"Iteration (= # samples) {iterations}: Current positive proportion = {current_positive_count/ (current_negative_count + current_positive_count):.4f}")
                                
        num_current_samples = len(self.samples)
        sample_sizes = np.random.randint(min_sample_size, max_sample_size + 1, size = n_samples)
        while resample_bool and iterations < max_iter and num_current_samples < n_samples:
            
            sample = samples[iterations]
            y_hat = self.pred_func(sample)
            if y_hat != self.y:
                current_negative_count += 1
            else:
                current_positive_count += 1
                
            self.samples.append(sample)
            self.sample_pred_results.append(y_hat)
            
            iterations += 1
            num_current_samples += 1
            if self.verbose and iterations % 200 == 0:
                print(f"Iteration (= # samples) {iterations}: Current positive proportion = {current_positive_count/ (current_negative_count + current_positive_count):.4f}")
                        
        enough_positive_bool = current_positive_count >= target_positive_count
        enough_negative_bool = current_negative_count >= target_negative_count
        iterations_below_max_iter_bool = iterations < max_iter
        
        if enough_positive_bool and enough_negative_bool:
            if self.verbose:
                print(f"Finished after {iterations} iterations")
        elif not iterations_below_max_iter_bool:
            print(f"Maximum iterations reached. Final positive proportion: {sum(self.sample_pred_results)/ len(self.sample_pred_results):.4f}")            
        elif resample_bool:
            if self.verbose: 
                print('Resampling was skipped')
        elif enough_positive_bool and not enough_negative_bool and iterations_below_max_iter_bool:
            # Positive threshold met, negative threshold not met
            if self.verbose:
                print(f"Enough positive samples {target_positive_count}, using negative sampling pool to increase negative samples")
            _ = self.__only_sample_negatives(n_extra_samples = target_negative_count - current_negative_count,
                                         min_sample_size = min_sample_size,
                                         max_sample_size = max_sample_size,
                                         )
            iterations += target_negative_count - current_negative_count
            
        elif not enough_positive_bool and enough_negative_bool and iterations_below_max_iter_bool:
            # Positive threshold met, negative threshold not met
            if self.verbose:
                print(f"Enough negative samples {target_negative_count}, using positive sampling pool to increase positive samples")
            _ = self.__only_sample_positives(n_extra_samples = target_negative_count - current_negative_count,
                                         min_sample_size = min_sample_size,
                                         max_sample_size = max_sample_size,
                                         )
            
            iterations = target_negative_count - current_negative_count
                    
        if iterations >= max_iter:
            # iterations maxxed out and neither positive nor negative samples were enough 
            if self.verbose:
                print(f"Maximum iterations reached. Final positive proportion: {sum(self.sample_pred_results)/ len(self.sample_pred_results):.4f}")
        
        if self.verbose:
            if len(self.samples) != n_samples or len(self.sample_pred_results) != n_samples or len(self.samples) != len(self.sample_pred_results):
                print(f"WARNING: Number of samples {len(self.samples)} or length of pred results {len(self.sample_pred_results)} and desired number of sample {n_samples} don't match")      
                  
            print(f"number of total samples: {len(self.samples)} " + 
                # f"number of unique samples: {len(set(self.samples))}" + ## fixme todo later 
                f"Final positive proportion: {sum(self.sample_pred_results)/ len(self.sample_pred_results):.4f}")
        
        return self.samples

    def __only_sample_positives(self, n_extra_samples: int,min_sample_size: int, max_sample_size: int):
        if self.positive_img_pool is None or len(self.positive_img_pool)  == 0:
            print("positive_img_pool is empty, generating positive samples")
            self.__generate_sampling_pool()
            
        return self.generate_balanced_random_samples(n_samples = n_extra_samples, 
                                              target_positive_proportion = 0.85, 
                                              min_sample_size=min_sample_size,
                                              max_sample_size= max_sample_size,
                                              pred_on_samples=True,
                                              )
        
    def __only_sample_negatives(self,n_extra_samples: int,min_sample_size: int, max_sample_size: int):
        if self.negative_img_pool is None or  len(self.negative_img_pool) == 0:
            print("negative_img_pool is empty, generating negative samples")     
            self.__generate_sampling_pool()
            
        return self.generate_balanced_random_samples(n_samples = n_extra_samples, 
                                              target_positive_proportion = 0.15, 
                                              min_sample_size=min_sample_size,
                                              max_sample_size= max_sample_size,
                                              pred_on_samples=True,
                                              )       
        
    
    def generate_random_samples_until_convergence(self, n_samples: int, target_positive_proportion: float, min_sample_prop: float = None, min_sample_size: int = None, max_sample_prop: float = None, max_sample_size: int = None, max_iter: int = 100000, append_to_original: bool = False, store_samples: bool = True, convergence_threshold: float = 0.05, adjust_to_n_samples: bool = True):
        self.__predict_on_all_single_img()
        pos_img_pool_none_bool = self.positive_img_pool is None or len(self.positive_img_pool) == 0 or not isinstance(self.positive_img_pool, list)
        neg_img_pool_none_bool = self.negative_img_pool is None or len(self.negative_img_pool) == 0 or not isinstance(self.negative_img_pool, list)
        if pos_img_pool_none_bool or neg_img_pool_none_bool:
            self.__generate_sampling_pool()

        self.__verify_sampling_input(n_samples=n_samples, min_sample_prop=min_sample_prop, min_sample_size=min_sample_size, max_sample_prop=max_sample_prop, max_sample_size=max_sample_size, target_positive_proportion=target_positive_proportion)

        pos_img_pool_none_bool = self.positive_img_pool is None or len(self.positive_img_pool) == 0 or not isinstance(self.positive_img_pool, list)
        neg_img_pool_none_bool = self.negative_img_pool is None or len(self.negative_img_pool) == 0 or not isinstance(self.negative_img_pool, list)
        resample_pool_bool = True
        if len(self.positive_img_pool) == len(self.img_list) or len(self.positive_img_pool) == len(self.img_list):
            # this means that there samples are either all positive or all negative 
            resample_pool_bool = False

        if not isinstance(target_positive_proportion, float):
            raise TypeError(f"target_positive_proportion must be a float but got {type(target_positive_proportion)}")
        
        if not isinstance(convergence_threshold, float):
            raise TypeError(f"convergence_threshold must be a float but got {type(convergence_threshold)}")
        
        if not isinstance(max_iter, int):
            raise TypeError(f"max_iter must be a int but got {type(max_iter)}")
        
        if target_positive_proportion < 0.0 or target_positive_proportion > 1.0:
            raise ValueError(f"target_positive_proportion must be between 0 and 1 but is {target_positive_proportion}")
        
        if convergence_threshold < 0.0 or convergence_threshold > 1.0:
            raise ValueError(f"convergence_threshold must be between 0 and 1 but is {convergence_threshold}")
        
        if max_iter < 0:
            raise ValueError(f"max_iter must be greater than 0 {max_iter}")

        if n_samples == 0:
            return
        
        if max_iter == 0:
            return

        samples = []
        iterations = 0
        current_positive_proportion = 0
        num_imgs = len(self.img_list)
        
        if min_sample_size is None and min_sample_prop is not None:
            min_sample_size = int(num_imgs * min_sample_prop)

        if max_sample_size is None and max_sample_prop is not None:
            max_sample_size = int(num_imgs * max_sample_prop)

        while iterations < max_iter and abs(current_positive_proportion - target_positive_proportion) > convergence_threshold:
            # Randomly choose a sample size between min and max
            sample_size = np.random.randint(min_sample_size, max_sample_size + 1)

            # Random sampling from each pool
            sample = np.random.choice(self.img_list, sample_size, replace=False)
            
            # Calculate the proportion of positive samples in this sample
            sample_positive_count = np.sum(1 for img in sample if img in self.positive_img_pool)
            sample_positive_proportion = sample_positive_count / sample_size

            samples.append(sample)
            
            # Update the current overall positive proportion
            total_positive_count = sum(sum(1 for img in s if img in self.positive_img_pool) for s in samples)
            total_sample_size = sum(len(s) for s in samples)
            current_positive_proportion = total_positive_count / total_sample_size

            iterations += 1

            if self.verbose and iterations % 1000 == 0:
                print(f"Iteration {iterations}: Current positive proportion = {current_positive_proportion:.4f}")

        if iterations == max_iter and self.verbose:
            print(f"Maximum iterations reached. Final positive proportion: {current_positive_proportion:.4f}")
        elif self.verbose:
            print(f"Converged after {iterations} iterations. Final positive proportion: {current_positive_proportion:.4f}")

        if self.verbose:
            print(f"Number of samples generated: {len(samples)}")

        if append_to_original:
            samples = self.samples + samples
        
        if store_samples:
            self.samples = samples

        return samples

    def generate_balanced_random_samples(self, n_samples: int, target_positive_proportion: float, min_sample_prop: float = None, min_sample_size: int = None, max_sample_prop: float = None, max_sample_size: int = None, pred_on_samples = False):
        if self.single_img_results is None or len(self.single_img_results) <= 0 or not isinstance(self.single_img_results, dict): 
            self.__predict_on_all_single_img()
            
        pos_img_pool_none_bool = self.positive_img_pool is None or len(self.positive_img_pool) == 0 or not isinstance(self.positive_img_pool, list)
        neg_img_pool_none_bool = self.negative_img_pool is None or len(self.negative_img_pool) == 0 or not isinstance(self.negative_img_pool, list)
        if pos_img_pool_none_bool or neg_img_pool_none_bool:
            self.__generate_sampling_pool()
        num_imgs = len(self.img_list)

        self.__verify_sampling_input(n_samples=n_samples, target_positive_proportion=target_positive_proportion, min_sample_prop=min_sample_prop, min_sample_size=min_sample_size, max_sample_prop=max_sample_prop, max_sample_size=max_sample_size)

        # Calculate min and max sample sizes

        if min_sample_size is None and min_sample_prop is not None:
            min_sample_size = int(num_imgs * min_sample_prop)

        if max_sample_size is None and max_sample_prop is not None:
            max_sample_size = int(num_imgs * max_sample_prop)

        if n_samples == 0:
            return

        if not 0.0 <= target_positive_proportion <= 1.0: 
            raise ValueError(f"Class ratio {target_positive_proportion} must be between 0.0 and 1.0")
        
        
        # if append_to_original and len(self.samples) == 0:
        #     raise ValueError("Cannot append to random samples because there were no existing samples found")
        
        # Calculate all possible number of samples
        num_all_possible_samples = np.sum([np.math.comb(len(self.img_list), i) for i in range(min_sample_size, max_sample_size + 1)])

        # Limit n_samples
        original_n_samples = n_samples
        n_samples = min(n_samples, num_all_possible_samples)

        if n_samples < original_n_samples:
            print(f"Warning: n_samples was reduced from {original_n_samples} to {n_samples} to match the number of all possible unique samples.")

        # Reset the random sample to nothing
        samples = [None] * n_samples
        if pred_on_samples:
            sample_pred_results = [None] * n_samples
        
        for indx in tqdm(range(n_samples)):
            # Randomly choose a sample size between min and max
            sample_size = np.random.randint(min_sample_size, max_sample_size + 1)

            # Calculate the number of samples needed from each pool
            positive_samples_needed = int(sample_size * target_positive_proportion)
            negative_samples_needed = sample_size - positive_samples_needed

            # Adjust if we don't have enough samples in either pool
            positive_samples_needed = min(positive_samples_needed, len(self.positive_img_pool))
            negative_samples_needed = min(negative_samples_needed, len(self.negative_img_pool))

            # Random sampling from each pool
            positive_samples = np.random.choice(self.positive_img_pool, positive_samples_needed, replace=False)
            negative_samples = np.random.choice(self.negative_img_pool, negative_samples_needed, replace=False)

            # Combine samples
            sample = list(positive_samples) + list(negative_samples)
            samples[indx] = sample
            
            if pred_on_samples:
                y_hat = self.pred_func(sample)
                sample_pred_results[indx] = y_hat
        
        if self.verbose:
            print(f"Appending {len(samples)} samples to {len(self.samples)} samples")
        if pred_on_samples:
            if self.verbose:
                print(f"Appending {len(sample_pred_results)} sample prediction results to {len(self.sample_pred_results)} sample prediction results")
            if self.sample_pred_results is not None:
                self.sample_pred_results += sample_pred_results
            else:
                self.sample_pred_results = sample_pred_results
                            
        if self.samples is None: 
            self.samples = samples       
        else:
            self.samples += samples
            
        
        return self.samples

    def __predict_on_all_single_img(self):
        self.single_img_results = {img: self.pred_func([img]) for img in tqdm(self.img_list)}
        if self.verbose:
            num_correct = len([i for i in self.single_img_results.values() if i == self.y])
            accuracy =  num_correct / len(self.single_img_results)
            print(f"Accuracy on all single images: {accuracy}, # correct: {num_correct}, total # of imgs: {len(self.img_list)}")

    def __generate_sampling_pool(self):
        assert len(self.single_img_results) != 0, "Single image prediction results do not exist. Try running self.__predict_on_all_single_img() first"
        assert len(self.single_img_results) == len(self.img_list), f"Some image predictions were not complete. Num of images: {len(self.img_list)}, num of results: {len(self.single_img_results)}"
        self.negative_img_pool = {k for k,v in self.single_img_results.items() if v != self.y}
        self.positive_img_pool = set(self.img_list) - self.negative_img_pool

        self.negative_img_pool = list(self.negative_img_pool)
        self.positive_img_pool = list(self.positive_img_pool)

        assert (len(self.negative_img_pool) + len(self.positive_img_pool)) == len(self.img_list), "Negative image pool and positive image pools are incomplete, excluding some images" 

    def generate_pred_results_matrix(self):
        n_images = len(self.img_list)
        sample_len = len(self.samples)
        assert self.samples is not None 
        assert self.sample_pred_results is not None
        assert len(self.samples) == len(self.sample_pred_results)
        self.pred_results_x = np.zeros((sample_len, n_images), dtype = int)
        self.pred_results_yhat = np.array(self.sample_pred_results)
        # self.pred_results_y = np.full(shape = sample_len, fill_value=self.y)
        
        for row_idx, sample in enumerate(self.samples):
            for img in sample:
                col_idx = self.img_to_indx[img]
                self.pred_results_x[row_idx, col_idx] = 1
                
        if np.isnan(self.pred_results_x).any():
            print('there are nans in pred_results_x, replacing them with 0')
            self.pred_results_x = np.nan_to_num(self.pred_results_x)

    def get_imgs_marginal_relation(self, corr_type: str = 'pearson', conf_level: float = 0.95):
        self.generate_pred_results_matrix()
        n_imgs = len(self.indx_to_img)
        img_pred_corr = {}
        # all_corr = [None] * n_imgs
        # all_corr_std = [None] * n_imgs
        # all_corr_CI = [None] * n_imgs
        # all_corr_p_val = [None] * n_imgs
        self.indx_to_corr = {}
        self.indx_to_corr_CI = {}
        self.indx_to_corr_upper_CI = {}
        self.indx_to_corr_lower_CI = {}
        self.indx_to_corr_p_val = {}
        self.indx_to_corr_std = {}
        
        for img_indx in self.indx_to_img:
            
            corr, corr_std, corr_CI, corr_p_val =self.calculate_img_corr(img_indx, corr_type)
            if corr_CI is not None:
                corr_lower_CI, corr_upper_CI = corr_CI
            else:
                corr_lower_CI, corr_upper_CI = None, None
            self.indx_to_corr[img_indx] = corr
            self.indx_to_corr_std[img_indx] = corr_std
            self.indx_to_corr_p_val[img_indx] = corr_p_val
            self.indx_to_corr_CI[img_indx] = corr_CI
            self.indx_to_corr_lower_CI[img_indx] = corr_lower_CI
            self.indx_to_corr_upper_CI[img_indx] = corr_upper_CI
            
            img_pred_corr[img_indx] = {
                "corr": corr,
                "corr_std": corr_std,
                "corr_CI": corr_CI,
                "corr_upper_CI": corr_upper_CI,
                "corr_lower_CI": corr_lower_CI,
                "corr_p_val": corr_p_val,
            }
            
        self.img_pred_corr = img_pred_corr
        
        return self.img_pred_corr

    def get_img_marginal_relation(self):
        pass
    
    def calculate_img_corr(self, img, corr_type: str = 'pearson'):
        if isinstance(img, str):
            img_indx = self.img_to_indx[img]
        if isinstance(img, int):
            assert img in self.indx_to_img.keys()
            img_indx = img
            
        img_col = self.pred_results_x[:, img_indx].copy()
        n = len(img_col)
        if corr_type == 'pearson' or corr_type == 'p':
            unique_img_col_val = np.unique(img_col)
            unique_pred_results_y_val = np.unique(self.pred_results_yhat)
            
            corr, corr_p_val = pearsonr(img_col, self.pred_results_yhat)
            corr_std = np.sqrt((1- corr ** 2) / (n-2))
            upper_CI = min(corr + corr_std, 1.0)
            lower_CI = max(corr - corr_std, -1.0)
            corr_CI = (lower_CI, upper_CI)
            # if not isinstance(corr, (int, float)):
            if len(unique_img_col_val) == 1:
                print(f"WARNING: {self.mi_id} sampling results yielded only unique values for an image column")
                print(f"Unique img_col values: {unique_img_col_val}")
                corr = 0 
                corr_std = 0
                corr_CI = 0
                corr_p_val = 0
                
            if len(unique_pred_results_y_val) == 1:
                
                print(f"WARNING: {self.mi_id} sampling results yielded only unique values for y_hat")
                print(f"Unique y_hat values: {unique_pred_results_y_val}")
                corr, corr_std, corr_CI, corr_p_val = 0, None, None, None 
        elif corr_type == 'matthews' or corr_type == 'mcc':
            true = np.full(self.pred_results_yhat.shape, self.y, dtype=int)
            corr = matthews_corrcoef(true, self.pred_results_yhat)
            corr_std = None
            corr_CI = None
            corr_p_val = None
        elif corr_type == 'spearman' or corr_type == 's':
            corr_result = spearmanr(img_col, self.pred_results_yhat)
            corr = corr_result.correlation
            corr_CI = None
            corr_std = None
            corr_p_val = corr_result.pvalue
            
        return corr, corr_std, corr_CI, corr_p_val

    def plot_results(self):
        pass

    def plot_sampling_corr_heatmap(self):
        self.sampling_corr = np.corrcoef(self.pred_results_x.T)
        
        heatmap_labels = [self.indx_to_img[indx] for indx, _ in enumerate(self.pred_results_x.T)]
        sns.heatmap(self.sampling_corr, 
                    annot=False, 
                    xticklabels=heatmap_labels,
                    yticklabels=heatmap_labels,)
        plt.savefig(os.path.join(self.result_dir,
                        f"image-sampling-heatmap.png"), bbox_inches='tight')
        plt.show()
        plt.clf()
        
    def create_sampling_corr_df(self):
        self.sampling_corr = np.corrcoef(self.pred_results_x.T)
        colnames = [self.indx_to_img[indx] for indx, _ in enumerate(self.pred_results_x.T)]
        
        return pd.DataFrame(self.sampling_corr, columns=colnames)
        
    def plot_img_corr_heatmap(self, image_encoder, transform, device):
        assert self.all_img_abs_filepaths is not None 
        
        img_list = []
        for img_filepath in self.all_img_abs_filepaths:

            img_list.append(transform(Image.open(img_filepath)))
            
        images = torch.stack(img_list, dim=0).to(device)
        self.encoded_images = images
                                            
        x = image_encoder(images).detach()
        corr_x = torch.corrcoef(x)
        np_corr_x = corr_x.cpu().numpy()
        self.encoded_img_corr = np_corr_x.copy()
        ticklabels = [i.split('/')[-1].split('.')[0] for i in self.all_img_abs_filepaths]
        self.encoded_corr_df_index = ticklabels
        sns.heatmap(np_corr_x, xticklabels=ticklabels, yticklabels=ticklabels,
                    annot = True)
        plt.savefig(os.path.join(self.result_dir,
                        f"image-encoded-corr-heatmap.png"), bbox_inches='tight')
        plt.show()
        plt.clf()
    
    def plot_image_bar_plot(self):
        fig = plt.figure(figsize=(15,40)) 
        
        def offset_image(x, y, img_path, bar_is_too_short, ax, zoom=0.06, vertical=False):
            img = plt.imread(img_path)
                
            im = OffsetImage(img, zoom=zoom, cmap='gray')
            im.image.axes = ax
            

            if vertical:
                # if y >= 0:
                #     xybox = (0, 10)  # Offset above the bar for positive values
                # else:
                #     xybox = (0, -10)  # Offset below the bar for negative values
                    
                if bar_is_too_short:
                    y = 0
                y_offset = -25 if y >= 0 else 25  # Adjust offset based on bar direction
                xybox = (0, y_offset)
            else:
                if bar_is_too_short:
                    x = 0
                x_offset = -25 if x >= 0 else 25  # Adjust offset based on bar direction
                xybox = (x_offset, 0)
                
            ab = AnnotationBbox(im, (x, y), xybox=xybox, frameon=False,
                                xycoords='data', boxcoords="offset points", pad=0)

            ax.add_artist(ab)

        labels = self.img_list.copy()
        values = [self.indx_to_corr[self.img_to_indx[i]] 
                  for i in labels
                  ]
        
        label_val = [(val, label) for val, label in sorted(zip(values, labels), 
                                              key = lambda pair: pair[0], 
                                              reverse= True)]
        values, labels = zip(*label_val)
        del label_val
        
        colors = ["deepskyblue" 
                  if val > 0 
                  else "salmon" 
                  for val in values]
        
        zoom = 0.065
        # if len(values) >= 22:
        #     zoom = 0.06
        # elif 22 > len(values) >= 10:
        #     zoom = 0.08
        # else:
        #     zoom = 0.15

        height = 0.8
        
        stds = None
        if None not in set(self.indx_to_corr_std.values()):
            stds = [self.indx_to_corr_std[self.img_to_indx[i]] for i in labels]
            bar_labels = [f"{values[indx]:.2f}±{std:.3f}" for indx, std in enumerate(stds)]
            
            for indx, val in enumerate(values):
                plt.text(val, indx, bar_labels[indx],
                         va='center',
                         )
        else:
        
            for indx, val in enumerate(values):
                plt.text(val, indx, f"{val:.2f}",
                         va='center',
                         )
                
        plt.barh(y=labels, width=values, 
                 height=height, color=colors, 
                 align='center', alpha=0.7, 
                 xerr = stds, ecolor='silver',
                 error_kw=dict(lw=3,),
                )

        if isinstance(values, np.ndarray):
            max_value = values.max()
        elif isinstance(values, (list, tuple, set)):
            max_value = max(values)
            
        ax = plt.gca()
        for _, (label, value) in enumerate(zip(labels, values)):
            img_indx = self.img_to_indx[label]
            img_abs_filepath = self.indx_to_abs_filepath[img_indx]
            offset_image(x = value, 
                         img_path = img_abs_filepath, 
                         y = label, 
                         bar_is_too_short=value < max_value / 10, 
                         zoom=zoom,
                         ax=ax,)
        plt.subplots_adjust(left=0.15)
        fig.set_tight_layout(True)
        fig.patch.set_facecolor('white')
        fig.savefig(os.path.join(self.result_dir,
                        f"hbar.png"), bbox_inches='tight')
        plt.show()
        plt.clf()
        
        fig_width = len(labels) + len(labels)/4
        fig_width = max(fig_width, 13)
        fig_height = len(labels) // 2
        fig_height = max(fig_height, 8)
        fig = plt.figure(figsize=(fig_width,fig_height))            
        
        ### bar container trial
        # fig, ax = plt.subplots()
        # fig.set_size_inches(30, 150)
        # # bar_container = ax.bar(labels, values, width=0.8, color=colors)
        # bar_container = ax.bar(labels, values, width=0.8, color=colors, align='center', alpha=0.8)
        # ax.set(ylabel='Correlations', title='Image & Prediction Result Correlation')
        # ax.bar_label(bar_container, fmt='{:,.0f}')
        
        stds = None
        if None not in set(self.indx_to_corr_std.values()):
            stds = [self.indx_to_corr_std[self.img_to_indx[i]] for i in labels]
            bar_labels = [f"{values[indx]:.2f}\n±{std:.3f}" for indx, std in enumerate(stds)]
            
            for indx, val in enumerate(values):
                plt.text(indx, val, 
                         bar_labels[indx], 
                         ha='center', 
                         ) 
        else:

            for indx, val in enumerate(values):
                plt.text(indx, val, 
                         f"{val:.2f}",
                         ha='center', 
                         ) 

        zoom = 0.25 / 4
        plt.bar(x=labels, height=values, 
                width=0.8, color=colors, 
                align='center', alpha=0.8, 
                yerr=stds, ecolor='lightgray', 
                error_kw=dict(lw=3,),
                )
        ax = plt.gca()
            
        for _, (label, value) in enumerate(zip(labels, values)):
            img_indx = self.img_to_indx[label]
            img_abs_filepath = self.indx_to_abs_filepath[img_indx]
            offset_image(y = value, img_path = img_abs_filepath, 
                         x = label, 
                         bar_is_too_short=value < max_value / 10, 
                         ax=ax, 
                         zoom=zoom, 
                         vertical=True)
        plt.subplots_adjust(left=0.15)
        fig.set_tight_layout(True)
        fig.patch.set_facecolor('white')
        fig.savefig(os.path.join(self.result_dir,
                        f"vbar.png"), bbox_inches='tight')
        plt.show()
        plt.clf()

    def save_results(self):
        assert self.result_dir is not None
        if self.indx_to_img is not None and self.pred_results_x is not None and self.pred_results_yhat is not None:
            df_colnames = [self.indx_to_img[i] for i in range(self.pred_results_x.shape[1])]
            result_df = pd.DataFrame(self.pred_results_x, columns=df_colnames)
            result_df.to_csv(os.path.join(self.result_dir,"design_matrix.csv"), index=None)
            result_df['yhat'] = self.pred_results_yhat
            result_df['y'] = np.full(len(self.pred_results_yhat), self.y)
            result_df.to_csv(os.path.join(self.result_dir,"pred_results.csv"), index=None)
        if self.img_pred_corr is not None:
            corr_df_index = [self.indx_to_img[i] for i, _ in enumerate(sorted(self.img_pred_corr))]
            corr_data = [val for _, val in sorted(self.img_pred_corr.items())]
            corr_df = pd.DataFrame(corr_data, index = corr_df_index)
            corr_df.to_csv(os.path.join(self.result_dir, "corr_results.csv"))
            
        sampling_corr_df = self.create_sampling_corr_df()
        sampling_corr_df.to_csv(os.path.join(self.result_dir, "sampling_corr.csv"))
        
        if self.encoded_img_corr is not None and self.encoded_corr_df_index is not None: 
            encoded_corr_df = pd.DataFrame(self.encoded_img_corr, index = self.encoded_corr_df_index)
            encoded_corr_df.to_csv(os.path.join(self.result_dir, "encoded_img_corr.csv"))
            
    def marginal_relation_pipeline(self, corr_type: str = 'all', conf_level: float = 0.95, min_sample_prop: float = 0.2, min_sample_size: int = None, max_sample_prop: float = 1.0, max_sample_size: int = None, store_samples: bool = False):
        # for now this is just for one subject, all subject implementation coming later 
        pass

@dataclass
class LIME_all_subj_pipeline:
    test_data_id: str
    result_dir: str = None
    cuda_device_no: int = 0
    image_encoder_id: str = 'densenet121'
    graph_encoder_id: str = 'SETNET_GAT'
    num_classes: int = 2
    num_layers: int = 1
    input_dim: int = 1024
    test_data_dir: str = "/home/liuusa_tw/data/cropped_images/"
    ckpt_name: str = None
    test_data_list_name: str = None
    metadata_name: str = 'meta_data/TWB_ABD_expand_modified_gasex_21072022.csv'
    image_based: bool = True
    verbose: bool = True
    sample: set = None
    
    def __post_init__(self):
        
        if self.test_data_list_name is None:
            self.test_data_list_name = 'fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset'+str(self.test_data_id)+'/test_dataset'+str(self.test_data_id)+'.csv'
        if self.ckpt_name is None:
            self.ckpt_name = 'model_tl_twbabd'+str(self.test_data_id)+'/best_results.ckpt'
            
        if self.result_dir is None:
            result_timestamp = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
            self.result_dir = os.path.join("/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results", result_timestamp)
            if not os.path.exists(self.result_dir):
                os.mkdir(self.result_dir)
                
        self.__verify_input()
        self.device = torch.device(f'cuda:{self.cuda_device_no}' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.meta_data = pd.read_csv(self.metadata_name, sep=",")
        self.test_data_list = pd.read_csv(self.test_data_list_name)
        self.checkpoint = torch.load(self.ckpt_name)
        
        ## Call pretrained image encoder ###
        _, self.pretrained_image_encoder = models.image_encoder_model(name=self.image_encoder_id, 
                                                    pretrained=True, 
                                                    num_classes=self.num_classes, 
                                                    device=self.device)
        self.pretrained_image_encoder = self.pretrained_image_encoder.eval() 
        ### Call graph encoder ###
        self.graph_encoder, _=models.encoder_model(name=self.graph_encoder_id, 
                                            input_dim=self.input_dim,
                                            num_layers=self.num_layers,
                                            num_classes=self.num_classes,
                                            device=self.device)
        ### Load trained weights ###
        self.checkpoint = torch.load(self.ckpt_name)
        self.graph_encoder.load_state_dict(self.checkpoint['model_state_dict'])
        self.graph_encoder = self.graph_encoder.eval()
        
        

    def __verify_input(self):
        assert (isinstance(self.result_dir, str)
                and os.path.exists(self.result_dir))
        assert (isinstance(self.image_encoder_id, str) and self.image_encoder_id in [
                'resnet50', 'densenet121', 'vitl16in21k'])
        assert (isinstance(self.graph_encoder_id, str)
                and self.graph_encoder_id in ['SETNET_GAT'])
        assert (isinstance(self.cuda_device_no, int)
                and self.cuda_device_no > -1)
        assert (isinstance(self.input_dim, int)
                and self.input_dim > 1)
        assert (isinstance(self.num_classes, int) and self.num_classes > 0)
        assert (isinstance(self.num_layers, int) and self.num_layers > 0)

    def marginal_relation_pipeline(self, corr_type: str = 'all', conf_level: float = 0.95, min_sample_prop: float = 0.2, min_sample_size: int = None, max_sample_prop: float = 1.0, max_sample_size: int = None, store_samples: bool = False):
        # all subject implementation coming later 
        pass
    

    def get_marginal_relations_of_all_subj(self, n_samples: int = 1000):
        
        print(f"Results will be saved to {self.result_dir}")
        
        for mi_id_indx, mi_id in tqdm(enumerate(self.test_data_list['MI_ID'])):
            if self.verbose:
                print(mi_id)
                
            img_id_list = ast.literal_eval(self.meta_data[self.meta_data['MI_ID']==mi_id]['IMG_ID_LIST'].to_list()[0])
            
            y = self.meta_data[self.meta_data['MI_ID']==mi_id]['liver_fatty'].to_list()[0]
            
            ### Create graph data using image features ###
            mi_id_data=datasets.single_data_loader(mi_id=mi_id,
                                            img_id_list=img_id_list,
                                            image_transform=self.transform,
                                            pretrained_image_encoder=self.pretrained_image_encoder,
                                            y=y,
                                            num_classes=self.num_classes,
                                            device=self.device)
            
            y = mi_id_data.y

            def subj_pred_func(input_img_id_list: list, mi_id = mi_id):
                
                y = self.meta_data[self.meta_data['MI_ID']==mi_id]['liver_fatty'].to_list()[0]
                ### Create graph data using image features ###
                mydata= datasets.single_data_loader(mi_id=mi_id,
                                                img_id_list=input_img_id_list,
                                                image_transform=self.transform,
                                                pretrained_image_encoder=self.pretrained_image_encoder,
                                                y=y,
                                                num_classes=self.num_classes,
                                                device=self.device)
                ### Classification ###
                x = mydata.x.to(self.device)
                y = mydata.y
                A = mydata.edge_index_corr.to(self.device)
                b = torch.zeros(x.shape[0], dtype=torch.int64).to(self.device)
                train_mask=1 
                h = self.graph_encoder(x, A, b, train_mask)
                _, y_hat = torch.max(h, dim=1)
                y_hat = y_hat.data.to('cpu').numpy()[0]
                
                return int(y_hat)
                        
            mi_id_LIME = LIME_subj_pipeline(test_data_id = self.test_data_id, 
                                            img_list = img_id_list,
                                            mi_id = mi_id,
                                            img_dir = os.getenv('CROP_IMAGE_DIR_PATH'),
                                            pred_func = subj_pred_func,
                                            verbose = True,
                                            image_based = True,
                                            y = y,
                                            result_parent_dir = self.result_dir,
                                            )
            _ = mi_id_LIME.predict_on_random_samples_until_convergence(n_samples = n_samples, 
                                                                       target_positive_proportion = 0.5, 
                                                                       min_sample_size = 3,
                                                                       max_sample_size = len(img_id_list), 
                                                                       )
            
            num_correct = len([i for i in 
                               mi_id_LIME.sample_pred_results
                               if i == y])
            accuracy =  num_correct / len(mi_id_LIME.sample_pred_results)
            print(f"Accuracy for {mi_id}: {accuracy}, True class: {y} # correct: {num_correct}, total # of imgs: {len(img_id_list)}")
            
            mi_id_LIME.get_imgs_marginal_relation(corr_type='pearson')
            mi_id_LIME.plot_sampling_corr_heatmap()
            mi_id_LIME.plot_img_corr_heatmap(transform=self.transform, 
                                             image_encoder=self.pretrained_image_encoder, 
                                             device=self.device,)
            mi_id_LIME.plot_image_bar_plot()
            mi_id_LIME.save_results()
            
            if mi_id_indx % 50 == 0:
                del mi_id_LIME
                gc.collect()
                
                if self.verbose:
                    print(f"Finished {mi_id_indx + 1} subjects out of {len(self.test_data_list['MI_ID'])} total subjects ")
                    
                    
@dataclass
class logistic_regression_on_custom_LIME:
    result_dir: str
    
    def logistic_regression_on_all_subj(self):
        pass
    
    def logistic_regression_on_single_subj(self):
        pass