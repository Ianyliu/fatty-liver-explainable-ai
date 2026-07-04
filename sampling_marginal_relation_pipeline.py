import ast
import datetime
import gc
import inspect
import os
from dataclasses import dataclass
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from skimage.segmentation import mark_boundaries
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from torch_geometric.data import Data
from tqdm import tqdm

from usflc_xai import datasets, models

@dataclass
class LIME_subj_pipeline:
    """
    A class to handle the LIME subject pipeline for image sampling and prediction analysis.
    Attributes:
    -----------
    test_data_id : str
        Identifier for the test data.
    img_list : set
        Set of image identifiers.
    y : bool
        Ground truth label.
    mi_id : str
        Model identifier.
    img_dir : str
        Directory containing images.
    pred_func : Callable[[list], int]
        Prediction function that takes a list of images and returns an integer prediction.
    result_parent_dir : str
        Parent directory to store results.
    image_based : bool, optional
        Flag indicating if the pipeline is image-based (default is True).
    verbose : bool, optional
        Flag for verbose output (default is False).
    Methods:
    --------
    __post_init__():
        Initializes the pipeline and verifies inputs.
    __verify_input():
        Verifies the input parameters.
    __verify_sampling_input(n_samples, target_positive_proportion, min_sample_prop=None, min_sample_size=None, max_sample_prop=None, max_sample_size=None):
        Verifies the sampling input parameters.
    __verify_pred_function_signature():
        Verifies the signature of the prediction function.
    predict_on_random_samples_until_convergence(n_samples, target_positive_proportion, min_sample_prop=None, min_sample_size=None, max_sample_prop=None, max_sample_size=None, max_iter=100000, append_to_original=True):
        Predicts on random samples until convergence.
    __only_sample_positives(n_extra_samples, min_sample_size, max_sample_size):
        Samples only positive images.
    __only_sample_negatives(n_extra_samples, min_sample_size, max_sample_size):
        Samples only negative images.
    generate_balanced_random_samples(n_samples, target_positive_proportion, min_sample_prop=None, min_sample_size=None, max_sample_prop=None, max_sample_size=None, pred_on_samples=False):
        Generates balanced random samples.
    __predict_on_all_single_img():
        Predicts on all single images.
    __generate_sampling_pool():
        Generates the sampling pool.
    generate_pred_results_matrix():
        Generates the prediction results matrix.
    get_imgs_marginal_relation(corr_type='pearson', conf_level=0.95):
        Gets the marginal relation of images.
    calculate_img_corr(img, corr_type='pearson'):
        Calculates the correlation of an image.
    plot_sampling_corr_heatmap():
        Plots the sampling correlation heatmap.
    create_sampling_corr_df():
        Creates a DataFrame of the sampling correlation.
    plot_img_corr_heatmap(image_encoder, transform, device):
        Plots the image correlation heatmap.
    plot_image_bar_plot():
        Plots a bar plot of image correlations.
    save_results():
        Saves the results to the result directory.
    """
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
        assert self.test_data_id in [f'{i:02}' for i in range(1, 11)], "Invalid test_data_id"

        self.__verify_input()
        self.__verify_pred_function_signature()

        self.sample_pred_results = []
        self.samples = []
        self.bootstrap_sample = []
        self.img_list = sorted(self.img_list)
        self.all_img_abs_filepaths = sorted(self.all_img_abs_filepaths)
        self.img_to_indx = dict(enumerate(self.img_list))
        self.indx_to_img = dict(enumerate(self.img_list))
        self.indx_to_abs_filepath = dict(enumerate(self.all_img_abs_filepaths))
        self.result_dir = os.path.join(self.result_parent_dir, self.mi_id)
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    def __verify_input(self):
        """
        Verifies the input attributes of the class instance.
        Raises:
            TypeError: If `test_data_id` is not a string.
            TypeError: If `y` is not an integer.
            TypeError: If `img_list` is not a list.
            ValueError: If `img_list` is empty or contains duplicate elements.
            FileNotFoundError: If any image file in `img_list` does not exist.
            FileNotFoundError: If `result_parent_dir` does not exist.
        """
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

        # Check img_list is not empty and contains unique elements
        if not self.img_list or len(self.img_list) == 0:
            raise ValueError("img_list must have length > 0")
        if len(self.img_list) != len(set(self.img_list)):
            print("Image list is not unique! Converting img_list to a unique img list.")
            self.img_list = list(set(self.img_list))
        
        # Check all image paths exist
        self.all_img_abs_filepaths = [os.path.join(self.img_dir, f"{self.mi_id}_{img}.jpg") for img in self.img_list]
        if not all(os.path.exists(filepath) for filepath in self.all_img_abs_filepaths):
            missing_files = [filepath for filepath in self.all_img_abs_filepaths if not os.path.exists(filepath)]
            raise FileNotFoundError(f"The following image files do not exist: {missing_files}")

        # Check result_parent_dir exists
        if not os.path.exists(self.result_parent_dir):
            raise FileNotFoundError(f"{self.result_parent_dir} does not exist!")

    # %%
    def __verify_sampling_input(self, n_samples: int, target_positive_proportion: float, min_sample_prop: float = None, min_sample_size: int = None, max_sample_prop: float = None, max_sample_size: int = None, ):
        """
        Verify the input parameters for the sampling process.
        Parameters:
        -----------
        n_samples : int
            The number of samples to be drawn.
        target_positive_proportion : float
            The target proportion of positive samples.
        min_sample_prop : float, optional
            The minimum proportion of samples to be drawn, must be between 0.0 and 1.0.
        min_sample_size : int, optional
            The minimum number of samples to be drawn, must be between 0 and the total number of images.
        max_sample_prop : float, optional
            The maximum proportion of samples to be drawn, must be between 0.0 and 1.0.
        max_sample_size : int, optional
            The maximum number of samples to be drawn, must be between 0 and the total number of images.
        Raises:
        -------
        ValueError
            If no images are available in the sampling pool.
            If both min_sample_size and min_sample_prop are provided.
            If both max_sample_size and max_sample_prop are provided.
            If min_sample_prop is not between 0.0 and 1.0.
            If max_sample_prop is not between 0.0 and 1.0.
            If min_sample_size is not between 0 and the total number of images.
            If max_sample_size is not between 0 and the total number of images.
            If min_sample_prop is greater than max_sample_prop.
            If min_sample_size is greater than max_sample_size.
        TypeError
            If n_samples is not an integer.
            If target_positive_proportion is not a float.
            If min_sample_prop is provided and is not a float.
            If max_sample_prop is provided and is not a float.
            If min_sample_size is provided and is not an integer.
            If max_sample_size is provided and is not an integer.
        """
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

    # %%
    def __verify_pred_function_signature(self):
        """
        Verifies the signature of the prediction function (pred_func) and tests its output.
        This method performs the following checks:
        1. Verifies that the number of required parameters in pred_func matches the expected number.
        2. Tests the pred_func with the provided image list (self.img_list).
        3. Ensures that the output of pred_func is of type int.
        Raises:
            TypeError: If the output of pred_func is not of type int.
        """
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

    # %%
    def predict_on_random_samples_until_convergence(self, n_samples: int, target_positive_proportion: float, min_sample_prop: float = None, min_sample_size: int = None, max_sample_prop: float = None, max_sample_size: int = None, max_iter: int = 100000, append_to_original: bool = True):
        """
        Predict on random samples until convergence to a target positive proportion.
        Parameters:
        -----------
        n_samples : int
            The number of samples to generate.
        target_positive_proportion : float
            The target proportion of positive samples (between 0 and 1).
        min_sample_prop : float, optional
            The minimum proportion of the total images to be used in a sample.
        min_sample_size : int, optional
            The minimum number of images to be used in a sample.
        max_sample_prop : float, optional
            The maximum proportion of the total images to be used in a sample.
        max_sample_size : int, optional
            The maximum number of images to be used in a sample.
        max_iter : int, optional
            The maximum number of iterations to perform (default is 100000).
        append_to_original : bool, optional
            Whether to append the new samples to the original list of samples (default is True).
        Returns:
        --------
        list
            A list of samples generated during the process.
        Raises:
        -------
        TypeError
            If `target_positive_proportion` is not a float or `max_iter` is not an int.
        ValueError
            If `target_positive_proportion` is not between 0 and 1, `max_iter` is less than or equal to 0,
            `n_samples` is 0, or `n_samples` is greater than or equal to `max_iter`.
        Notes:
        ------
        - The function will print warnings and progress information if `self.verbose` is set to True.
        - The function ensures that the generated samples meet the target positive proportion as closely as possible
          within the given number of iterations.
        - If all images belong to one class, class balance cannot be guaranteed, and resampling will be skipped.
        """
        self.__predict_on_all_single_img()
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

    # %%
    def __only_sample_positives(self, n_extra_samples: int,min_sample_size: int, max_sample_size: int):
        """
        Generates a specified number of extra positive samples from the positive image pool.
        If the positive image pool is empty, it generates the positive samples first.
        Args:
            n_extra_samples (int): The number of extra positive samples to generate.
            min_sample_size (int): The minimum size of the sample.
            max_sample_size (int): The maximum size of the sample.
        Returns:
            List: A list of balanced random samples with a target positive proportion.
        """
        if self.positive_img_pool is None or len(self.positive_img_pool)  == 0:
            print("positive_img_pool is empty, generating positive samples")
            self.__generate_sampling_pool()
            
        return self.generate_balanced_random_samples(n_samples = n_extra_samples, 
                                              target_positive_proportion = 0.85, 
                                              min_sample_size=min_sample_size,
                                              max_sample_size= max_sample_size,
                                              pred_on_samples=True,
                                              )
        
    # %%    
    def __only_sample_negatives(self,n_extra_samples: int,min_sample_size: int, max_sample_size: int):
        """
        Generates additional negative samples if the negative image pool is empty and then 
        returns a balanced set of random samples with a specified number of extra samples.
        Args:
            n_extra_samples (int): The number of extra samples to generate.
            min_sample_size (int): The minimum size of the sample.
            max_sample_size (int): The maximum size of the sample.
        Returns:
            list: A list of balanced random samples with the specified number of extra samples.
        """
        if self.negative_img_pool is None or  len(self.negative_img_pool) == 0:
            print("negative_img_pool is empty, generating negative samples")     
            self.__generate_sampling_pool()
            
        return self.generate_balanced_random_samples(n_samples = n_extra_samples, 
                                              target_positive_proportion = 0.15, 
                                              min_sample_size=min_sample_size,
                                              max_sample_size= max_sample_size,
                                              pred_on_samples=True,
                                              )       
    
    #%%
    def generate_balanced_random_samples(self, n_samples: int, target_positive_proportion: float, min_sample_prop: float = None, min_sample_size: int = None, max_sample_prop: float = None, max_sample_size: int = None, pred_on_samples = False):
        """
        Generate balanced random samples from positive and negative image pools.
        Parameters:
        -----------
        n_samples : int
            The number of samples to generate.
        target_positive_proportion : float
            The proportion of positive samples in each generated sample. Must be between 0.0 and 1.0.
        min_sample_prop : float, optional
            The minimum proportion of the total images to be used in a sample. If provided, min_sample_size is calculated as int(num_imgs * min_sample_prop).
        min_sample_size : int, optional
            The minimum number of images to be used in a sample.
        max_sample_prop : float, optional
            The maximum proportion of the total images to be used in a sample. If provided, max_sample_size is calculated as int(num_imgs * max_sample_prop).
        max_sample_size : int, optional
            The maximum number of images to be used in a sample.
        pred_on_samples : bool, optional
            If True, predictions will be made on the generated samples using the pred_func method.
        Returns:
        --------
        list
            A list of generated samples. Each sample is a list of image identifiers.
        Raises:
        -------
        ValueError
            If target_positive_proportion is not between 0.0 and 1.0.
        """
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
    # %%
    def __predict_on_all_single_img(self):
        """
        Predicts on all single images in the image list and stores the results.

        This method iterates over the list of images (`self.img_list`), applies the prediction function (`self.pred_func`) 
        to each image, and stores the results in `self.single_img_results`.

        If `self.verbose` is True, it calculates and prints the accuracy of the predictions, the number of correct predictions, 
        and the total number of images.

        Attributes:
            self.single_img_results (dict): A dictionary where the keys are images and the values are the predictions.
            self.img_list (list): A list of images to be predicted.
            self.pred_func (function): A function that takes a list of images and returns predictions.
            self.verbose (bool): A flag to indicate whether to print the accuracy and other details.
            self.y: The ground truth label to compare predictions against.

        Prints:
            Accuracy on all single images, number of correct predictions, and total number of images (if `self.verbose` is True).
        """
        self.single_img_results = {img: self.pred_func([img]) for img in tqdm(self.img_list)}
        if self.verbose:
            num_correct = len([i for i in self.single_img_results.values() if i == self.y])
            accuracy =  num_correct / len(self.single_img_results)
            print(f"Accuracy on all single images: {accuracy}, # correct: {num_correct}, total # of imgs: {len(self.img_list)}")

    # %%
    def __generate_sampling_pool(self):
        """
        Generates the sampling pool by categorizing images into positive and negative pools based on prediction results.

        This method performs the following steps:
        1. Ensures that single image prediction results are available and complete.
        2. Categorizes images into negative and positive pools based on whether their prediction results match the expected value `self.y`.
        3. Converts the negative and positive pools from sets to lists.
        4. Ensures that the total number of images in both pools matches the total number of images in `self.img_list`.

        Raises:
            AssertionError: If single image prediction results do not exist.
            AssertionError: If the number of single image prediction results does not match the number of images.
            AssertionError: If the total number of images in the negative and positive pools does not match the total number of images in `self.img_list`.
        """
        assert len(self.single_img_results) != 0, "Single image prediction results do not exist. Try running self.__predict_on_all_single_img() first"
        assert len(self.single_img_results) == len(self.img_list), f"Some image predictions were not complete. Num of images: {len(self.img_list)}, num of results: {len(self.single_img_results)}"
        self.negative_img_pool = {k for k,v in self.single_img_results.items() if v != self.y}
        self.positive_img_pool = set(self.img_list) - self.negative_img_pool

        self.negative_img_pool = list(self.negative_img_pool)
        self.positive_img_pool = list(self.positive_img_pool)

        assert (len(self.negative_img_pool) + len(self.positive_img_pool)) == len(self.img_list), "Negative image pool and positive image pools are incomplete, excluding some images" 

    # %%
    def generate_pred_results_matrix(self):
        """
        Generates a prediction results matrix for the given samples and images.
        This method initializes and populates two matrices: `pred_results_x` and `pred_results_yhat`.
        - `pred_results_x` is a binary matrix indicating the presence of images in each sample.
        - `pred_results_yhat` is an array containing the prediction results for each sample.
        The method performs the following steps:
        1. Initializes `pred_results_x` as a zero matrix with dimensions (number of samples, number of images).
        2. Initializes `pred_results_yhat` as an array from `self.sample_pred_results`.
        3. Iterates over each sample and sets the corresponding entries in `pred_results_x` to 1 based on the presence of images.
        4. Checks for any NaN values in `pred_results_x` and replaces them with 0 if found.
        Raises:
            AssertionError: If `self.samples` or `self.sample_pred_results` is None, or if their lengths do not match.
        Attributes:
            self.pred_results_x (np.ndarray): Binary matrix indicating the presence of images in each sample.
            self.pred_results_yhat (np.ndarray): Array containing the prediction results for each sample.
        """
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

    # %%
    def get_imgs_marginal_relation(self, corr_type: str = 'pearson', conf_level: float = 0.95):
        """
        Calculate the marginal relation (correlation) between images and predictions.
        This method generates a matrix of prediction results and calculates the correlation
        between each image and its predictions using the specified correlation type. It also
        computes the standard deviation, confidence interval, and p-value for the correlation.
        Parameters:
        corr_type (str): The type of correlation to calculate. Default is 'pearson'.
        conf_level (float): The confidence level for the confidence interval. Default is 0.95.
        Returns:
        dict: A dictionary where the keys are image indices and the values are dictionaries
              containing the following keys:
              - "corr": The correlation value.
              - "corr_std": The standard deviation of the correlation.
              - "corr_CI": The confidence interval of the correlation.
              - "corr_upper_CI": The upper bound of the confidence interval.
              - "corr_lower_CI": The lower bound of the confidence interval.
              - "corr_p_val": The p-value of the correlation.
        """
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
    
    # %%
    def calculate_img_corr(self, img, corr_type: str = 'pearson'):
        """
        Calculate the correlation between an image column and the predicted results.
        Parameters:
        img (str or int): The image identifier, either as a string (image name) or an integer (image index).
        corr_type (str): The type of correlation to calculate. Options are 'pearson' (or 'p'), 'matthews' (or 'mcc'), 
                         and 'spearman' (or 's'). Default is 'pearson'.
        Returns:
        tuple: A tuple containing the following elements:
            - corr (float): The calculated correlation coefficient.
            - corr_std (float or None): The standard deviation of the correlation coefficient (only for Pearson correlation).
            - corr_CI (tuple or None): The confidence interval of the correlation coefficient (only for Pearson correlation).
            - corr_p_val (float or None): The p-value of the correlation coefficient.
        Raises:
        ValueError: If the input type for 'img' is not str or int, or if the correlation type is invalid.
        Notes:
        - If the image column or predicted results contain only unique values, warnings will be printed and the correlation 
          values will be set to 0 or None.
        """
        if isinstance(img, str):
            img_indx = self.img_to_indx[img]
        elif isinstance(img, int):
            assert img in self.indx_to_img.keys()
            img_indx = img
        else:
            raise ValueError(f"Invalid input type for 'img'. Expected str or int, got {type(img)}")
            
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
        else:
            raise ValueError(f"Invalid correlation type. Expected 'pearson', 'p','matthews','mcc','spearman', or's', got {corr_type}")
            
        return corr, corr_std, corr_CI, corr_p_val

    # %%
    def plot_sampling_corr_heatmap(self):
        """
        Plots a heatmap of the correlation matrix of the sampling predictions.
        This method calculates the correlation matrix of the transposed prediction results
        and plots a heatmap using seaborn. The heatmap is saved as an image file in the 
        specified result directory.
        Attributes:
            self.sampling_corr (ndarray): The correlation matrix of the transposed prediction results.
            heatmap_labels (list): List of labels for the heatmap axes, derived from the indices of the prediction results.
        Saves:
            A heatmap image file named 'image-sampling-heatmap.png' in the result directory.
        """
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
        
    # %%
    def create_sampling_corr_df(self):
        """
        Creates a DataFrame containing the correlation coefficients of the prediction results.
        This method calculates the correlation matrix of the transposed prediction results
        and returns it as a pandas DataFrame with column names corresponding to the image indices.
        Returns:
            pd.DataFrame: A DataFrame containing the correlation coefficients with image indices as column names.
        """
        self.sampling_corr = np.corrcoef(self.pred_results_x.T)
        colnames = [self.indx_to_img[indx] for indx, _ in enumerate(self.pred_results_x.T)]
        
        return pd.DataFrame(self.sampling_corr, columns=colnames)
    # %% 
    def plot_img_corr_heatmap(self, image_encoder, transform, device):
        """
        Plots a heatmap of the correlation matrix of encoded images.
        Args:
            image_encoder (torch.nn.Module): The image encoder model to encode the images.
            transform (callable): A function/transform that takes in an image and returns a transformed version.
            device (torch.device): The device (CPU or GPU) to perform computations on.
        Raises:
            AssertionError: If `self.all_img_abs_filepaths` is None.
        Side Effects:
            - Sets `self.encoded_images` to the tensor of transformed images.
            - Sets `self.encoded_img_corr` to the numpy array of the correlation matrix.
            - Sets `self.encoded_corr_df_index` to the list of image filenames without extensions.
            - Saves the heatmap plot as "image-encoded-corr-heatmap.png" in `self.result_dir`.
            - Displays the heatmap plot.
        """
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
    
    # %%
    def plot_image_bar_plot(self):
        """
        Plots horizontal and vertical bar plots with images overlaid on the bars.
        This function generates two types of bar plots:
        1. Horizontal bar plot with images overlaid on the bars.
        2. Vertical bar plot with images overlaid on the bars.
        The function uses the correlation values and their corresponding images to create the plots.
        The images are displayed at the end of each bar, and the bars are colored based on the sign of the correlation values.
        Parameters:
        None
        Returns:
        None
        Notes:
        - The function saves the generated plots as 'hbar.png' and 'vbar.png' in the specified result directory.
        - The function handles cases where the correlation values have associated standard deviations.
        - The function adjusts the size and layout of the plots based on the number of bars.
        Example usage:
        self.plot_image_bar_plot()
        """
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

    # %%
    def save_results(self):
        """
        Saves various results to CSV files in the specified result directory.
        This method performs the following operations:
        1. Saves the design matrix and prediction results to 'design_matrix.csv' and 'pred_results.csv' respectively.
        2. Saves the image prediction correlations to 'corr_results.csv'.
        3. Saves the sampling correlations to 'sampling_corr.csv'.
        4. Saves the encoded image correlations to 'encoded_img_corr.csv'.
        Preconditions:
        - `self.result_dir` must not be None.
        - If `self.indx_to_img`, `self.pred_results_x`, and `self.pred_results_yhat` are not None, they must have compatible shapes.
        - If `self.img_pred_corr` is not None, it must be a dictionary with sortable keys.
        - If `self.encoded_img_corr` and `self.encoded_corr_df_index` are not None, they must have compatible shapes.
        Raises:
        - AssertionError: If `self.result_dir` is None.
        Returns:
            None
        """
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
    
    # %%
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
        
    # %%
    def __verify_input(self):
        """
        Verifies the input parameters for the class instance.

        This method checks the following conditions:
        - `self.result_dir` is a string and the directory exists.
        - `self.image_encoder_id` is a string and one of 'resnet50', 'densenet121', 'vitl16in21k'.
        - `self.graph_encoder_id` is a string and is 'SETNET_GAT'.
        - `self.cuda_device_no` is an integer and greater than -1.
        - `self.input_dim` is an integer and greater than 1.
        - `self.num_classes` is an integer and greater than 0.
        - `self.num_layers` is an integer and greater than 0.

        Raises:
            AssertionError: If any of the conditions are not met.
        """
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
    
    # %%
    def get_marginal_relations_of_all_subj(self, n_samples: int = 1000):
        """
        Computes and saves the marginal relations of all subjects in the test data list.
        Parameters:
        -----------
        n_samples : int, optional
            The number of samples to use for LIME predictions until convergence (default is 1000).
        Description:
        ------------
        This method iterates over each subject in the test data list, loads the corresponding image data,
        and uses LIME (Local Interpretable Model-agnostic Explanations) to predict on random samples until
        convergence. It then calculates the accuracy of the predictions, generates correlation heatmaps,
        and saves the results.
        The method performs the following steps for each subject:
        1. Prints the results directory.
        2. Loads the image ID list and the true class label for the subject.
        3. Creates graph data using image features.
        4. Defines a prediction function for the subject.
        5. Initializes the LIME pipeline and predicts on random samples until convergence.
        6. Calculates and prints the accuracy of the predictions.
        7. Generates and plots correlation heatmaps and bar plots.
        8. Saves the results.
        9. Periodically clears memory to manage resources.
        Parameters:
        -----------
        n_samples : int, optional
            The number of samples to use for LIME predictions until convergence (default is 1000).
        Returns:
        --------
        None
        """
        
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
