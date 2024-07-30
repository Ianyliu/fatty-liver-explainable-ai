import ast
import datetime
import gc
import itertools
import os
import time
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from PIL import Image
from skimage.segmentation import mark_boundaries

# this enables 10-100X acceleration of scikit-learn, see https://intel.github.io/scikit-learn-intelex/latest/ for more details
from sklearnex import patch_sklearn  # isort:skip # nopep8
patch_sklearn()  # isort:skip # nopep8
from sklearn.linear_model import HuberRegressor, SGDRegressor  # isort:skip # nopep8
from sklearn.metrics import confusion_matrix  # isort:skip # nopep8

from torch_geometric.data import Data
from tqdm import tqdm
from usflc_xai import datasets, models

# import ast
# import datetime
# import gc
# import glob
# import os
# import time
# from dataclasses import dataclass
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from lime import lime_image
# from lime.wrappers.scikit_image import SegmentationAlgorithm
# from PIL import Image
# from skimage.segmentation import mark_boundaries

# # this enables 10-100X acceleration of scikit-learn, see https://intel.github.io/scikit-learn-intelex/latest/ for more details
# from sklearnex import patch_sklearn
# patch_sklearn()
# from sklearn.linear_model import HuberRegressor, SGDRegressor
# from sklearn.metrics import confusion_matrix

# from torch_geometric.data import Data
# from tqdm import tqdm
# from usflc_xai import datasets, models


print(torch.__version__)
print(torchvision.__version__)


@dataclass
class LIME_pipeline:
    test_data_id: str
    result_dir: str
    cuda_device_no: int = 0
    image_encoder_id: str = 'densenet121'
    graph_encoder_id: str = 'SETNET_GAT'
    num_classes: int = 2
    num_layers: int = 1
    test_data_dir: str = "/home/liuusa_tw/data/cropped_images/"
    ckpt_name: str = None
    test_data_list_name: str = None
    metadata_name: str = None
    all_possible_test_sets: dict = None

    def __post_init__(self):
        assert (self.test_data_id in ['01', '02', '03', '04',
                                      '05', '06', '07', '08', '09', '10'])
        assert (isinstance(self.result_dir, str)
                and os.path.exists(self.result_dir))
        assert (isinstance(self.cuda_device_no, int))
        assert (isinstance(self.image_encoder_id, str) and self.image_encoder_id in [
                'resnet50', 'densenet121', 'vitl16in21k'])
        assert (isinstance(self.graph_encoder_id, str)
                and self.graph_encoder_id in ['SETNET_GAT'])
        assert (isinstance(self.cuda_device_no, int)
                and self.cuda_device_no > -1)
        assert (isinstance(self.num_classes, int) and self.num_classes > 0)
        assert (isinstance(self.num_layers, int) and self.num_layers > 0)

        if self.image_encoder_id == 'resnet50':
            self.input_dim = 2048
        elif self.image_encoder_id == 'vitl16in21k':
            self.input_dim = 768
        elif self.image_encoder_id == 'densenet121':
            self.input_dim = 1024

        self.__load_data()
        self.__load_model()
        self.explainer = lime_image.LimeImageExplainer()

    def __load_data(self):

        if self.metadata_name is None:
            self.metadata_name = 'meta_data/TWB_ABD_expand_modified_gasex_21072022.csv'
        if self.test_data_list_name is None:
            self.test_data_list_name = f'fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset{self.test_data_id}/test_dataset{self.test_data_id}.csv'

        assert (os.path.exists(self.metadata_name))
        assert (os.path.exists(self.test_data_list_name))

        self.meta_data = pd.read_csv(self.metadata_name, sep=",")
        self.test_data_list = pd.read_csv(self.test_data_list_name)

        print(
            f'There are {len(self.meta_data)} subjects in the meta data list.')
        print(
            f'There are {len(self.test_data_list)} subjects in the test dataset{self.test_data_id}.')

    def __load_model(self):
        """Loads model
        """

        self.device = torch.device(
            f'cuda:{self.cuda_device_no}' if torch.cuda.is_available() else 'cpu')
        if self.ckpt_name is None:
            self.ckpt_name = f'model_tl_twbabd{self.test_data_id}/best_results.ckpt'
        assert (os.path.exists(self.ckpt_name))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        ## Call pretrained image encoder ###
        _, self.pretrained_image_encoder = models.image_encoder_model(name=self.image_encoder_id,
                                                                      pretrained=True,
                                                                      num_classes=self.num_classes,
                                                                      device=self.device)

        self.pretrained_image_encoder = self.pretrained_image_encoder.eval()

        ### Call graph encoder ###
        self.graph_encoder, _ = models.encoder_model(name=self.graph_encoder_id,
                                                     input_dim=self.input_dim,
                                                     num_layers=self.num_layers,
                                                     num_classes=self.num_classes,
                                                     device=self.device)
        ### Load trained weights ###

        self.checkpoint = torch.load(self.ckpt_name)
        self.graph_encoder.load_state_dict(self.checkpoint['model_state_dict'])
        self.graph_encoder = self.graph_encoder.eval()

    def __flatten(self, lst):
        return set(item for sublist in lst for item in sublist)
    
    def __convert_tuple_to_list(self, tuple_input):
        return [item for subtuple in tuple_input for item in subtuple]
    
    def generate_all_possible_test_sets(self):
        """Generates all possible test sets"""

        self.all_possible_test_sets = {}
        for i in tqdm(range(len(self.test_data_list))):
            mi_id = self.test_data_list['MI_ID'][i]
            img_id_list = ast.literal_eval(
                self.meta_data[self.meta_data['MI_ID'] == mi_id]['IMG_ID_LIST'].to_list()[0])
            
            self.__generate_all_test_sets_for_single_img(mi_id, img_id_list)
        
        print(f"{len(self.all_possible_test_sets)} test sets generated")
            
                
    def __generate_all_test_sets_for_single_img(self, mi_id: str, img_id_list: list):
        # Generate all possible combinations
            for k in range(1, len(img_id_list) + 1):
                test_combinations = list(itertools.combinations(img_id_list, k))
                self.all_possible_test_sets[mi_id] = test_combinations

    def __single_prediction(self, mi_id: str, img_id_list: list, y: int):
        ### Create graph data using image features ###

            mydata = datasets.single_data_loader(mi_id=mi_id,
                                                 img_id_list=img_id_list,
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
            train_mask = 1

            h = self.graph_encoder(x, A, b, train_mask)
            _, y_hat = torch.max(h, dim=1)
            y_hat = y_hat.data.to('cpu').numpy()[0]
            
            return y, y_hat
    
    def predict_on_test(self):
        """Predict on test data"""
        self.accuracy = 0
        self.pred_results = [None] * len(self.test_data_list)
        self.pred_results_dict = {
            'y': [None] * len(self.test_data_list),
            'y_hat': [None] * len(self.test_data_list),

        }

        for k in tqdm(range(len(self.test_data_list))):

            ## Subject id in the test data ##
            mi_id = self.test_data_list['MI_ID'][k]

            ## Obtain image ids and label y for each subject. ###

            img_id_list = ast.literal_eval(
                self.meta_data[self.meta_data['MI_ID'] == mi_id]['IMG_ID_LIST'].to_list()[0])
            y = self.meta_data[self.meta_data['MI_ID'] == mi_id]['liver_fatty'].to_list()[
                0]
            
            y, y_hat = self.__single_prediction(mi_id=mi_id, img_id_list=img_id_list, y=y)

            crt = (y_hat == y)
            self.accuracy += crt
            self.pred_results[k] = {
                "indx": k,
                "y": y,
                "y_hat": y_hat,
                "crt": crt,
                "mi_id": mi_id,
                "img_id_list": img_id_list,
            }
            self.pred_results_dict['y'][k] = y
            self.pred_results_dict['y_hat'][k] = y_hat

        # del x
        del y
        # del A
        # del b
        del crt
        del y_hat
        del mi_id
        del k
        del img_id_list

        gc.collect()

        print('Accuracy is {:.4f}.'.format(
            self.accuracy / len(self.test_data_list)))

        self.confusion_matrix_results = confusion_matrix(
            self.pred_results_dict['y'],
            self.pred_results_dict['y_hat'])
        
        if self.confusion_matrix_results.shape[0] != 2 and self.confusion_matrix_results.shape[1] != 2:
            print("Confusion matrix error")
        else:
            print(self.confusion_matrix_results, self.confusion_matrix_results.shape)

            self.tn, self.fp, self.fn, self.tp = self.confusion_matrix_results.ravel()
            print(
                f"There were {self.tp} true positives, {self.tn} true negatives, {self.fp} false positives, {self.fn} false negatives.")

        tp_mid = []
        tp_mid_img_list = {}
        for pred_result in self.pred_results:
            if pred_result.get("y") == 1 == pred_result.get("y_hat"):
                tp_mid.append(pred_result.get("mi_id"))
                tp_mid_img_list[pred_result.get(
                    "mi_id")] = pred_result.get("img_id_list")

        self.tp_mid = tp_mid.copy()
        self.tp_mid_img_list = tp_mid_img_list.copy()
        del tp_mid
        del tp_mid_img_list

        self.pred_results_df = pd.DataFrame(self.pred_results_dict)
        pred_results_csv_path = os.path.join(
            self.result_dir, f"pred_results{self.test_data_id}.csv")
        if os.path.exists(pred_results_csv_path):
            print(
                f"Overwriting previous prediction results for {self.test_data_id}, {pred_results_csv_path} was last modified at {os.path.getmtime(pred_results_csv_path)}")

        self.pred_results_df.to_csv(pred_results_csv_path)
        print(f"Predictions saved to {pred_results_csv_path}")

    def __pre_filter(self, y: int, img_id_list: list, mi_id: str):
        indx = 0
        img_tp_id = set(img_id_list)
        for img_id in img_id_list:
            y, y_hat = self.__single_prediction(mi_id=mi_id, img_id_list=[img_id], y=y)
            crt = (y_hat == y)
            self.accuracy += crt
            indx_pred_result = {
                "indx": indx,
                "y": y,
                "y_hat": y_hat,
                "crt": crt,
                "mi_id": mi_id,
                "img_id_list": [img_id],
            }
            self.pred_results.append(indx_pred_result)
            self.pred_results_dict['y'].append(y)
            self.pred_results_dict['y_hat'].append(y_hat)
            indx += 1
                
            if not crt:
                # Remove incorrect IDs from img_tp_id
                img_tp_id.remove(img_id)
                            
        del indx
        return img_tp_id

    def monte_carlo_adaptive_prediction_on_test_sets(self, num_samples: int =1000, max_combination_size: int = None, pre_filter: bool = True):
        """Predict on all possible test sets. Requires all test sets to be generated."""
        
        # One loop of single images, using the new set of images sample from it iteratively
        self.accuracy = 0
        self.pred_results = []
        self.pred_results_dict = {
            'y': [],
            'y_hat': [],

        }
        
        indx = 0
        
        for i in tqdm(range(len(self.test_data_list))):

            ## Subject id in the test data ##
            mi_id = self.test_data_list['MI_ID'][i]

            ## Obtain image ids and label y for each subject. ###
            img_id_list = ast.literal_eval(
                self.meta_data[self.meta_data['MI_ID'] == mi_id]['IMG_ID_LIST'].to_list()[0])
            y = self.meta_data[self.meta_data['MI_ID'] == mi_id]['liver_fatty'].to_list()[
                0]
            self.mc_test_sets = []
            # Convert to set for efficient removal
            available_ids = set(img_id_list)
            # Determine the maximum size for combinations
            if max_combination_size is None:
                max_size = len(img_id_list)
            else:
                max_size = min(max_combination_size, len(img_id_list))
                
            if pre_filter:
                available_ids = self.__pre_filter(y=y, 
                                                  img_id_list=img_id_list, 
                                                  mi_id=mi_id)
                indx = len(self.pred_results) 
            

            # Generate random samples
            for j in range(num_samples):
                if j % 2 == 10:
                    print(f"Generated {j+1} samples, {num_samples-(j+1)} more samples to go")
                    
                if not available_ids:  # If all IDs have been eliminated, break
                    break
                
                # Randomly choose the size of this combination
                size = min(random.randint(1, max_size), len(available_ids))
                
                # Randomly sample 'size' elements from available_ids
                combination = random.sample(list(available_ids), size)
                
                y, y_hat = self.__single_prediction(mi_id=mi_id, img_id_list=combination, y=y)
                crt = (y_hat == y)
                self.accuracy += crt
                indx_pred_result = {
                    "indx": indx,
                    "y": y,
                    "y_hat": y_hat,
                    "crt": crt,
                    "mi_id": mi_id,
                    "img_id_list": combination,
                }
                self.pred_results.append(indx_pred_result)
                self.pred_results_dict['y'].append(y)
                self.pred_results_dict['y_hat'].append(y_hat)
                indx += 1
                                        
                self.mc_test_sets.append(tuple(combination))
                
        del indx
        
        print('Accuracy is {:.4f}.'.format(
            self.accuracy / len(self.pred_results)))

        self.confusion_matrix_results = confusion_matrix(
            self.pred_results_dict['y'],
            self.pred_results_dict['y_hat'])

        self.tn, self.fp, self.fn, self.tp = self.confusion_matrix_results.ravel()
        print(
            f"There were {self.tp} true positives, {self.tn} true negatives, {self.fp} false positives, {self.fn} false negatives.")

        self.__output_pred_results()

    def exhaustive_adaptive_prediction_on_test_sets(self):
        """Predict on all possible test sets. Requires all test sets to be generated."""
        self.accuracy = 0
        self.pred_results = []
        self.pred_results_dict = {
            'y': [],
            'y_hat': [],

        }
        
        indx = 0
        
        for i in tqdm(range(len(self.test_data_list))):

            ## Subject id in the test data ##
            mi_id = self.test_data_list['MI_ID'][i]

            ## Obtain image ids and label y for each subject. ###
            img_id_list = ast.literal_eval(
                self.meta_data[self.meta_data['MI_ID'] == mi_id]['IMG_ID_LIST'].to_list()[0])
            y = self.meta_data[self.meta_data['MI_ID'] == mi_id]['liver_fatty'].to_list()[
                0]
            test_img_set = set(img_id_list)
            print(f"{2 ** len(img_id_list) - 1} samples will be generated for subj. {mi_id}")
            
            for j in range(1, len(img_id_list) + 1):
                test_img_combinations = list(itertools.combinations(test_img_set, j))
                new_test_img_combinations = test_img_combinations.copy()
                
                for k in test_img_combinations:
                    if len(k) > 1 and isinstance(k[0], tuple):
                        test_img_ids = self.__convert_tuple_to_list(k)
                        print('convert tuple to list because length > 1')
                    else:
                        test_img_ids = list(k)
                        
                        
                    y, y_hat = self.__single_prediction(mi_id=mi_id, img_id_list=test_img_ids, y=y)
                    crt = (y_hat == y)
                    self.accuracy += crt
                    indx_pred_result = {
                        "indx": indx,
                        "y": y,
                        "y_hat": y_hat,
                        "crt": crt,
                        "mi_id": mi_id,
                        "img_id_list": img_id_list,
                    }
                    self.pred_results.append(indx_pred_result)
                    self.pred_results_dict['y'].append(y)
                    self.pred_results_dict['y_hat'].append(y_hat)
                    indx += 1
                    
                    if not crt:
                        new_test_img_combinations.remove(k)
                        
                test_img_combinations = self.__flatten(new_test_img_combinations)
            
        gc.collect()
        del indx    
        del i
        del j 
        del k

        print('Accuracy is {:.4f}.'.format(
            self.accuracy / len(self.pred_results)))

        self.confusion_matrix_results = confusion_matrix(
            self.pred_results_dict['y'],
            self.pred_results_dict['y_hat'])

        self.tn, self.fp, self.fn, self.tp = self.confusion_matrix_results.ravel()
        print(
            f"There were {self.tp} true positives, {self.tn} true negatives, {self.fp} false positives, {self.fn} false negatives.")
        
        self.__output_pred_results()

    def __predict_fn(self, images: np.ndarray):
        """Prediction function used for LIME

        Parameters
        ----------
        images : np.ndarray
            _description_

        Returns
        -------
        probs
            Probabilities
        """
        image_features = []
        for img in images:
            img = self.transform(Image.fromarray(
                (img * 255).astype(np.uint8))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.pretrained_image_encoder(img)
            image_features.append(feature.squeeze(0))

        # Create a batch tensor from image features
        image_features = torch.stack(image_features)

        # Create dummy edge_index and batch
        num_images = len(images)
        edge_index = torch.tensor([[i, j] for i in range(num_images) for j in range(
            num_images) if i != j], dtype=torch.long).t().contiguous().to(self.device)
        batch = torch.arange(num_images, dtype=torch.long).to(self.device)

        # Create dummy Data object for graph encoder
        data = Data(x=image_features.to(self.device),
                    edge_index=edge_index,
                    y=torch.tensor([0]*num_images, dtype=torch.long).to(self.device))  # Dummy labels

        # Predict
        with torch.no_grad():
            # Assuming train_mask is 1
            output = self.graph_encoder(data.x, data.edge_index, batch, 1)
            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
        return probs

    def __safe_heatmap_conversion(self, heatmap):
        # Convert to a numpy array if it's not already
        heatmap_array = np.array(heatmap)

        # Create a mask for None values
        none_mask = heatmap_array == None

        # Convert to float, replacing None with np.nan
        heatmap_float = np.where(
            none_mask, np.nan, heatmap_array.astype(float))

        return heatmap_float

    def __explaination_heatmap(self, dict_heatmap: dict, explanation, i_img_save_dir: str, label, label_indx: int, img_indx: int):
        """Generates heatmap for LIME.

        Parameters
        ----------
        dict_heatmap : dict
            _description_
        explanation : _type_
            _description_
        i_img_save_dir : str
            Directory to save the heatmap plot
        label : _type_
            _description_
        label_indx : int
            Label of index used for saving the heatmap image
        img_indx : int
            Index of image used for saving the file 
        """

        # Map each explanation weight to the corresponding superpixel
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        heatmap_float = self.__safe_heatmap_conversion(heatmap)
        print(f"Heatmap dtype: {heatmap.dtype}")

        # Check if heatmap contains any non-nan values
        if np.any(~np.isnan(heatmap_float)):
            # Remove nan values for calculating vmax
            heatmap_no_nan = heatmap_float[~np.isnan(heatmap_float)]
            vmax = np.max(np.abs(heatmap_no_nan))

            # Plot. The visualization makes more sense if a symmetrical colorbar is used.
            plt.imshow(heatmap_float, cmap='RdBu', vmin=-vmax, vmax=vmax)
            plt.colorbar()
            plt.title(f"Heatmap for label {label}")
            plt.savefig(os.path.join(i_img_save_dir,
                        f"heatmap-{img_indx}-{label_indx}-label{label}.png"))
            plt.show()
        else:
            print(f"Skipping heatmap for label {label} due to no valid data")

    def explain_subject(self, img_filepaths: list, subj_id: str, segmentation: str = 'slic', regression: str = ''):
        """Explain a subject's predictions using LIME on its image files.

        Parameters
        ----------
        img_filepaths : list
            List of file paths to images, need to be absolute paths.
        subj_id : str
            _description_
        segmentation : str, optional
            Segementation algorithms, choices include 'slic', 'quickshift', and 'felzenszwalb', by default 'slic'
        regression : str, optional
            Regression algorithms, choices include 'huber', 'sgd', 'ridge'. By default 'ridge'
        """

        subj_id = subj_id.strip()
        segmentation = segmentation.strip().lower()
        regression = regression.strip().lower()

        assert (all(os.path.exists(i) for i in img_filepaths))

        if segmentation == 'slic':
            segmenter = SegmentationAlgorithm(algo_type="slic",
                                              enforce_connectivity=True,
                                              max_num_iter=50,
                                              compactnessfloat=10.0,
                                              n_segments=100,
                                              sigma=0.0,)
        elif segmentation == 'quickshift':
            segmenter = SegmentationAlgorithm(algo_type="quickshift",
                                              kernel_size=1,
                                              max_dist=2)
        elif segmentation == 'felzenszwalb':
            segmenter = SegmentationAlgorithm(algo_type="felzenszwalb",
                                              scale=1,
                                              min_size=10,
                                              sigma=0.8)
        else:
            # if segmentation == '' or segmentation is None:
            segmenter = None

        if regression == 'huber':
            model_regressor = HuberRegressor()
        elif regression == 'sgd':
            model_regressor = SGDRegressor()
        else:
            # if regression == '' or regression is None or regression == 'ridge':
            model_regressor = None

        images = np.array([plt.imread(i) for i in img_filepaths])
        assert (len(images) == len(img_filepaths))

        seg_str = segmentation
        if segmenter is None:
            seg_str = "noseg"

        reg_str = regression
        if model_regressor is None:
            reg_str = "ridge"

        result_timestamp = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')

        subj_result_dir = os.path.join(
            self.result_dir, f"{subj_id}-{seg_str}-{reg_str}-{result_timestamp}")
        assert (not os.path.exists(subj_result_dir))
        os.mkdir(subj_result_dir)

        for img_indx, i_img in enumerate(images):

            explanation = self.explainer.explain_instance(i_img,
                                                          self.__predict_fn,
                                                          top_labels=20,
                                                          hide_color=0,
                                                          num_features=100,
                                                          num_samples=1000,
                                                          segmentation_fn=segmenter,
                                                          model_regressor=model_regressor,)

            i_img_save_dir = os.path.join(subj_result_dir, f"{img_indx}/")
            os.mkdir(i_img_save_dir)
            plt.imshow(i_img, cmap='gray')
            plt.savefig(os.path.join(i_img_save_dir,
                        f"original-{img_indx}.png"))
            plt.show()

            self.__visualize_lime(explanation=explanation,
                                  i_img_save_dir=i_img_save_dir,
                                  img_indx=img_indx)

    def __output_pred_results(self):
        self.pred_results_df = pd.DataFrame(self.pred_results)
        pred_results_csv_path = os.path.join(
            self.result_dir, f"pred_results{self.test_data_id}.csv")
        if os.path.exists(pred_results_csv_path):
            prev_csv_date = datetime.datetime.fromtimestamp(os.path.getmtime(pred_results_csv_path)).strftime('%Y-%m-%d %H:%M:%S')
            print(
                f"Previous prediction results for {self.test_data_id}, {pred_results_csv_path} was last modified at {prev_csv_date}")
            
            prev_df = pd.read_csv(pred_results_csv_path)
            prev_df.to_csv(f"pred_results{self.test_data_id}-{prev_csv_date}.csv", index=None)
            
        self.pred_results_df.to_csv(pred_results_csv_path)
        print(f"Predictions saved to {pred_results_csv_path}")

    def __visualize_lime(self, explanation, i_img_save_dir: str, img_indx: int):
        for label_indx, label in enumerate(explanation.top_labels):

            temp, mask = explanation.get_image_and_mask(
                label, positive_only=False, num_features=10, hide_rest=False)
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask), cmap='gray')
            plt.imshow(temp, cmap='gray')
            plt.title(f"Top label: {label}")
            plt.savefig(os.path.join(i_img_save_dir,
                        f"10-{img_indx}-{label_indx}-label{label}.png"))
            plt.show()

            temp, mask = explanation.get_image_and_mask(
                label, positive_only=False, num_features=100, hide_rest=False)
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask), cmap='gray')
            plt.imshow(temp, cmap='gray')
            plt.title(f"Top label: {label}")
            plt.savefig(os.path.join(i_img_save_dir,
                        f"100-{img_indx}-{label_indx}-label{label}.png"))
            plt.show()

            self.__explaination_heatmap(dict_heatmap=dict(explanation.local_exp[label]),
                                        explanation=explanation,
                                        i_img_save_dir=i_img_save_dir,
                                        label=label,
                                        label_indx=label_indx,
                                        img_indx=img_indx)

    def explain_tps(self, segmentation_algo="slic", regression_algo="ridge"):
        """Explain all true positives. True positives are predicted by the `self.predict_on_test` method.

        Parameters
        ----------
        segmentation_algo : str, optional
            Segementation algorithms, choices include 'slic', 'quickshift', and 'felzenszwalb', by default 'slic'
        regression_algo : str, optional
            Regression algorithms, choices include 'huber', 'sgd', 'ridge'. By default 'ridge'
        """

        assert (self.tp_mid is not None and self.tp_mid_img_list is not None)
        if len(self.tp_mid_img_list) != len(self.tp_mid):
            print("Warning: tp_mid and tp_mid_img_list do not have the same length")

        for tp_mid in tqdm(self.tp_mid):
            tp_mid_img_list = self.tp_mid_img_list.get(tp_mid)
            if tp_mid_img_list is None:
                print(
                    f'Skipping over {tp_mid} because the image filepaths were not found')
                continue

            tp_mid_img_list = [os.path.join(
                self.test_data_dir, f"{tp_mid}_{i}.jpg") for i in tp_mid_img_list]
            if not all([os.path.exists(i) for i in tp_mid_img_list]):
                print(
                    f'Skipping over {tp_mid} because some of the image filepaths were not found')
                continue

            self.explain_subject(img_filepaths=tp_mid_img_list,
                                 subj_id=tp_mid,
                                 segmentation=segmentation_algo,
                                 regression=regression_algo)
