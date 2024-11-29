import ast
import datetime
import glob
import os
import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from classifiers import RidgeClassifierWithStats

load_dotenv()

@dataclass
class RidgeRun:
    """
    A class to perform Ridge Regression analysis on image data.

    Attributes:
    -----------
    crop_image_dir : str
        Directory path where cropped images are stored.
    test_data_id : str
        Identifier for the test data.
    metadata_name : str
        Path to the metadata CSV file.
    result_dir : str
        Directory path where results will be saved.
    all_subj_save_dir : str
        Directory path where all subject results will be saved.
    n_bootstrap_iterations : int
        Number of bootstrap iterations for the Ridge regression.

    Methods:
    --------
    __post_init__():
        Initializes additional attributes and creates necessary directories.
    get_selected_mi_ids():
        Retrieves selected MI IDs based on certain criteria.
    select_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
        Selects unique columns from a DataFrame.
    run():
        Executes the Ridge regression analysis and saves the results.
    """
    crop_image_dir: str = field(default_factory=lambda: os.getenv('CROP_IMAGE_DIR_PATH'))
    test_data_id: str = '09'
    metadata_name: str = 'meta_data/TWB_ABD_expand_modified_gasex_21072022.csv'
    result_dir: str = "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/07-12-2024-03-57-58/"
    all_subj_save_dir: str = field(init=False)
    n_bootstrap_iterations: int = 50000

    def __post_init__(self):
        current_timestamp = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        self.all_subj_save_dir = os.path.join("/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results", f"ridge-{current_timestamp}")
        # self.all_subj_save_dir = "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/ridge-08-06-2024-01-24-40"
        if not os.path.exists(self.all_subj_save_dir):
            os.mkdir(self.all_subj_save_dir)
        self.test_data_list_name = f'fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset{self.test_data_id}/test_dataset{self.test_data_id}.csv'
        self.test_data_list = pd.read_csv(self.test_data_list_name)
        self.meta_data = pd.read_csv(self.metadata_name, sep=",")
        self.selected_mi_ids = self.get_selected_mi_ids()

    def get_selected_mi_ids(self):
        """
        Get the set of selected MI_IDs based on specific criteria.

        This method filters the MI_IDs from the test data list based on the following criteria:
        1. The 'liver_fatty' value in the meta data for the MI_ID is greater than 0.
        2. The length of the 'IMG_ID_LIST' in the meta data for the MI_ID is greater than or equal to 20.

        Returns:
            set: A set of MI_IDs that meet the specified criteria.
        """
        ground_truth_pos_mi_ids = [mi_id for mi_id in self.test_data_list['MI_ID'] if self.meta_data[self.meta_data['MI_ID'] == mi_id]['liver_fatty'].to_list()[0] > 0]
        selected_mi_ids = [mi_id for mi_id in ground_truth_pos_mi_ids if len(ast.literal_eval(self.meta_data[self.meta_data['MI_ID'] == mi_id]['IMG_ID_LIST'].to_list()[0])) >= 20]
        return set(selected_mi_ids)

    @staticmethod
    def select_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Select unique columns from a DataFrame.

        This function transposes the input DataFrame, removes duplicate rows (which correspond to duplicate columns in the original DataFrame), and then transposes it back to return a DataFrame with unique columns.

        Parameters:
        df (pd.DataFrame): The input DataFrame from which to select unique columns.

        Returns:
        pd.DataFrame: A DataFrame containing only unique columns from the input DataFrame.
        """
        df_t = df.T
        df_unique = df_t.drop_duplicates()
        return df_unique.T

    def run(self):
        """
        Executes the ridge regression analysis on prediction results.

        This method performs the following steps:
        1. Identifies CSV files containing prediction results.
        2. Filters out subjects that have already been processed.
        3. Reads and processes the prediction results for each subject.
        4. For each subject, if the predictions are not all the same, it fits a Ridge Classifier model.
        5. Saves the model summary and plots the results.

        Raises:
            KeyboardInterrupt: If the process is interrupted by the user.
            Exception: If there is an issue with fitting the Ridge Classifier model.

        Prints:
            - The directory where results will be saved.
            - Skips subjects with uniform predictions.
            - Skips subjects if there are issues with the Ridge Classifier model.
            - The final directory where all results are saved.
        """
        if self.result_dir.endswith("/"):
            csv_paths = glob.glob(self.result_dir + "*/pred_results.csv")
        else:
            csv_paths = glob.glob(self.result_dir + "/*/pred_results.csv")

        print(f"Results will be saved to {self.all_subj_save_dir}")
        mi_ids = [i.split("/pred_results.csv")[0].split("/")[-1] for i in csv_paths]
        completed_subj = {f.name for f in os.scandir(self.all_subj_save_dir) if f.is_dir()}
        selected_mi_ids = self.selected_mi_ids - completed_subj
        df_dict = {mi_ids[i]: csv_paths[i] for i, _ in enumerate(mi_ids)}
        df_dict = {k: pd.read_csv(v) for k, v in df_dict.items()}
        df_dict = {k: v.drop_duplicates() for k, v in df_dict.items()}

        miid_imgid_dict = {mi_id: ast.literal_eval(self.meta_data[self.meta_data['MI_ID'] == mi_id]['IMG_ID_LIST'].to_list()[0]) for mi_id in mi_ids}

        for i, (mi_id, df) in tqdm(enumerate(df_dict.items()), total=len(selected_mi_ids)):
            if mi_id not in selected_mi_ids:
                continue
            if len(df['yhat'].unique()) < 2:
                print(f"{mi_id} was skipped because all y_hat were the same_values")
                continue

            print(mi_id)
            X_df = df.drop(['yhat', 'y'], axis=1).copy()
            X_df = self.select_unique_columns(X_df)
            y_df = df[['yhat']].copy()
            y_ = y_df.to_numpy().ravel()
            minority_class_size = min(y_df.value_counts())
            n_splits = min(minority_class_size, 10)

            img_filepaths = [os.path.join(self.crop_image_dir, f"{mi_id}_{img_id}.jpg") for img_id in X_df.columns]
            try:
                model = RidgeClassifierWithStats(n_alphas=100, n_bootstrap=10000, n_jobs=-1)
                model.custom_fit(X_df, y_, n_splits=n_splits)
            except KeyboardInterrupt as e:
                exc_type, _, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e, exc_type, fname, exc_tb.tb_lineno)
                sys.exit(0)
            except Exception as e:
                exc_type, _, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e, exc_type, fname, exc_tb.tb_lineno)
                print(f"{mi_id} was skipped because Ridge is having some issues")
                continue
            except:
                print(f"{mi_id} was skipped because Ridge is having some issues")
                continue

            save_dir = os.path.join(self.all_subj_save_dir, mi_id)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            summary_df = model.summary()
            summary_df.to_csv(os.path.join(save_dir, "ridge_coefficients.csv"), index=None)
            model.plot_results(img_filepaths, summary_df=summary_df, save_dir=save_dir)

        print(f"Results saved to {self.all_subj_save_dir}")

if __name__ == "__main__":
    ridge_run = RidgeRun()
    ridge_run.run()
