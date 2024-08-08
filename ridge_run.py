import ast
import datetime
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from classifiers import RidgeClassifierWithStats

load_dotenv()

CROP_IMAGE_DIR = os.getenv('CROP_IMAGE_DIR_PATH')
test_data_id = '09'
metadata_name = 'meta_data/TWB_ABD_expand_modified_gasex_21072022.csv'
test_data_list_name = 'fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset'+str(test_data_id)+'/test_dataset'+str(test_data_id)+'.csv'
test_data_list = pd.read_csv(test_data_list_name)
meta_data = pd.read_csv(metadata_name, sep=",")

ground_truth_pos_mi_ids = [mi_id
         for mi_id in test_data_list['MI_ID'] 
         if meta_data[meta_data['MI_ID']==mi_id]['liver_fatty'].to_list()[0]  > 0 ]
selected_mi_ids = [mi_id for mi_id in ground_truth_pos_mi_ids
                   if len(ast.literal_eval(meta_data[meta_data['MI_ID']==mi_id]['IMG_ID_LIST'].to_list()[0])) >= 20
                   ]
selected_mi_ids = set(selected_mi_ids)

def select_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Transpose the dataframe to compare columns
    df_t = df.T

    # Drop duplicate rows (which were originally columns)
    df_unique = df_t.drop_duplicates()
    
    # Transpose back to original orientation
    return df_unique.T

result_dir = "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/07-12-2024-03-57-58/"
current_timestamp = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
all_subj_save_dir = os.path.join("/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results",f"ridge-{current_timestamp}")
all_subj_save_dir = "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/ridge-08-06-2024-01-24-40"
if not os.path.exists(all_subj_save_dir):
    os.mkdir(all_subj_save_dir)
if result_dir.endswith("/"):
    csv_paths = glob.glob(result_dir+ "*/pred_results.csv")
else:
    csv_paths = glob.glob(result_dir+ "/*/pred_results.csv")
    
print(f"Results will be saved to {all_subj_save_dir}")
mi_ids = [i.split("/pred_results.csv")[0].split("/")[-1] for i in csv_paths]
completed_subj = [f.name for f in os.scandir(all_subj_save_dir) if f.is_dir()]
completed_subj = set(completed_subj)
selected_mi_ids = selected_mi_ids - completed_subj
df_dict = {mi_ids[i]: csv_paths[i] for i, _ in enumerate(mi_ids)}
df_dict = {k: pd.read_csv(v) for k, v in df_dict.items()}
df_dict = {k: v.drop_duplicates() for k, v in df_dict.items()}

miid_imgid_dict = {mi_id: ast.literal_eval(meta_data[meta_data['MI_ID']==mi_id]['IMG_ID_LIST'].to_list()[0]) 
                   for mi_id in mi_ids}

n_bootstrap_iterations = 50000

for i, (mi_id, df) in tqdm(enumerate(df_dict.items()), total = len(selected_mi_ids)):
    if mi_id not in selected_mi_ids:
        continue
    if len(df['yhat'].unique()) < 2:
        print(f"{mi_id} was skipped because all y_hat were the same_values")
        continue
    
    print(mi_id)
    
    # if mi_id == "P0009584" or mi_id == "P0007360": 
    #     # these mi_id have some problems
    #     continue 
    
    
    print(mi_id)
    X_df = df.drop(['yhat', 'y'], axis=1).copy()
    X_df = select_unique_columns(X_df)
    y_df = df[['yhat']].copy()
    y_ = y_df.to_numpy().ravel()
    minority_class_size = min(y_df.value_counts())
    n_splits = min(minority_class_size, 10)

    img_filepaths = [os.path.join(CROP_IMAGE_DIR, f"{mi_id}_{img_id}.jpg") for img_id in X_df.columns]
    try: 
        model = RidgeClassifierWithStats(n_alphas=100, n_bootstrap=10000, n_jobs=-1,)
        model.custom_fit(X_df, y_, n_splits=n_splits,)
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

    save_dir = os.path.join(all_subj_save_dir, mi_id)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Print summary
    summary_df = model.summary()
    summary_df.to_csv(os.path.join(save_dir, "ridge_coefficients.csv"),
                   index=None)

    model.plot_results(img_filepaths, summary_df = summary_df, 
                       save_dir=save_dir)
    
print(f"Results saved to {all_subj_save_dir}")