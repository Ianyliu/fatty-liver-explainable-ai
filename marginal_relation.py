import ast
import datetime
import gc
import glob
import inspect
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Union

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


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

def select_unique_columns(df):
    # Transpose the dataframe to compare columns
    df_t = df.T

    # Drop duplicate rows (which were originally columns)
    df_unique = df_t.drop_duplicates()
    
    # Transpose back to original orientation
    return df_unique.T



def offset_image(x, y, img_path, bar_is_too_short, ax, zoom=0.06, vertical=False):
    img = plt.imread(img_path)
        
    im = OffsetImage(img, zoom=zoom, cmap='gray')
    im.image.axes = ax
    
    if vertical:
            
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

def plot_results(img_filepaths, summary_df, save_dir = None):

    summary_df = summary_df.copy()
        
    if save_dir is not None and not os.path.exists(save_dir):
        try:
            os.mkdir(save_dir)
        except OSError as e:
            raise ValueError(f"Save directory does not exist: {save_dir}")

    if len(img_filepaths) != summary_df.shape[0]:
        raise ValueError("Length of img_filepaths does not match the number of ElasticNet features (rows) in the summary DataFrame.")
    
    plt.figure(figsize=(15,40)) 

    labels = list(summary_df.index)
    values = [summary_df["corrs"][i] 
                for i in range(len(labels))
                ]
    upper_CIs = [summary_df["upper_CIs"][i] 
                for i in range(len(labels))
                ]
    lower_CIs = [summary_df["lower_CIs"][i] 
                for i in range(len(labels))
                ]
    ses = [summary_df["corr_stds"][i] 
                for i in range(len(labels))
                ]
    sesignificant = [summary_df["corr_significance"][i] for i in range (len(labels))]

    sorted_img_filepaths = img_filepaths.copy()
        
    sorted_data = [(val, label, upper_CI, lower_CI, se, img_path, significance) for val, label, upper_CI, lower_CI, se, img_path, significance in 
                sorted(zip(values, labels, upper_CIs, lower_CIs, ses, sorted_img_filepaths, sesignificant), 
                                            key = lambda pair: pair[0], 
                                            reverse= True)]
    values, labels, upper_CIs, lower_CIs, ses, sorted_img_filepaths, sesignificant = zip(*sorted_data)
    del sorted_data

    def get_color(val, significance):
        if val > 0 and significance == "SIGNIFICANT":
            return "deepskyblue"
        elif val < 0 and significance == "SIGNIFICANT":
            return "salmon"
        elif val > 0 and significance != "SIGNIFICANT":
            return "skyblue"
        elif val < 0 and significance != "SIGNIFICANT":
            return "mistyrose"
        else:
            return "white"
        
    colors = [get_color(val, sesignificant[indx]) for indx, val in enumerate(values)]

    zoom = 0.065


    height = 0.8

    bar_labels = [f"{values[indx]:.2f}±{se:.3f}" for indx, se in enumerate(ses)]

    for indx, val in enumerate(values):
        plt.text(val, indx, bar_labels[indx],
                    va='center',
                    )
            
    plt.barh(y=labels, width=values, 
                height=height, color=colors, 
                align='center', alpha=0.7, 
                xerr = ses, ecolor='silver',
                error_kw=dict(lw=3,),
            )

    if isinstance(values, np.ndarray):
        max_value = values.max()
    elif isinstance(values, (list, tuple, set)):
        max_value = max(values)
        
    ax = plt.gca()
    for indx, (label, value) in enumerate(zip(labels, values)):
        img_abs_filepath = sorted_img_filepaths[indx]
        offset_image(x = value, 
                        img_path = img_abs_filepath, 
                        y = label, 
                        bar_is_too_short=value < max_value / 10, 
                        zoom=zoom,
                        ax=ax,)
    plt.xlabel = "Correlation Coefficients"
    plt.subplots_adjust(left=0.15)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "hbar.png"))
    plt.show()
    plt.clf()

    fig_width = len(labels) + len(labels)/4
    fig_width = max(fig_width, 13)
    fig_height = len(labels) // 2
    fig_height = max(fig_height, 8)
    plt.figure(figsize=(fig_width,fig_height))            

    bar_labels = [f"{values[indx]:.2f}\n±{se:.3f}" for indx, se in enumerate(ses)]


    zoom = 0.25 / 4
    plt.bar(x=labels, height=values, 
            width=0.8, color=colors, 
            align='center', alpha=0.8, 
            yerr=ses, ecolor='lightgray', 
            error_kw=dict(lw=3,),
            )

    for indx, val in enumerate(values):
        plt.text(indx, val, bar_labels[indx],
                    ha='center',
                    )
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(-0.5, len(labels) - 0.6)
        
    for indx, (label, value) in enumerate(zip(labels, values)):
        img_abs_filepath = sorted_img_filepaths[indx]
        offset_image(y = value, img_path = img_abs_filepath, 
                        x = label, 
                        bar_is_too_short=value < max_value / 10, 
                        ax=ax, 
                        zoom=zoom, 
                        vertical=True)
        
    # Set x-axis ticks and labels with larger font size
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=15)
    ax.tick_params(axis='y', labelsize=15) 
    plt.xticks(rotation=65, ha='right')
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(bottom=0.2)
    ax.set_ylabel("Correlation Coefficients", fontsize=16,)  # Adjust the title and font size as needed

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir,
                    f"vbar.png"))
    plt.show()
    plt.clf()



def corr_pipeline():
    result_dir = "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/07-12-2024-03-57-58/"
    current_timestamp = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
    all_subj_save_dir = os.path.join("/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results",f"correlation-old-dataset-{current_timestamp}")
    os.mkdir(all_subj_save_dir)
    if result_dir.endswith("/"):
        csv_paths = glob.glob(result_dir+ "*/pred_results.csv")
    else:
        csv_paths = glob.glob(result_dir+ "/*/pred_results.csv")
        
    print(f"Results will be saved to {all_subj_save_dir}")
    mi_ids = [i.split("/pred_results.csv")[0].split("/")[-1] for i in csv_paths]
    df_dict = {mi_ids[i]: csv_paths[i] for i, _ in enumerate(mi_ids)}
    df_dict = {k: pd.read_csv(v) for k, v in df_dict.items()}
    df_dict = {k: v.drop_duplicates() for k, v in df_dict.items()}

    miid_imgid_dict = {mi_id: ast.literal_eval(meta_data[meta_data['MI_ID']==mi_id]['IMG_ID_LIST'].to_list()[0]) 
                    for mi_id in mi_ids}

    for i, (mi_id, df) in tqdm(enumerate(df_dict.items()), total = len(selected_mi_ids)):
        if mi_id not in selected_mi_ids:
            continue 
        
        if len(df['yhat'].unique()) < 2:
            print(f"{mi_id} was skipped because all y_hat were the same_values")
            continue
                
        X_df = df.drop(['yhat', 'y'], axis=1).copy()
        X_df = select_unique_columns(X_df)
        y_df = df[['yhat']].copy()
        y_ = y_df.to_numpy().ravel()

        img_filepaths = [os.path.join(CROP_IMAGE_DIR, f"{mi_id}_{img_id}.jpg") for img_id in X_df.columns]
        num_img = X_df.shape[1]
        
        corr_dict = {}
        corrs, corr_stds, upper_CIs, lower_CIs, corr_CIs, corr_p_vals = [None] * num_img, [None] * num_img, [None] * num_img, [None] * num_img, [None] * num_img, [None] * num_img    
        for indx, img in enumerate(X_df.columns):
            img_col = X_df[img].to_numpy()
            unique_img_col_val = np.unique(img_col)
            if len(unique_img_col_val) == 1:
                print(f"WARNING: {mi_id} sampling results yielded only unique values for an image column")
                print(f"Unique img_col values: {unique_img_col_val}")
                corr = 0.0 
                corr_std = 1.0
                upper_CI = 1.0
                lower_CI = -1.0
                corr_CI = (lower_CI, upper_CI)
                corr_p_val = 1.0
                corrs[indx] = corr
                corr_stds[indx] = corr_std
                upper_CIs[indx] = upper_CI
                lower_CIs[indx] = lower_CI
                corr_CIs[indx] = corr_CI
                corr_p_vals[indx] = corr_p_val
                continue
                
            corr, corr_p_val = pearsonr(img_col, y_)
            corr_std = np.sqrt((1- corr ** 2) / (num_img-2))
            upper_CI = min(corr + corr_std, 1.0)
            lower_CI = max(corr - corr_std, -1.0)
            corr_CI = (lower_CI, upper_CI)
            
            corrs[indx] = corr
            corr_stds[indx] = corr_std
            upper_CIs[indx] = upper_CI
            lower_CIs[indx] = lower_CI
            corr_CIs[indx] = corr_CI
            corr_p_vals[indx] = corr_p_val
            
        corr_dict["corrs"] =corrs
        corr_dict["corr_stds"] =corr_stds
        corr_dict["upper_CIs"] =upper_CIs
        corr_dict["lower_CIs"] =lower_CIs
        corr_dict["corr_CIs"] =corr_CIs
        corr_dict["corr_p_vals"] =corr_p_vals
        corr_dict["corr_significance"] = ['INSIGNIFICANT' if p_val >= 0.05 
                                                else 'SIGNIFICANT'
                                                for _, p_val in enumerate(corr_p_vals) ]
        corr_dict["IMG"] = X_df.columns
        
        save_dir = os.path.join(all_subj_save_dir, mi_id)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        summary_df = pd.DataFrame.from_dict(corr_dict) 
        summary_df.set_index("IMG", inplace = True)
        summary_df.to_csv(os.path.join(save_dir, "correlations.csv"))
        
        plot_results(img_filepaths=img_filepaths,
                    summary_df = summary_df,
                    save_dir = save_dir,)
        
if __name__ == "__main__":
    corr_pipeline()