import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import ast
import pandas as pd 
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
from tqdm import tqdm
import gc
from glmnet import LogitNet
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.model_selection import LeaveOneOut, KFold
from typing import Union, Callable
from dataclasses import dataclass, field

@dataclass
class ElasticNetClassifierWithStats:
    
    n_alphas: int = 100
    alphas: Union[list, set, np.ndarray, tuple] = field(init=False)
    n_bootstrap: int = 250
    ci_level: float = 0.95
    n_jobs: int = -1
    best_alpha: float = field(init=False)
    best_lambda: float = field(init=False)
    ci_high_: float = field(init=False)
    ci_low_: float = field(init=False)
    bootstrap_coefs: np.ndarray = field(init=False)
    se_: float = field(init=False)
    _model: Callable = field(init=False)
    coef_: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    feature_names_in_: Union[list, tuple, np.ndarray] = field(init=False)
    candidate_models: list = field(default_factory=list, init=False)
    verbose: bool = True
    summary_df: pd.DataFrame = field(init=False)
    
    def __post_init__(self):
        self.alphas = np.linspace(0.01, 1.0, num = self.n_alphas)
        pass
        # self._model = LogitNet(fit_intercept=True, n_lambda=self.n_lambda, alpha = 0.5, 
        #      standardize=False, n_splits=10, scoring= 'roc_auc', 
        #      n_jobs = self.n_jobs, random_state=0)
        
    def custom_fit(self, X, y, n_splits: int = 10):
        
        best_alpha = None
        best_auc_score = 0
        best_model_accuracy = None
        self.candidate_models = [None] * len(self.alphas)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        for indx, alpha in tqdm(enumerate(self.alphas), total= len(self.alphas)):
            
            enet_model = LogitNet(fit_intercept=True, n_lambda=100, alpha = alpha, 
                                  n_splits=n_splits, standardize=False, scoring= 'roc_auc', 
                                  n_jobs = self.n_jobs, random_state=0)
            enet_model.fit(X_train,y_train)
            self.candidate_models[indx] = enet_model
            
            enet_pred = enet_model.decision_function(X_test)
            auc_score = roc_auc_score(y_test, enet_pred)
            
            if auc_score >= best_auc_score:
                best_auc_score = auc_score
                best_alpha = alpha
                best_model_accuracy = enet_model.score(X_test,y_test).tolist()
                
        print(f"Best ROC-AUC: {best_auc_score}",)
        print(f"Accuracy from best alpha: {best_model_accuracy}",)
        if self.verbose:
            print(f"Refitting model with best alpha {best_alpha}")
        self._model = LogitNet(fit_intercept=True, n_lambda=1000, alpha = best_alpha, 
                    standardize=False, scoring= 'roc_auc', 
                    n_jobs = self.n_jobs, random_state=0, n_splits = n_splits * 2)    
        self._model.fit(X, y) 
           
        self.X = X
        self.y = y
        self._model.fit(self.X, self.y)
        if isinstance(self.X, pd.DataFrame):
            self.feature_names_in_ = self.X.columns
        self.best_alpha = best_alpha
        self.best_lambda = self._model.lambda_best_[0]
        self.coef_ = self._model.coef_[0]
        print("Best alpha value: ", self.best_alpha)
        print("Best lambda value: ", self.best_lambda)
        print("Intercept value: ", self._model.intercept_)
        self._bootstrap()
        return self
    
    def _bootstrap(self, n_splits: int = 10):
        if self.verbose:
            print("Boostrapping...")
        def bootstrap_fit(X, y):
            indices = np.random.choice(len(X), len(X), replace=True)
            if isinstance(X, pd.DataFrame):
                X_boot = X.iloc[indices]
                # print(X_boot.shape, X_boot.drop_duplicates().shape, X.shape, X.drop_duplicates().shape)
            else:
                X_boot = X[indices]
            if isinstance(y, pd.DataFrame):
                y_boot = y.iloc[indices]
            else:
                y_boot = y[indices]
                
            bootstrap_model = LogitNet(fit_intercept=True, n_lambda=100, alpha = self.best_alpha, 
                    standardize=False, scoring= 'roc_auc', n_splits = n_splits,
                    n_jobs = self.n_jobs, random_state=0)
            bootstrap_model.fit(X_boot, y_boot)
            return bootstrap_model.coef_[0]
        
        bootstrap_results = Parallel(n_jobs=self.n_jobs)(
            delayed(bootstrap_fit)(self.X, self.y) for _ in range(self.n_bootstrap)
        )
        
        self.bootstrap_coefs = np.array(bootstrap_results)
        self.se_ = np.std(self.bootstrap_coefs, axis=0, ddof=1)
        self.ci_low_ = np.percentile(self.bootstrap_coefs, (1 - self.ci_level) / 2 * 100, axis=0)
        self.ci_high_ = np.percentile(self.bootstrap_coefs, (1 + self.ci_level) / 2 * 100, axis=0)
        
    def custom_predict(self, X):
        return self._model.predict(X)
    
    def custom_predict_proba(self, X):
        return self._model.decision_function(X)
    
    def custom_score(self, X, y):
        return self._model.score(X, y).tolist()
    
    def visualize_alphas(self):
        
        candidate_alphas = [candidate_model.alpha for candidate_model in tqdm(self.candidate_models)]
        candidate_coefs = [candidate_model.coef_[0] for candidate_model in tqdm(self.candidate_models)]
            
        ax = plt.gca()

        ax.plot(candidate_alphas, candidate_coefs)
        ax.set_xscale("linear")
        plt.xlabel("alpha")
        plt.ylabel("weights")
        plt.title("ElasticNet classifier coefficients as a function of the regularization")
        plt.axis("tight")
        plt.show()
        
        # ax = plt.gca()

        # ax.plot(candidate_alphas, candidate_coefs)
        # ax.set_xscale("log")
        # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
        # plt.xlabel("alpha")
        # plt.ylabel("weights")
        # plt.title("ElasticNet classifier coefficients as a function of the regularization")
        # plt.axis("tight")
        # plt.show()
    
    def summary(self):
        # p_values = np.mean(np.abs(self.coef_ - np.mean(self.bootstrap_coefs, axis=0)) <= 
        #                    np.abs(self.bootstrap_coefs - np.mean(self.bootstrap_coefs, axis=0)), axis=0)
        # np.mean(np.abs(self.bootstrap_coefs - np.mean(self.bootstrap_coefs)) >= np.abs(self.coef_ - np.mean(self.bootstrap_coefs)))
        
        try:
            feature_names = self.feature_names_in_
        except AttributeError as e:
            feature_names = np.arange(len(self.coef_))
            
        summary_df = pd.DataFrame()
        summary_df['Estimate'] = self.coef_
        summary_df['SE'] = self.se_
        summary_df['BoostrapLowerCI'] = self.ci_low_
        summary_df['BoostrapUpperCI'] = self.ci_high_
        summary_df['SELowerCI'] = self.coef_ - 1.96 * self.se_ 
        summary_df['SEUpperCI'] = self.coef_ + 1.96 * self.se_
        summary_df['BootstrapSignificance'] = ['INSIGNIFICANT' if self.ci_low_[i] < 0.0 < self.ci_high_[i] 
                                                else 'SIGNIFICANT'
                                                for i, _ in enumerate(self.coef_) ]
        summary_df['SESignificance'] = ['INSIGNIFICANT' if summary_df['SELowerCI'][i] <= 0.0 <= summary_df['SEUpperCI'][i] 
                                                else 'SIGNIFICANT'
                                                for i, _ in enumerate(self.coef_) ]
        summary_df.index = feature_names
        
        self.summary_df = summary_df.copy()

        return summary_df
    
    def plot_results(self, img_filepaths, summary_df = None, save_dir = None):
        if self.summary_df is None and summary_df is None:
            raise ValueError("No summary DataFrame available. Use summary() method to create one.")
        
        if summary_df is None and self.summary_df is not None: 
            summary_df = self.summary_df.copy()
            
        if save_dir is not None and not os.path.exists(save_dir):
            raise ValueError(f"Save directory does not exist: {save_dir}")

        if len(img_filepaths) != summary_df.shape[0]:
            raise ValueError("Length of img_filepaths does not match the number of ElasticNet features (rows) in the summary DataFrame.")
        

        plt.figure(figsize=(15,40)) 

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

        labels = list(summary_df.index)
        values = [summary_df["Estimate"][i] 
                    for i in range(len(labels))
                    ]
        upper_CIs = [summary_df["BoostrapUpperCI"][i] 
                    for i in range(len(labels))
                    ]
        lower_CIs = [summary_df["BoostrapLowerCI"][i] 
                    for i in range(len(labels))
                    ]
        ses = [summary_df["SE"][i] 
                    for i in range(len(labels))
                    ]
        sesignificant = [summary_df["SESignificance"][i] for i in range (len(labels))]

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
        plt.xlabel = "ElasticNet Classification Coefficients"
        plt.subplots_adjust(left=0.15)
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
        ax.set_ylabel("ElasticNet Classification Coefficients", fontsize=16,)  # Adjust the title and font size as needed

        plt.savefig(os.path.join(save_dir,
                        f"vbar.png"))
        plt.show()
        plt.clf()