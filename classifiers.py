import ast
import gc
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from glmnet import LogitNet
from joblib import Parallel, delayed
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.linear_model import (LogisticRegressionCV, RidgeClassifier,
                                  RidgeClassifierCV)
from sklearn.metrics import (classification_report, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from tqdm import tqdm

from usflc_xai import datasets, models


@dataclass
class GLMNET_ElasticNetClassifierWithStats:
    
    n_alphas: int = 100
    alphas: Union[list, set, np.ndarray, tuple] = field(init=False, default=None)
    n_bootstrap_iterations: int = 250
    ci_level: float = 0.95
    n_jobs: int = -1
    best_alpha: float = field(init=False)
    best_lambda: float = field(init=False)
    best_auc_score: float = field(init=False)
    ci_high_: float = field(init=False)
    ci_low_: float = field(init=False)
    bootstrap_coefs: np.ndarray = field(init=False)
    se_: float = field(init=False)
    _model: Callable = field(init=False)
    coef_: np.ndarray = field(init=False)
    feature_names_in_: Union[list, tuple, np.ndarray] = field(init=False)
    candidate_models: list = field(default_factory=list, init=False)
    verbose: bool = True
    summary_df: pd.DataFrame = field(init=False)
    X: Union[pd.DataFrame, np.ndarray] = field(init=False)
    y: Union[pd.DataFrame, np.ndarray] = field(init=False)
    
    def __post_init__(self):
        if self.alphas is None:
            self.alphas = np.linspace(0.01, 1.0, num = self.n_alphas)
        
    def fit(self, X, y, n_splits: int = 10):
        
        if X is None or not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError(f"X must be pd.DataFrame or np.ndarray but is {type(X)}")
        
        if y is None or not isinstance(y, (np.ndarray, pd.DataFrame)):
            raise TypeError(f"y must be pd.DataFrame or np.ndarray but is {type(y)}")
        
        if n_splits is None or not isinstance(n_splits, int):
            raise TypeError(f"n_splits must be type int but is {type(n_splits)}")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must be of same length but X is {len(X)} and y is of length {len(y)}")
        
        self.n_splits = n_splits
        self.X, self.y = X, y
        self._set_feature_names()
        self._select_best_model()
        self._fit_final_model()
                
        self._bootstrap()
        return self
    
    def _select_best_model(self):
        
        # best_alpha = None
        # best_auc_score = 0
        # best_model_accuracy = None
        self.candidate_models = [None] * len(self.alphas)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.05, random_state=42, stratify=self.y)
        
        for indx, alpha in tqdm(enumerate(self.alphas), total= len(self.alphas)):
            
            enet_model = LogitNet(fit_intercept=True, n_lambda=100, alpha = alpha, 
                                  n_splits=self.n_splits, standardize=False, scoring= 'roc_auc', 
                                  n_jobs = self.n_jobs, random_state=0)
            enet_model.fit(X_train,y_train)
            self.candidate_models[indx] = enet_model
            
            # enet_pred = enet_model.decision_function(X_test)
            # auc_score = roc_auc_score(y_test, enet_pred)
            
            # if auc_score >= best_auc_score:
            #     best_auc_score = auc_score
            #     best_alpha = alpha
            #     best_model_accuracy = enet_model.score(X_test,y_test).tolist()
                
        auc_scores = np.array([roc_auc_score(y_test, model.decision_function(X_test)) 
                       for model in self.candidate_models])
        best_index = np.argmax(auc_scores)
        best_alpha = self.alphas[best_index]
        best_auc_score = auc_scores[best_index]
        best_model_accuracy = self.candidate_models[best_index].score(X_test, y_test).tolist()
        
        print(f"Best ROC-AUC: {best_auc_score}",)
        print(f"Accuracy from best alpha: {best_model_accuracy}",)
        self.best_alpha = best_alpha
        self.best_auc_score = best_auc_score
    
    def _fit_final_model(self):
        if self.verbose:
            print(f"Refitting model with best alpha {self.best_alpha}")
        self._model = LogitNet(fit_intercept=True, n_lambda=1000, alpha = self.best_alpha, 
                    standardize=False, scoring= 'roc_auc', 
                    n_jobs = self.n_jobs, random_state=0, n_splits = self.n_splits * 2)            
        self._model.fit(self.X, self.y)
        
        self.best_lambda = self._model.lambda_best_[0]
        self.coef_ = self._model.coef_[0]
        
        if self.verbose:
            print("Best alpha value: ", self.best_alpha)
            print("Best lambda value: ", self.best_lambda)
            print("Intercept value: ", self._model.intercept_)
    
    def bootstrap_fit(self, X, y):
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
                standardize=False, scoring= 'roc_auc', n_splits = self.n_splits,
                n_jobs = self.n_jobs, random_state=0)
        bootstrap_model.fit(X_boot, y_boot)
        return bootstrap_model.coef_[0]
    
    def _bootstrap(self):
        if self.verbose:
            print("Boostrapping...")
        
        bootstrap_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.bootstrap_fit)(self.X, self.y) for _ in range(self.n_bootstrap_iterations)
        )
        
        self.bootstrap_coefs = np.array(bootstrap_results)
        self.se_ = np.std(self.bootstrap_coefs, axis=0, ddof=1)
        self.ci_low_ = np.percentile(self.bootstrap_coefs, (1 - self.ci_level) / 2 * 100, axis=0)
        self.ci_high_ = np.percentile(self.bootstrap_coefs, (1 + self.ci_level) / 2 * 100, axis=0)
    
    def _set_feature_names(self):
        self.feature_names_in_ = list(self.X.columns) if isinstance(self.X, pd.DataFrame) else np.arange(self.X.shape[0])    
    
    def predict(self, X):
        return self._model.predict(X)
    
    def predict_proba(self, X):
        return self._model.decision_function(X)
    
    def score(self, X, y):
        return self._model.score(X, y).tolist()
    
    def visualize_alphas(self, save_dir = None):
        
        candidate_alphas = [candidate_model.alpha for candidate_model in tqdm(self.candidate_models)]
        candidate_coefs = [candidate_model.coef_[0] for candidate_model in tqdm(self.candidate_models)]
            
        ax = plt.gca()

        ax.plot(candidate_alphas, candidate_coefs)
        ax.set_xscale("linear")
        plt.xlabel("alpha")
        plt.ylabel("weights")
        plt.title("ElasticNet classifier coefficients as a function of the regularization")
        plt.axis("tight")
        if save_dir is not None:
            if os.path.exists(save_dir):
                plt.savefig(os.path.join(save_dir, 'alphas_vs_weights.png'))
            else:
                print(f"Could not save alphas_vs_weights.png because {save_dir} is not a valid path")
        plt.show()
    
    def summary(self):
        
        try:
            feature_names = self.feature_names_in_
        except AttributeError as e:
            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(e, exc_type, fname, exc_tb.tb_lineno)
            
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

    def _offset_image(self, x, y, img_path, bar_is_too_short, ax, zoom=0.06, vertical=False):
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
    
    def plot_results(self, img_filepaths, summary_df = None, save_dir = None):
        if self.summary_df is None and summary_df is None:
            raise ValueError("No summary DataFrame available. Use summary() method to create one.")
        
        if summary_df is None and self.summary_df is not None: 
            summary_df = self.summary_df.copy()
            
        if save_dir is not None and not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except OSError as e:
                raise ValueError(f"Save directory does not exist: {save_dir}")

        if len(img_filepaths) != summary_df.shape[0]:
            raise ValueError("Length of img_filepaths does not match the number of ElasticNet features (rows) in the summary DataFrame.")
        
        plt.figure(figsize=(15,40)) 

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
            self._offset_image(x = value, 
                            img_path = img_abs_filepath, 
                            y = label, 
                            bar_is_too_short=value < max_value / 10, 
                            zoom=zoom,
                            ax=ax,)
        plt.xlabel = "ElasticNet Classification Coefficients"
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
            self._offset_image(y = value, img_path = img_abs_filepath, 
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

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir,
                        f"vbar.png"))
        plt.show()
        plt.clf()
        
        
        


@dataclass
class ElasticNetClassifierWithStats:
    
    n_Cs: int = 10
    n_l1_ratios: int = 50
    l1_ratios: np.ndarray= field(init=False, default=None)
    n_bootstrap_iterations: int = 250
    ci_level: float = 0.95
    n_jobs: int = -1
    max_iter: int = 10000
    intercept: float = field(init=False, default=None)
    best_l1_ratio: float = field(init=False)
    best_C: float = field(init=False)
    best_auc_score: float = field(init=False)
    ci_high_: float = field(init=False)
    ci_low_: float = field(init=False)
    bootstrap_coefs: np.ndarray = field(init=False)
    se_: float = field(init=False)
    _model: Callable = field(init=False)
    coef_: np.ndarray = field(init=False)
    feature_names_in_: Union[list, tuple, np.ndarray] = field(init=False)
    verbose: bool = True
    summary_df: pd.DataFrame = field(init=False)
    X: Union[pd.DataFrame, np.ndarray] = field(init=False)
    y: Union[pd.DataFrame, np.ndarray] = field(init=False)
    _y_indices_0: np.ndarray = field(init=False)
    _y_indices_1: np.ndarray = field(init=False)
    random_state: int = 42
    
    def __post_init__(self):
        if self.l1_ratios is None:
            self.l1_ratios = np.linspace(0.0, 1.0, num = self.n_l1_ratios)
        
    def fit(self, X, y, n_splits: int = 10):
        
        if X is None or not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError(f"X must be pd.DataFrame or np.ndarray but is {type(X)}")
        
        if y is None or not isinstance(y, (np.ndarray, pd.DataFrame)):
            raise TypeError(f"y must be pd.DataFrame or np.ndarray but is {type(y)}")
        
        if n_splits is None or not isinstance(n_splits, int):
            raise TypeError(f"n_splits must be type int but is {type(n_splits)}")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must be of same length but X is {len(X)} and y is of length {len(y)}")
        
        self.n_splits = n_splits
        _, counts = np.unique(y, return_counts=True)
        self.n_splits = min(self.n_splits, np.min(counts))
        if self.n_splits < 10: 
            self.n_splits -= 1
        self.X, self.y = X, y
        self._model = LogisticRegressionCV(Cs=self.n_Cs,
                                           fit_intercept=True,
                                           scoring="roc_auc",
                                           max_iter=self.max_iter,
                                           cv=self.n_splits, 
                                           n_jobs=self.n_jobs,
                                           refit=True,
                                           random_state=self.random_state,
                                           penalty='elasticnet', 
                                           l1_ratios=self.l1_ratios, 
                                           solver='saga')
        self._model.fit(X, y)
        
        self._y_indices_0 = np.where(y == 0)[0]
        self._y_indices_1 = np.where(y == 1)[0]
        
        self._set_feature_names()
        try: 
            self.best_auc_score = np.max(self._model.scores_.get(1))
        except Exception as e:
            print(e)
            self.best_auc_score = None
            
        self.best_C = self._model.C_[0]
        self.best_l1_ratio = self._model.l1_ratio_[0]
        self.intercept = self._model.intercept_[0]
        self.coef_ = self._model.coef_[0]
        if self.verbose:
            print("Best ROC-AUC score: ", self.best_auc_score)
            print("Best C value: ", self.best_C)
            print("Best L1 ratio: ", self.best_l1_ratio)
            print("Intercept value: ", self.intercept)
                
        self._bootstrap()
        return self

    
    def bootstrap_fit(self, X, y):
        indices = np.random.choice(len(X), len(X), replace=True)
        if isinstance(X, pd.DataFrame):
            X_boot = X.iloc[indices]
        else:
            X_boot = X[indices]
        if isinstance(y, pd.DataFrame):
            y_boot = y.iloc[indices]
        else:
            y_boot = y[indices]
            
        values, counts = np.unique(y_boot, return_counts=True)
            
        if np.min(counts) < self.n_splits:
            majority_indx = np.argmax(counts)
            majority_class = values[majority_indx]
            num_indices_needed = self.n_splits
            if majority_class == 1:
                new_indx = np.random.choice(self._y_indices_0, num_indices_needed)
            else:
                new_indx = np.random.choice(self._y_indices_1, num_indices_needed)
                
            indices = np.concatenate((indices[:len(X) - num_indices_needed], new_indx), axis=None)
            
            if isinstance(X, pd.DataFrame):
                X_boot = X.iloc[indices]
            else:
                X_boot = X[indices]
            if isinstance(y, pd.DataFrame):
                y_boot = y.iloc[indices]
            else:
                y_boot = y[indices]
        
        bootstrap_model = LogisticRegressionCV(Cs=[self.best_C],
                                           fit_intercept=True,
                                           scoring="roc_auc",
                                           max_iter=self.max_iter,
                                           cv=self.n_splits, 
                                           n_jobs=self.n_jobs,
                                           refit=True,
                                           random_state=self.random_state,
                                           penalty='elasticnet', 
                                           l1_ratios=[self.best_l1_ratio], 
                                           solver='saga')
        try:
            bootstrap_model.fit(X_boot, y_boot)
        except ValueError as e:
            print(e)
            sys.exit(1)
        return bootstrap_model.coef_[0]
    
    def _bootstrap(self):
        if self.verbose:
            print("Boostrapping...")
                
        # bootstrap_results = Parallel(n_jobs=self.n_jobs)(
        #     delayed(self.bootstrap_fit)(self.X, self.y) for _ in range(self.n_bootstrap_iterations)
        # )
        
        bootstrap_results = [self.bootstrap_fit(self.X, self.y) for _ in range(self.n_bootstrap_iterations)]
        
        self.bootstrap_coefs = np.array(bootstrap_results)
        self.se_ = np.std(self.bootstrap_coefs, axis=0, ddof=1)
        self.ci_low_ = np.percentile(self.bootstrap_coefs, (1 - self.ci_level) / 2 * 100, axis=0)
        self.ci_high_ = np.percentile(self.bootstrap_coefs, (1 + self.ci_level) / 2 * 100, axis=0)
    
    def _set_feature_names(self):
        self.feature_names_in_ = list(self.X.columns) if isinstance(self.X, pd.DataFrame) else np.arange(self.X.shape[0])    
    
    def predict(self, X):
        return self._model.predict(X)
    
    def predict_proba(self, X):
        return self._model.decision_function(X)
    
    def score(self, X, y):
        return self._model.score(X, y)
    
    def summary(self):
        
        try:
            feature_names = self.feature_names_in_
        except AttributeError as e:
            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(e, exc_type, fname, exc_tb.tb_lineno)
            
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

    def _offset_image(self, x, y, img_path, bar_is_too_short, ax, zoom=0.06, vertical=False):
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
    
    def plot_results(self, img_filepaths, summary_df = None, save_dir = None):
        if self.summary_df is None and summary_df is None:
            raise ValueError("No summary DataFrame available. Use summary() method to create one.")
        
        if summary_df is None and self.summary_df is not None: 
            summary_df = self.summary_df.copy()
            
        if save_dir is not None and not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except OSError as e:
                raise ValueError(f"Save directory does not exist: {save_dir}")

        if len(img_filepaths) != summary_df.shape[0]:
            raise ValueError("Length of img_filepaths does not match the number of ElasticNet features (rows) in the summary DataFrame.")
        
        plt.figure(figsize=(15,40)) 

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
            self._offset_image(x = value, 
                            img_path = img_abs_filepath, 
                            y = label, 
                            bar_is_too_short=value < max_value / 10, 
                            zoom=zoom,
                            ax=ax,)
        plt.xlabel = "ElasticNet Classification Coefficients"
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
            self._offset_image(y = value, img_path = img_abs_filepath, 
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

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir,
                        f"vbar.png"))
        plt.show()
        plt.clf()
        
        
        
import numpy as np
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from typing import Callable, Union

@dataclass
class RidgeClassifierWithStats:
    
    n_alphas: int 
    n_bootstrap: int = 250
    ci_level: float = 0.95
    n_jobs: int = -1
    model: Callable = field(init=False, default=None)
    alphas: np.ndarray = field(init= False, default= None)
    _y_indices_0: np.ndarray = field(init=False, default=None)
    _y_indices_1: np.ndarray = field(init=False, default=None)
    feature_names_in_: Union[list, tuple, np.ndarray] = field(init=False)
    summary_df: pd.DataFrame = field(init=False, default=None)
    n_splits: int = field(init=False, default=10)
    
    def __post_init__(self):
        self.alphas = np.linspace(0.01, 1.0, num = self.n_alphas)
        
    def custom_fit(self, X: pd.DataFrame, y, n_splits = 10):
                
        self.n_splits = n_splits
        _, counts = np.unique(y, return_counts=True)
        self.n_splits = min(self.n_splits, np.min(counts))
        if self.n_splits < 10: 
            self.n_splits -= 1
        self.X = X
        self.y = y
        self._y_indices_0 = np.where(y == 0)[0]
        self._y_indices_1 = np.where(y == 1)[0]
        self.model = RidgeClassifierCV(alphas = self.alphas, 
                                       fit_intercept=True,
                                       cv = self.n_splits, 
                                       scoring = "roc_auc", 
                                       )
        self.model.fit(X, y)
        self.coef_ = self.model.coef_[0]
        self.alpha = self.model.alpha_
        self._set_feature_names()
        print(f"Best ROC-AUC: {self.model.best_score_} for best alpha: {self.model.alpha_}")
        print("Mean accuracy on training data: ", self.custom_score(X, y))
        print("Intercept value: ", self.model.intercept_[0])
        self._bootstrap()
        return self
    
    def bootstrap_fit(self, X: pd.DataFrame, y, n_splits = 10) -> np.ndarray:
        indices = np.random.choice(len(X), len(X), replace=True)

        if isinstance(y, pd.DataFrame):
            y_boot = y.iloc[indices]
        else:
            y_boot = y[indices]
            
        values, counts = np.unique(y_boot, return_counts=True)
        
        if np.min(counts) < n_splits:
            majority_indx = np.argmax(counts)
            majority_class = values[majority_indx]
            num_indices_needed = n_splits
            if majority_class == 1:
                new_indx = np.random.choice(self._y_indices_0, num_indices_needed)
            else:
                new_indx = np.random.choice(self._y_indices_1, num_indices_needed)
                
            indices = np.concatenate((indices[:len(X) - num_indices_needed], new_indx), axis=None)  
            
            if isinstance(y, pd.DataFrame):
                y_boot = y.iloc[indices]
            else:
                y_boot = y[indices] 
        
        if isinstance(X, pd.DataFrame):
            X_boot = X.iloc[indices]
        else:
            X_boot = X[indices]
            
        model = RidgeClassifierCV(alphas=[self.alpha], scoring = "roc_auc", cv = min(n_splits, len(y)))
        model.fit(X_boot, y_boot)
        return model.coef_[0]
    
    def _bootstrap(self):

        
        bootstrap_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.bootstrap_fit)(self.X, self.y) for _ in range(self.n_bootstrap)
        )
        
        self.bootstrap_coefs = np.array(bootstrap_results)
        self.se_ = np.std(self.bootstrap_coefs, axis=0, ddof=1)
        self.ci_low_ = np.percentile(self.bootstrap_coefs, (1 - self.ci_level) / 2 * 100, axis=0)
        self.ci_high_ = np.percentile(self.bootstrap_coefs, (1 + self.ci_level) / 2 * 100, axis=0)
    
    def _set_feature_names(self):
        self.feature_names_in_ = list(self.X.columns) if isinstance(self.X, pd.DataFrame) else np.arange(self.X.shape[0])     
    
    def custom_predict(self, X: pd.DataFrame):
        return self.model.predict(X)
    
    def custom_predict_proba(self, X: pd.DataFrame):
        return self.model.decision_function(X)
    
    def custom_score(self, X: pd.DataFrame, y):
        return self.model.score(X, y)
    
    def visualize_alphas(self, X: pd.DataFrame, y):
        coefs = []
        for a in tqdm(self.alphas):
            ridge = RidgeClassifier(alpha=a)
            ridge.fit(X, y)
            coefs.append(ridge.coef_[0])
            
        ax = plt.gca()

        ax.plot(self.alphas, coefs)
        ax.set_xscale("linear")
        plt.xlabel("alpha")
        plt.ylabel("weights")
        plt.title("Ridge coefficients as a function of the regularization")
        plt.axis("tight")
        plt.show()
        
        ax = plt.gca()

        ax.plot(self.alphas, coefs)
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
        plt.xlabel("alpha")
        plt.ylabel("weights")
        plt.title("Ridge coefficients as a function of the regularization")
        plt.axis("tight")
        plt.show()
    
    def summary(self):   
            
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
        # summary_df['pval'] = p_values
        summary_df.index = self.feature_names_in_
        self.summary_df = summary_df
        # summary_df = summary_df.T
        
        # summary_data = {
        #     "Coefficients": feature_names,
        #     "Estimates": self.coef_,
        #     "Std.Error": self.se_,
        #     "CI.low": self.ci_low_,
        #     "CI.up": self.ci_high_,
        #     "p-value": p_values,
        # }
        
        return summary_df.copy()
    
    def _offset_image(self, x, y, img_path, bar_is_too_short, ax, zoom=0.06, vertical=False):
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
    
    def plot_results(self, img_filepaths, summary_df = None, save_dir = None):
        if self.summary_df is None and summary_df is None:
            raise ValueError("No summary DataFrame available. Use summary() method to create one.")
        
        if summary_df is None and self.summary_df is not None: 
            summary_df = self.summary_df.copy()
            
        if save_dir is not None and not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except OSError as e:
                raise ValueError(f"Save directory does not exist: {save_dir}")

        if len(img_filepaths) != summary_df.shape[0]:
            raise ValueError("Length of img_filepaths does not match the number of ElasticNet features (rows) in the summary DataFrame.")
        
        plt.figure(figsize=(15,40)) 

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
            self._offset_image(x = value, 
                            img_path = img_abs_filepath, 
                            y = label, 
                            bar_is_too_short=value < max_value / 10, 
                            zoom=zoom,
                            ax=ax,)
        plt.xlabel = "Ridge Classification Coefficients"
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
            self._offset_image(y = value, img_path = img_abs_filepath, 
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
        ax.set_ylabel("Ridge Classification Coefficients", fontsize=16,)  # Adjust the title and font size as needed

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir,
                        f"vbar.png"))
        plt.show()
        plt.clf()