import os
import pandas as pd
import ast
import glob
import shutil
import datetime

class OutputOrganizer:
    def __init__(self, crop_image_dir, test_data_id, metadata_name, test_data_list_name, result_dirs, save_dir):
        self.crop_image_dir = crop_image_dir
        self.test_data_id = test_data_id
        self.metadata_name = metadata_name
        self.test_data_list_name = test_data_list_name
        self.result_dirs = result_dirs
        self.save_dir = save_dir
        self.test_data_list = pd.read_csv(test_data_list_name)
        self.meta_data = pd.read_csv(metadata_name, sep=",")
        self.selected_mi_ids = self.get_selected_mi_ids()
        self.result_dict = self.initialize_result_dict()
        self.current_timestamp = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        self.save_dir = os.path.join(save_dir, f"organized-output-{self.current_timestamp}")
        self.create_directory(self.save_dir)

    def get_selected_mi_ids(self):
        ground_truth_pos_mi_ids = [mi_id for mi_id in self.test_data_list['MI_ID'] 
                                   if self.meta_data[self.meta_data['MI_ID'] == mi_id]['liver_fatty'].to_list()[0] > 0]
        selected_mi_ids = [mi_id for mi_id in ground_truth_pos_mi_ids
                           if len(ast.literal_eval(self.meta_data[self.meta_data['MI_ID'] == mi_id]['IMG_ID_LIST'].to_list()[0])) >= 20]
        return selected_mi_ids

    def initialize_result_dict(self):
        return {mi_id: {"csv_paths": [], "plot_paths": []} for mi_id in self.selected_mi_ids}

    def create_directory(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            print(f"{path} already exists, files may be overwritten")

    def process_results(self):
        for result_dir in self.result_dirs:
            completed_subj = [f.name for f in os.scandir(result_dir) if f.is_dir()]
            for mi_id in completed_subj:
                glob_path = os.path.join(result_dir, mi_id)
                if glob_path[-1] != '/':
                    glob_path = glob_path + "/"
                mi_id_csvs = glob.glob(glob_path + "*.csv")
                if len(mi_id_csvs) != 0:
                    self.result_dict[mi_id]["csv_paths"] += mi_id_csvs
                mi_id_img = glob.glob(glob_path + "*.png")
                if len(mi_id_img) != 0:
                    self.result_dict[mi_id]["plot_paths"] += mi_id_img

    def save_results(self):
        for mi_id, v in self.result_dict.items():
            subj_dir = os.path.join(self.save_dir, mi_id)
            self.create_directory(subj_dir)
            v["dfs"] = [pd.read_csv(csv_path) for csv_path in v["csv_paths"]]
            img_id_list = ast.literal_eval(self.meta_data[self.meta_data['MI_ID'] == mi_id]['IMG_ID_LIST'].to_list()[0])
            csv_dest_dir = os.path.join(subj_dir, "csv")
            self.create_directory(csv_dest_dir)
            for indx, df in enumerate(v["dfs"]):
                type_dir = os.path.join(subj_dir, v["csv_paths"][indx].split("/")[-1].split(".csv")[0].split("_")[0])
                self.create_directory(type_dir)
                coef_col, significant_df, insignificant_df, positive_df, negative_df = self.process_dataframe(df, img_id_list)
                if coef_col is None:
                    continue
                df.to_csv(os.path.join(csv_dest_dir, f"{indx}-" + v["csv_paths"][indx].split("/")[-1]), index=None)
                self.save_images(mi_id, positive_df, negative_df, insignificant_df, coef_col, type_dir)
            self.save_plots(subj_dir, v["plot_paths"])

    def process_dataframe(self, df, img_id_list):
        if "Estimate" in df.columns and "SESignificance" in df.columns:
            coef_col = "Estimate"
            if "IMG" not in df.columns:
                df["IMG"] = img_id_list
            significant_df = df.loc[df['SESignificance'] == "SIGNIFICANT"].copy()
            insignificant_df = df.loc[df['SESignificance'] == "INSIGNIFICANT"].copy()
            positive_df = significant_df.loc[significant_df["Estimate"] > 0.0].copy()
            negative_df = significant_df.loc[significant_df["Estimate"] < 0.0].copy()
        elif "corrs" in df.columns and "corr_significance" in df.columns:
            coef_col = "corrs"
            significant_df = df.loc[df['corr_significance'] == "SIGNIFICANT"].copy()
            insignificant_df = df.loc[df['corr_significance'] == "INSIGNIFICANT"].copy()
            positive_df = significant_df.loc[significant_df["corrs"] > 0.0].copy()
            negative_df = significant_df.loc[significant_df["corrs"] < 0.0].copy()
        elif "corr" in df.columns and "corr_p_val" in df.columns:
            coef_col = "corr"
            if "IMG" not in df.columns:
                df["IMG"] = img_id_list
            significant_df = df.loc[df['corr_p_val'] < 0.05].copy()
            insignificant_df = df.loc[df['corr_p_val'] >= 0.05].copy()
            positive_df = significant_df.loc[significant_df["corr"] > 0.0].copy()
            negative_df = significant_df.loc[significant_df["corr"] < 0.0].copy()
        else:
            return None, None, None, None, None
        return coef_col, significant_df, insignificant_df, positive_df, negative_df

    def save_images(self, mi_id, positive_df, negative_df, insignificant_df, coef_col, type_dir):
        positive_df = positive_df.sort_values([coef_col], ascending=[False])
        negative_df = negative_df.sort_values([coef_col], ascending=[True])
        positive_ranked_img = positive_df["IMG"]
        negative_ranked_img = negative_df["IMG"]
        positive_output_img = positive_df["IMG"].astype(str) + "_" + positive_df[coef_col].round(2).astype(str)
        negative_output_img = negative_df["IMG"].astype(str) + "_" + negative_df[coef_col].round(2).astype(str)
        positive_img_paths = [os.path.join(self.crop_image_dir, f"{mi_id}_{img_id}.jpg") for img_id in positive_ranked_img]
        negative_img_paths = [os.path.join(self.crop_image_dir, f"{mi_id}_{img_id}.jpg") for img_id in negative_ranked_img]
        neutral_img_paths = [os.path.join(self.crop_image_dir, f"{mi_id}_{img_id}.jpg") for img_id in insignificant_df["IMG"]]
        pos_img_dir = os.path.join(type_dir, "positive/")
        self.create_directory(pos_img_dir)
        neg_img_dir = os.path.join(type_dir, "negative/")
        self.create_directory(neg_img_dir)
        neutral_img_dir = os.path.join(type_dir, "neutral/")
        self.create_directory(neutral_img_dir)
        output_positive_img_paths = [pos_img_dir + f"{indx+1}-{i}.png" for indx, i in enumerate(positive_output_img)]
        output_neutral_img_paths = [neutral_img_dir + i.split('/')[-1] for _, i in enumerate(neutral_img_paths)]
        output_negative_img_paths = [neg_img_dir + f"{indx+1}-{i}.png" for indx, i in enumerate(negative_output_img)]
        for src_img, dest_img in zip(positive_img_paths, output_positive_img_paths):
            shutil.copy2(src_img, dest_img)
        for src_img, dest_img in zip(neutral_img_paths, output_neutral_img_paths):
            shutil.copy2(src_img, dest_img)
        for src_img, dest_img in zip(negative_img_paths, output_negative_img_paths):
            shutil.copy2(src_img, dest_img)

    def save_plots(self, subj_dir, plot_paths):
        plot_save_dir = os.path.join(subj_dir, "plots")
        self.create_directory(plot_save_dir)
        img_save_paths = [os.path.join(plot_save_dir, f"{indx}-" + filepath.split("/")[-1])
                          for indx, filepath in enumerate(plot_paths)]
        for src_img, dest_img in zip(plot_paths, img_save_paths):
            shutil.copy2(src_img, dest_img)

    def run(self):
        self.process_results()
        self.save_results()
        print(f"All results saved to {self.save_dir}")

if __name__ == "__main__":
    CROP_IMAGE_DIR = os.getenv('CROP_IMAGE_DIR_PATH')
    test_data_id = '09'
    metadata_name = 'meta_data/TWB_ABD_expand_modified_gasex_21072022.csv'
    test_data_list_name = f'fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset{test_data_id}/test_dataset{test_data_id}.csv'
    result_dirs = [
        "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/elastic-net-old-dataset-08-01-2024-06-59-39",
        "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/ridge-08-06-2024-01-24-40",
        "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/correlation-old-dataset-08-01-2024-03-37-02"
    ]
    save_dir = "/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results/organized_output"
    organizer = OutputOrganizer(CROP_IMAGE_DIR, test_data_id, metadata_name, test_data_list_name, result_dirs, save_dir)
    organizer.run()
