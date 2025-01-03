from __future__ import print_function, division
import math
import os
import pdb
import pickle
from torch.utils.data import DataLoader
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm


class CustomDataset_path(Dataset):
    def __init__(self, csv_file, wsi_dir):
        """
        Args:
            csv_file (string): CSV文件的路径，包含案例信息。
            data_dir (string): 包含WSI特征的文件夹的根目录。
        """
        self.data_frame = pd.read_csv(csv_file)

        self.wsi_dir = wsi_dir

        # 尝试加载已保存的特征
        self.wsi_features = self.load_WSI_features()

    def load_WSI_features(self):
        """
        预先加载所有WSI特征。
        """
        features_dict = {}
        for idx in tqdm(range(len(self.data_frame)), desc="Preloading WSI features"):
            case_id = str(self.data_frame.iloc[idx]['case_id'])
            wsi_marker = int(self.data_frame.iloc[idx]['WSI_marker'])
            if wsi_marker == 1:  # 仅当WSI_marker为1时加载特征
                matched_slide_path_list = [os.path.join(self.wsi_dir, d) for d in os.listdir(self.wsi_dir) if
                                           d.startswith(case_id)]
                path_features = []
                for slide_id_path in matched_slide_path_list:
                    wsi_feat_path = os.path.join(slide_id_path, 'features.pt')
                    if os.path.exists(wsi_feat_path):
                        wsi_bag = torch.load(wsi_feat_path)
                        path_features.append(wsi_bag)
                if path_features:  # 仅当存在特征时才添加
                    path_features = torch.cat(path_features, dim=0)
                    features_dict[case_id] = path_features
        return features_dict

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 从DataFrame中获取相关信息
        case_id, label, event_time, c = self.data_frame.iloc[idx, [self.data_frame.columns.get_loc(col) for col in
                                                                   ['case_id', 'disc_label', 'survival_months',
                                                                    'censorship']]]
        case_id = str(case_id)
        # 直接从预加载的特征中获取WSI特征、MRI特征和临床特征
        data_WSI = self.wsi_features.get(case_id, torch.Tensor())

        # 返回必要信息
        return case_id, data_WSI, label, event_time, c

def create_kfold_dataloaders(dataset, num_splits=5, random_seed=42, batch_size=1):
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_seed)
    dataloaders = []

    # 获取所有的索引
    indices = list(range(len(dataset)))

    for train_idx, val_idx in kf.split(indices):
        # 根据索引划分训练集和验证集
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # 创建DataLoader
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        dataloaders.append((train_loader, val_loader))

    return dataloaders


if __name__ == "__main__":

    dataset = CustomDataset_path(csv_file='E:/cervical_prog_project/data_os_labeled.csv', wsi_dir='E:/cervical_prog_project/data_feat/')

    # 创建五折交叉验证的DataLoader
    kfold_dataloaders = create_kfold_dataloaders(dataset, num_splits=5, random_seed=42, batch_size=1)

    # 迭代每一个折的训练和验证集
    for fold, (train_loader, val_loader) in enumerate(kfold_dataloaders):
        print(f"Fold {fold + 1}")

        # 训练集
        for batch_idx, (case_id, data_WSI, label, event_time, c) in enumerate(
                train_loader):
            # 在这里编写训练代码
            pass

        # 验证集
        for batch_idx, (case_id, data_WSI, label, event_time, c) in enumerate(val_loader):
            # 在这里编写验证代码
            pass