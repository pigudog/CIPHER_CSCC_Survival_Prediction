from __future__ import print_function

import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from datasets.dataset_path_os_single_241004 import *
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code
from utils.core_utils import train

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
import sys
import os

### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')


parser.add_argument('--results_dir', type=str, default='./results/', help='Results directory (Default: ./results)')
parser.add_argument('--exp_name', type=str, default='cervical_os_single', help='')

parser.add_argument('--path_input_dim', type=int, default=512, help='path_input_dim')

### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--n_classes', type=int, default=4, help='n_classes')

parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible experiment (default: 1)')

parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')

### Model Parameters.
parser.add_argument('--model_type', type=str, choices=['snn', 'deepset', 'amil', 'mi_fcn', 'mcat'], default='amil',
                    help='Type of model (Default: mcat)')
parser.add_argument('--mode', type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'], default='path',
                    help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion', type=str, choices=['None', 'concat', 'bilinear'], default='None',
                    help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig', action='store_true', default=False,
                    help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats', action='store_true', default=False,
                    help='Use genomic features as tabular features.')
parser.add_argument('--drop_out', action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi', type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc', type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs', type=int, default=80, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--weight_con', type=float, default=0.5, help='Learning rate (default: 0.0001)')

parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'],
                    default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None',
                    help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)

print("Experiment Name:", args.exp_code)

from datetime import datetime

nick_name = str(datetime.now().strftime("%Y%m%d%H%M%S"))


settings = {'batch_size': args.batch_size,
            'num_splits': args.k,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt}

print('\nLoad Dataset')

dataset = CustomDataset_path(csv_file='E:/cervical_prog_project/data_os_labeled.csv', wsi_dir='E:/cervical_prog_project/data_feat/')

# 创建五折交叉验证的DataLoader
kfold_dataloaders = create_kfold_dataloaders(dataset, num_splits=settings['num_splits'], random_seed=settings['seed'],
                                             batch_size=settings['batch_size'])

for fold, (train_loader, val_loader) in enumerate(kfold_dataloaders):
    print(f"Fold {fold + 1}")

    args.nick_name = nick_name + f"_Fold_{fold + 1}_"


    # 执行训练过程
    val_latest, cindex_latest = train(train_loader, val_loader, args)

    # 保存验证结果，文件名中包含当前的fold
    results_pkl_path = os.path.join(args.results_dir, f"Fold_{fold + 1}_latest_val_results_{args.nick_name}.pkl")
    save_pkl(results_pkl_path, val_latest)

    # 保存当前fold的结果到CSV文件中，文件名中包含当前的fold
    results_latest_df = pd.DataFrame({'folds': [fold + 1], 'val_cindex': [cindex_latest]})
    results_latest_df.to_csv(os.path.join(args.results_dir, f'Fold_{fold + 1}_summary_latest_{args.nick_name}.csv'))

print("Done!")

