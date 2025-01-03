from argparse import Namespace
from collections import OrderedDict
import os
import pickle

import numpy as np
from sksurv.metrics import concordance_index_censored
import sys
import torch
from tqdm import tqdm
from models.model_genomic import SNN
from itertools import chain
from models.model_set_mil import *

from utils.utils import *

from utils.coattn_train_utils import *
from utils.cluster_train_utils import *
import copy

def brier_score(predictions, times, events, times_to_evaluate):
    """
    Calculate Brier score for survival predictions.

    :param predictions: 2D array of shape (n_patients, n_times) with survival probabilities.
    :param times: Array of actual survival times for each patient.
    :param events: Array of event occurrences (1 if event occurred, 0 for censored).
    :param times_to_evaluate: List of times at which to evaluate the Brier score.
    :return: Dictionary of Brier scores for each evaluation time.
    """

    # Example usage
    # Assumptions for the example:
    # - 4 time points for predictions: 0, 1, 2, 3
    # - Predictions are made in the form of survival probabilities at each time point for each patient
    # - Times are the actual survival times of each patient
    # - Events indicate whether the event of interest (e.g., death) occurred (1) or the data is censored (0)

    brier_scores = {}
    for eval_time in times_to_evaluate:
        # Identify patients at risk at eval_time
        at_risk = times >= eval_time

        # Calculate squared differences for patients at risk
        squared_diffs = np.where(at_risk, (events - predictions[:, eval_time]) ** 2, 0)

        # Calculate Brier score for this time point
        brier_score = np.mean(squared_diffs)
        brier_scores[eval_time] = brier_score

    return brier_scores


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name: str = 'checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train(train_loader, val_loader, args: Namespace):
    """
        Train for a single fold
    """

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()
    else:
        writer = None

    print("Training on {} samples".format(len(train_loader)))
    print("Validating on {} samples".format(len(val_loader)))

    # Initialize loss function
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_surv = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_surv = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_surv = CoxSurvLoss()
    else:
        raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None
    print('Done!')

    # Initialize model
    print('\nInit Model...', end=' ')
    wsi_encoder = WSI_Projector(input_dim=512, hidden_dim=512).to(torch.device('cuda'))  # 256
    Multimodal_Surv_Net = MIL_Attention_FC_surv_Net(input_dim=512).to(torch.device('cuda'))  # 256
    fusion_network_advanced = MultiFeatureFusionNetworkAdvanced().to(torch.device('cuda'))  # 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 IHC_Projector 实例并加载权重
    def load_projector(model_path):
        projector = IHC_Projector(input_dim=512, hidden_dim=512).to(torch.device('cuda'))  # 256
        projector.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        projector.eval()
        return projector

    # 定义各个 projector 的路径
    projector_paths = {
        "ER": r"Z:\cervical_prog_project\code\marker_saved_models\marker_ER\fold_5\best_projector_auc_0.7709.pth",
        "PR": r"Z:\cervical_prog_project\code\marker_saved_models\marker_PR\fold_1\best_projector_auc_0.9296.pth",
        "P53": r"Z:\cervical_prog_project\code\marker_saved_models\marker_P53\fold_4\best_projector_auc_0.7925.pth",
        "CK14": r"Z:\cervical_prog_project\code\marker_saved_models\marker_CK14\fold_5\best_projector_auc_0.8526.pth",
        "P16": r"Z:\cervical_prog_project\code\marker_saved_models\marker_P16\fold_5\best_projector_auc_0.7218.pth",
        "Bcl2": r"Z:\cervical_prog_project\code\marker_saved_models\marker_Bcl2\fold_1\best_projector_auc_0.7599.pth",
        "Bax": r"Z:\cervical_prog_project\code\marker_saved_models\marker_Bax\fold_2\best_projector_auc_0.8123.pth",
        "TOPO": r"Z:\cervical_prog_project\code\marker_saved_models\marker_TOPO\fold_4\best_projector_auc_0.8000.pth",
        "GST": r"Z:\cervical_prog_project\code\marker_saved_models\marker_GST\fold_1\best_projector_auc_0.7813.pth",
        "P27": r"Z:\cervical_prog_project\code\marker_saved_models\marker_P27\fold_4\best_projector_auc_0.7881.pth",
        "CyclinD1": r"Z:\cervical_prog_project\code\marker_saved_models\marker_CyclinD1\fold_4\best_projector_auc_0.7658.pth",
        "Ki67": r"Z:\cervical_prog_project\code\marker_saved_models\marker_Ki67\fold_2\best_projector_auc_0.6976.pth",
    }

    # 加载所有 projector 并将它们放入字典中
    ihc_projectors = {name: load_projector(path) for name, path in projector_paths.items()}

    # 要做临床的cox 分析 FIGO stage
    print('Done!')

    # Initialize optimizer
    print('\nInit optimizer ...', end=' ')
    all_parameters = chain(wsi_encoder.parameters(), Multimodal_Surv_Net.parameters())
    optimizer = get_optim(all_parameters, args)
    print('Done!')

    # Setup EarlyStopping and best model tracking
    best_cindex = -np.inf
    best_model_wsi_encoder = None
    best_model_surv_net = None

    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose=True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    for epoch in tqdm(range(args.max_epochs)):
        train_loop_survival(epoch, ihc_projectors, wsi_encoder, fusion_network_advanced, Multimodal_Surv_Net, train_loader, optimizer, args.n_classes, writer, loss_surv, reg_fn, args.lambda_reg, args.gc, args.weight_con)

        val_cindex = validate_survival(epoch, ihc_projectors, wsi_encoder, fusion_network_advanced, Multimodal_Surv_Net, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_surv, reg_fn, args.lambda_reg, args.results_dir)

        # Check for the best model based on validation C-Index
        if val_cindex > best_cindex:
            best_cindex = val_cindex
            best_model_wsi_encoder = copy.deepcopy(wsi_encoder.state_dict())
            best_model_surv_net = copy.deepcopy(Multimodal_Surv_Net.state_dict())
            best_model_fusion_network_advanced = copy.deepcopy(fusion_network_advanced.state_dict())
            print(f'New best model found at epoch {epoch} with C-Index: {best_cindex:.4f}')

    # Load the best model for final evaluation
    if best_model_wsi_encoder is not None:
        wsi_encoder.load_state_dict(best_model_wsi_encoder)
        Multimodal_Surv_Net.load_state_dict(best_model_surv_net)
        fusion_network_advanced.load_state_dict(best_model_fusion_network_advanced)
    else:
        print("No best model found, using the last epoch model.")

    # Save the best model
    torch.save(wsi_encoder.state_dict(), os.path.join(args.results_dir, 'best_model_wsi_encoder_' + args.nick_name + '_CI_'+str(best_cindex)+'.pt'))
    torch.save(Multimodal_Surv_Net.state_dict(), os.path.join(args.results_dir, 'best_model_Multimodal_Surv_Net_' + args.nick_name + '_CI_'+str(best_cindex) + '.pt'))
    torch.save(fusion_network_advanced.state_dict(), os.path.join(args.results_dir,
                                                              'best_model_fusion_network_advanced_' + args.nick_name + '_CI_' + str(
                                                                  best_cindex) + '.pt'))

    # Final evaluation on the best model
    results_val_dict, final_val_cindex = summary_survival(ihc_projectors, wsi_encoder, fusion_network_advanced, Multimodal_Surv_Net, val_loader, args.n_classes)

    print('Final Val C-Index: {:.4f}'.format(final_val_cindex))
    writer.close()
    return results_val_dict, final_val_cindex

def train_loop_survival(epoch, ihc_projectors, encoder_wsi, fusion_network_advanced, SurvNet, loader, optimizer, n_classes, writer=None,
                        loss_surv=None, reg_fn=None, lambda_reg=0., gc=16, weight_con=0.5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_wsi.train().to(device)
    SurvNet.train().to(device)
    fusion_network_advanced.train().to(device)

    train_loss_surv, train_loss_surv_reg, train_loss = 0., 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (case_id, data_WSI, label, event_time, c) in enumerate(
            loader):

        case_id = str(case_id[0])
        data_WSI = data_WSI.to(device).squeeze().float()
        label = label.to(device).long()
        c = c.to(device)

        # 这些 Projector 是已经预训练的，不再参与训练，因此用 torch.no_grad()
        with torch.no_grad():
            # 定义一个空的列表来存储各个 projector 的输出
            features_list = []

            # 逐个将 data_WSI 传入到每个 projector 中
            for name, projector in ihc_projectors.items():
                feature = projector(data_WSI)  # 获得每个 projector 的输出
                features_list.append(feature)  # 将输出添加到列表中

        feat_wsi = encoder_wsi(data_WSI)
        # 使用改进后的融合网络融合所有特征
        fused_features = fusion_network_advanced(features_list, feat_wsi)

        hazards, S, Y_hat, _, _ = SurvNet(x=fused_features)  # return hazards, S, Y_hat, A_raw, results_dict

        loss_sur = loss_surv(hazards=hazards, S=S, Y=label, c=c)

        loss_sur_value = loss_sur.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(SurvNet) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_sur_value
        train_loss_surv_reg += loss_sur_value + loss_reg

        train_loss += train_loss_surv_reg

        # if (batch_idx + 1) % 100 == 0:
        # if (batch_idx + 1) % 1 == 0:
        #     print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx,
        #                                                                                                      loss_value + loss_reg,
        #                                                                                                      label.item(),
        #                                                                                                      float(
        #                                                                                                          event_time),
        #                                                                                                      float(
        #                                                                                                          risk),
        #                                                                                                      data_WSI.size(
        #                                                                                                          0)))
        # backward pass
        loss = loss_sur / gc + loss_reg
        loss.backward(retain_graph=True)

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships)
    c_index = \
        concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                   tied_tol=1e-08)[0]

    print(
        'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
            epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival(epoch, ihc_projectors, encoder_wsi, fusion_network_advanced, SurvNet, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None,
                      loss_surv=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_wsi.eval().to(device)
    SurvNet.eval().to(device)
    fusion_network_advanced.eval().to(device)

    val_loss_surv, val_loss = 0., 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    patient_results = {}
    hazard_scores = []

    for batch_idx, (case_id, data_WSI, label, event_time, c) in enumerate(
            loader):

        case_id = str(case_id[0])
        # print(case_id)

        data_WSI = data_WSI.to(device).squeeze().float()
        c = c.to(device)

        with torch.no_grad():

            # 定义一个空的列表来存储各个 projector 的输出
            features_list = []

            # 逐个将 data_WSI 传入到每个 projector 中
            for name, projector in ihc_projectors.items():
                feature = projector(data_WSI)  # 获得每个 projector 的输出
                features_list.append(feature)  # 将输出添加到列表中

            feat_wsi = encoder_wsi(data_WSI)
        # 使用改进后的融合网络融合所有特征
            fused_features = fusion_network_advanced(features_list, feat_wsi)

            hazards, S, Y_hat, _, _ = SurvNet(x=fused_features)  # return hazards, S, Y_hat, A_raw, results_dict

        hazard_scores.append(hazards.cpu().numpy())

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

    event_happens = (1 - all_censorships).astype(int)
    hazard_scores_array = np.concatenate(hazard_scores, axis=0)
    times_to_evaluate = list(range(n_classes))

    # Calculate Brier scores
    brier_scores = brier_score(hazard_scores_array, all_event_times, event_happens, times_to_evaluate)

    average_brier_score = np.mean(list(brier_scores.values()))

    c_index = \
        concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                   tied_tol=1e-08)[0]

    print('Epoch: {}, val_c_index: {:.4f}, val_brier_score: {:.4f}'.format(epoch, c_index,average_brier_score))

    if writer:
        writer.add_scalar('val/c-index', c_index, epoch)
        writer.add_scalar('val/average_brier_score', average_brier_score, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model,
                       ckpt_name=os.path.join(results_dir))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return c_index


def summary_survival(ihc_projectors, encoder_wsi, fusion_network_advanced, SurvNet, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_wsi.eval().to(device)
    SurvNet.eval().to(device)
    fusion_network_advanced.eval().to(device)

    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    # slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (case_id, data_WSI, label, event_time, c) in enumerate(
            loader):

        case_id = str(case_id[0])
        data_WSI = data_WSI.to(device).squeeze().float()
        c = c.to(device)

        with torch.no_grad():

            # 定义一个空的列表来存储各个 projector 的输出
            features_list = []

            # 逐个将 data_WSI 传入到每个 projector 中
            for name, projector in ihc_projectors.items():
                feature = projector(data_WSI)  # 获得每个 projector 的输出
                features_list.append(feature)  # 将输出添加到列表中

            feat_wsi = encoder_wsi(data_WSI)
        # 使用改进后的融合网络融合所有特征
            fused_features = fusion_network_advanced(features_list, feat_wsi)

            # Clear CUDA memory for features_list and feat_wsi
            print()
            del features_list
            del feat_wsi
            del fusion_network_advanced
            # torch.cuda.empty_cache()  # Free up unreferenced CUDA memory

            hazards, survival, Y_hat, _, _ = SurvNet(x=fused_features)  # return hazards, S, Y_hat, A_raw, results_dict

        risk = -torch.sum(survival, dim=1).cpu().numpy().item()
        event_time = event_time.item()
        c = c.item()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({case_id: {'case_id': np.array(case_id), 'risk': risk,
                                           'survival': event_time, 'censorship': c}})

    c_index = \
        concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                   tied_tol=1e-08)[0]
    return patient_results, c_index
