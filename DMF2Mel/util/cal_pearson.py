# PYTORCH version of the vlaai original code.
import torch
import pdb

import torch.nn.functional as F



def pearson_correlation(y_true, y_pred, axis=1):

    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)
    # print(y_true.shape)
    # print(y_pred.shape)
    # Compute the numerator and denominator of the pearson correlation.
    numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean),
        dim=axis,
        keepdim=False)

    std_true = torch.sum((y_true - y_true_mean)**2, dim=axis, keepdim=False)
    std_pred = torch.sum((y_pred - y_pred_mean)**2, dim=axis, keepdim=False)
    denominator = torch.sqrt(std_true * std_pred)
    
    pearsonR = torch.div(numerator, denominator + 1e-6)

    #assert torch.all(torch.lt(pearsonR, 1)) and torch.all(torch.gt(pearsonR, -1)), "Loss contains values outside the range of -1 to 1"
    

    
    # std_true = torch.sqrt(torch.sum((y_true - y_true_mean) ** 2, dim=axis, keepdim=False))
    # std_pred = torch.sqrt(torch.sum((y_pred - y_pred_mean) ** 2, dim=axis, keepdim=False))
    # denominator = torch.clamp(torch.sqrt(std_true * std_pred), min=1e-6)

    # pearsonR = torch.div(numerator, denominator)

    # # 使用clamp确保pearsonR的值在[-1, 1]范围内
    # pearsonR = torch.clamp(pearsonR, min=-1, max=1)

    return pearsonR


def pearson_loss(y_true, y_pred, axis=1):
    return -pearson_correlation(y_true, y_pred, axis=axis)

def pearson_metric(y_true, y_pred, axis=1):
    return pearson_correlation(y_true, y_pred, axis=axis)
    
def l1_loss(y_true, y_pred, axis=1):
    l1_dist = torch.abs(y_true - y_pred)
    l1_loss = torch.mean(l1_dist, axis = axis, keepdim=False)
    return l1_loss

def info_nce_loss(eeg_features, img_features, tau=0.1):
    '''
    输入：
        eeg_features:(B, C, T)
        img_features:(B, C, T)
    '''

    # 获取批次大小
    batch_size = eeg_features.shape[0]

    # 计算正样本的 logits
    pos_logits = torch.exp(F.cosine_similarity(eeg_features, img_features, dim=-1) / tau)
    pos_logits = torch.sum(pos_logits)

    # 计算负样本的 logits
    neg_logits = torch.empty(batch_size, device=eeg_features.device)

    for i in range(batch_size):
        # 计算当前 EEG 片段与所有其他图像片段的相似度
        # 这里我们排除自身的正样本
        neg_similarity = torch.exp(F.cosine_similarity(eeg_features[i], img_features[torch.arange(batch_size) != i],dim=-1) / tau)
        neg_logits[i] = torch.sum(neg_similarity)  # 将所有负样本的相似度相加

    # 计算损失
    loss = -torch.mean(torch.log(pos_logits / (pos_logits + neg_logits)))

    return loss



