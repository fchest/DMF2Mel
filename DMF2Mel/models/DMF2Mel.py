import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
import pdb
from models.SubLayers import SplineMapAttention, PositionwiseFeedForward

from torch.nn.utils import weight_norm

from torch.nn import init

import einops
from einops import rearrange
from einops.layers.torch import Rearrange

from mamba_ssm import Mamba

from models.bimamba import Bimamba_outer

from torch.nn import Module, Parameter, Softmax

import math
from itertools import repeat
from torch._jit_internal import Optional
from torch.nn.modules.module import Module
import collections

import functools

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


class DynamicFeatureAdjust(nn.Module):
    def __init__(self, dim, scale=0.1):
        super(DynamicFeatureAdjust, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim) * scale)  # 初始化权重为较小的值
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x * self.weight  # 动态调整权重
        x = x.permute(0, 3, 1, 2)
        return x


class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, stride=1, padding=1, attn_dropout=0.1):
        super(ContrastDrivenFeatureAggregation, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        self.attn_fg = nn.Linear(dim, kernel_size ** 2 * kernel_size ** 2 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 2 * kernel_size ** 2 * num_heads)
        self.v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(attn_dropout)  # 添加dropout防止过拟合

    def forward(self, x, fg, bg):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        v = self.v(x)
        v_unfolded = self.unfold(v.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim, self.kernel_size ** 2, -1).permute(0, 1, 4, 3, 2)

        attn_fg = self.attn_fg(fg).reshape(B, H * W, self.num_heads, self.kernel_size ** 2, self.kernel_size ** 2).permute(0, 2, 1, 3, 4)
        attn_bg = self.attn_bg(bg).reshape(B, H * W, self.num_heads, self.kernel_size ** 2, self.kernel_size ** 2).permute(0, 2, 1, 3, 4)

        attn_fg = self.dropout(attn_fg)  # 应用dropout
        attn_bg = self.dropout(attn_bg)  # 应用dropout

        x_fg = (attn_fg @ v_unfolded).sum(dim=3)
        x_fg = x_fg.permute(0, 1, 3, 2)
        x_bg = (attn_bg @ v_unfolded).sum(dim=3)
        x_bg = x_bg.permute(0, 1, 3, 2)

        new_C = self.num_heads * self.head_dim
        new_H = H
        new_W = W

        x_fg = x_fg.reshape(B, new_C, new_H, new_W)
        x_bg = x_bg.reshape(B, new_C, new_H, new_W)

        x = x_fg + x_bg
        return x

class DC_FAM(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(DC_FAM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True, groups=dim // self.num_heads, padding=1)

        # 动态特征调整模块
        self.dynamic_adjust_local = DynamicFeatureAdjust(dim, scale=0.1)  # 调整缩放因子
        self.dynamic_adjust_global = DynamicFeatureAdjust(dim, scale=0.1)  # 调整缩放因子

        # 对比驱动特征聚合模块
        self.cdfa = ContrastDrivenFeatureAggregation(dim=dim, num_heads=num_heads, attn_dropout=0.1)  # 添加dropout

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unsqueeze(2)

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)

        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], H * W, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2)).squeeze(2)
        f_conv = f_all.permute(0, 3, 1, 2).reshape(B, 9 * C // self.num_heads, H, W)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv).squeeze(2)

        out_conv = self.dynamic_adjust_local(out_conv)

        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        out = out.unsqueeze(2)
        out = self.project_out(out).squeeze(2)
        out = self.dynamic_adjust_global(out)

        fg = out_conv
        bg = out
        enhanced_feature = self.cdfa(out_conv, fg, bg)

        output = enhanced_feature + out_conv
        return output


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class FeedForwardModule1(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(FeedForwardModule1, self).__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthWiseConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


        
class ADAF(nn.Module):
    def __init__(self, in_channels):
        super(ADAF, self).__init__()
        inter_channels = in_channels // 16

        self.conv5a = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())
        self.conv5c = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())
        
        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(inter_channels, in_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(inter_channels, in_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(in_channels, in_channels, 1),
                                   nn.ReLU())



        self.adaptive_weight = nn.Parameter(torch.ones(2)) 
        self.sigmoid = nn.Sigmoid()


        self.feedback_conv = nn.Sequential(nn.Dropout2d(0.05, False), 
                                           DepthWiseConv2d(inter_channels, inter_channels, 1),
                                           nn.ReLU())

    def forward(self, x, feedback=None):
 
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        adaptive_weight = self.sigmoid(self.adaptive_weight)
        sa_output = adaptive_weight[0] * sa_conv
        sc_output = adaptive_weight[1] * sc_conv


        if feedback is not None:
            feedback = self.feedback_conv(feedback)
            sa_output += feedback
            sc_output += feedback

        sa_output1 = self.conv6(sa_output)
        sc_output2 = self.conv7(sc_output)

        feat_sum = sa_output1 + sc_output2

        sasc_output = self.conv8(feat_sum)

        return sasc_output

class HAMSnet(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(HAMSnet, self).__init__()

        self.convd1 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1) #640 - 320

        self.convd2 = nn.Conv1d(64, 64, kernel_size=2, stride=2) #320 - 160

        self.convd3 = nn.Conv1d(64, 64, kernel_size=2, stride=2) #160 - 80

        self.pool0 = nn.MaxPool1d(kernel_size=8, stride=8)

        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)


        self.sigmoid = nn.Sigmoid()

        d_k = 64 // 2
        d_v = 64 // 2



        self.block1 = ADAF(64)


    def forward(self, x):

        enc10 = x

        enc1 = self.convd1(x)  #320
        enc11 = enc1

        
        enc2 = self.convd2(enc1)  #160
        enc12 = enc2

        enc3 = self.convd3(enc2)  #80
        enc13 = enc3

        enc100 = self.pool0(enc10)  #640-80

        enc111 = self.pool1(enc11)  #320-80

        enc122 = self.pool2(enc12)  #180-80

        
        enc = enc100+enc111+enc122+enc13 
        
        
        encc = enc  #enc的size: torch.Size([32, 64, 80])

   

        
        enc = enc.unsqueeze(2)   #enc的size: torch.Size([32, 64,1, 80])


        enc = self.block1(enc)

        enc = self.block1(enc)

        enc = self.block1(enc)

        enc = enc.squeeze(2)

        enc = encc+enc



        dec0 = enc

        res = dec0



        dec0 = dec0.unsqueeze(2)


        dec0= self.block1(dec0)

        dec0 = dec0.squeeze(2)

     
        dec0 = dec0+res


        dec1 = F.interpolate(dec0 , size=160, mode='nearest')

        dec2 = F.interpolate(dec0, size=320, mode='nearest') 

        dec3 = F.interpolate(dec0 , size=640, mode='nearest')



        dec00 = enc13 + dec0

        dec11 = dec1+enc12
        
        dec22 = dec2+enc11
        
        dec33 = dec3+enc10
        

        dec001 = dec00.unsqueeze(2)  
        
        dec112 = dec11.unsqueeze(2) 
        
        dec221 = dec22.unsqueeze(2) 
        
        dec331 = dec33.unsqueeze(2)  


        dec001 = self.block1(dec001)

        dec112 = self.block1(dec112)

        dec221 = self.block1(dec221)

        dec331 = self.block1(dec331) 

        dec001 = dec001.squeeze(2)
        dec112 = dec112.squeeze(2)
        dec221 = dec221.squeeze(2)
        dec331 = dec331.squeeze(2)


        dec00 = dec00 + dec001

        dec11 = dec11 + dec112

        dec22 = dec22 + dec221

        dec33 = dec33 + dec331



        dec000 = F.interpolate(dec00, size=160, mode='nearest')


        dec0000 = self.sigmoid(dec000)



        dec111 = dec11 + dec000

        
        dec112 = F.interpolate(dec111, size=320, mode='nearest')


        dec1122 = self.sigmoid(dec112)


        
        dec222 = dec22 + dec112
        
        dec223 = F.interpolate(dec222, size=640, mode='nearest')


        dec2233 = self.sigmoid(dec223)

        dec333 = dec33 + dec223

        dec333 = dec333+x
        
        return dec333


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(FeedForwardModule, self).__init__()
        # 第一个线性变换层
        self.linear1 = nn.Linear(64, 2048)
        # 第二个线性变换层
        self.linear2 = nn.Linear(2048, 64)
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        # Swish激活函数
        self.swish = nn.SiLU()
        # 第一个Dropout层
        self.dropout1 = nn.Dropout(dropout_rate)
        # 第二个Dropout层
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 通过层归一化
        normalized = self.layer_norm(x)
        # 第一个线性变换
        x = self.linear1(normalized)
        # 应用Swish激活函数
        x = self.swish(x)
        # 第一个Dropout
        x = self.dropout1(x)
        # 第二个线性变换
        x = self.linear2(x)
        # 第二个Dropout
        x = self.dropout2(x)
        return x


class mambaBlock(nn.Module):
    def __init__(self, in_channel, d_model, d_inner, n_head, n_layers, fft_conv1d_kernel, fft_conv1d_padding,
                 dropout, g_con, within_sub_num=71, **kwargs):
        super(mambaBlock, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head

        self.within_sub_num = within_sub_num
        self.g_con = g_con


        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channel, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        self.Bimamba_outer = Bimamba_outer(
            d_model=64,
            d_state=128,
            d_conv=4,
            expand=2
        )



        self.ffm = FeedForwardModule(d_model, d_inner, dropout)

    def forward(self, dec_input, sub_id):

       
        dec_output = self.conv3(dec_input.transpose(1, 2))


        dec_output = dec_output.transpose(1, 2)


        dec_output = self.Bimamba_outer(dec_output)



#第一个ffm模块
        res1 = dec_output
        dec_output = self.ffm(dec_output)
        dec_output = dec_output * 0.5
        dec_output = dec_output+res1

        res2 = dec_output

        dec_output = self.conv3(dec_output.transpose(1, 2))

        dec_output = dec_output.transpose(1, 2)
   
        dec_output = dec_output+res2

#第二个ffm模块
        res1 = dec_output
        dec_output = self.ffm(dec_output)
        dec_output = dec_output * 0.5
        dec_output = dec_output+res1

        # print("After 第二个ffm:", dec_output.shape)


        return dec_output



class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=640):
        super(ScaledPositionalEncoding, self).__init__()
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 使用正余弦函数生成位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将位置编码矩阵注册为一个buffer，这样它就不会在训练中更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状是 [seq_len, batch_size, d_model]
        # 根据x的长度选择相应的位置编码并添加到x上



        y = self.pe[:x.size(1), :].unsqueeze(0)  


        x = x + self.pe[:x.size(1), :].unsqueeze(0)  
        return x



class Decoder(nn.Module):
    def __init__(self, in_channel, d_model, d_inner, n_head, n_layers, fft_conv1d_kernel, fft_conv1d_padding,
                 dropout, g_con, within_sub_num=85, **kwargs):
        super(Decoder, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head
        self.g_con = g_con
        self.within_sub_num = within_sub_num
        # self.slf_attn = MultiHeadAttention


        # self.DOblock = DOConv2d(in_channels=64, out_channels=32, kernel_size=1)

        self.positional_encoding = ScaledPositionalEncoding(d_model, max_len=640)

        self.sub_proj = nn.Linear(self.within_sub_num, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = SplineMapAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel,fft_conv1d_padding, dropout=dropout)



        self.fc = nn.Linear(64, 1)

        self.slf_attn1 = SplineMapAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)

        self.slf_attn2 = SplineMapAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)

        self.slf_attn3 = SplineMapAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)

        self.slf_attn4 = SplineMapAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)

        self.slf_attn5 = SplineMapAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)

        self.slf_attn6 = SplineMapAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)


        self.dcfam = DC_FAM(dim=640, num_heads=8)


        self.hams = HAMSnet(in_channels=640, mid_channels=640, out_channels=640)   #Parameters: 1.003841M;  

        self.mamba_block = mambaBlock(
            in_channel=64,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            n_layers=n_layers,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
            g_con=g_con,
            within_sub_num=within_sub_num)  

        
    def forward(self, dec_input, sub_id):


        qianru = self.positional_encoding(dec_input)

        sub_emb    =  F.one_hot(sub_id, self.within_sub_num)
        sub_emb    =  self.sub_proj(sub_emb.float())
        sub_emb    =  sub_emb.unsqueeze(1)

        sub_emb = self.layer_norm(sub_emb)
        dec_input = dec_input + sub_emb


        fft_output, _= self.slf_attn(
            qianru, dec_input, dec_input)


        fft_output = self.layer_norm(fft_output)

        fft_output = self.pos_ffn(fft_output)+fft_output


        dec_input = dec_input+fft_output

        
        original_dec_input = dec_input


        dec_input1 = dec_input


        dec_input0 = dec_input.permute(0, 2, 1)


        dec_input1 = dec_input1.permute(0, 2, 1)

        for _ in range(5):
            dec_input1 = self.hams(dec_input1)
            dec_input1 = dec_input1 + dec_input0

        dec_input1 = self.hams(dec_input1)


        dec_input1 = dec_input1.permute(0, 2, 1)



                
        dec_input2 = dec_input

         
         
        dec_input2 = dec_input2.unsqueeze(2)

        dec_input = dec_input.unsqueeze(2)

        for _ in range(4):
            dec_input2 = self.dcfam(dec_input2)
            dec_input2 = dec_input2 + dec_input
        
        
        dec_input2 = dec_input2.squeeze(2)

        dec_input = dec_input.squeeze(2)


        dec_input1, _= self.slf_attn1(dec_input1, dec_input1, dec_input1)


        dec_input2, _= self.slf_attn1(dec_input2, dec_input2, dec_input2)



        dec_input3, _= self.slf_attn1(dec_input2, dec_input1, dec_input1)

        dec_input3, _= self.slf_attn2(dec_input3, dec_input1, dec_input1)

        dec_input3, _= self.slf_attn3(dec_input2, dec_input3, dec_input3)

        dec_input3, _= self.slf_attn4(dec_input3, dec_input1, dec_input1)

        dec_input3, _= self.slf_attn5(dec_input2, dec_input3, dec_input3)

        dec_input3, _= self.slf_attn6(dec_input3, dec_input2, dec_input2)




        dec_input4= self.mamba_block(dec_input3, sub_id) 

        dec_input4 = dec_input4 +dec_input3

        dec_input4 = self.mamba_block(dec_input4, sub_id) 




        output = self.fc(dec_input4)

        return output