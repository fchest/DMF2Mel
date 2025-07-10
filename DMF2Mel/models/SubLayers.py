import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb



# from models.kankimi import MemoryAugmentedKAN

from models.AGKAN import AGKANLinear

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class SplineMapAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, S=64, dropout=0.1):
        """
        参数说明：
        n_head: 注意力头数
        d_model: 模型维度
        d_k: 键/查询维度
        d_v: 值维度
        S: 注意力压缩维度
        """
        super().__init__()
        
        # 基础参数配置
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        
        # 使用AGKAN替代传统线性变换
        self.w_qs = AGKANLinear(d_model, n_head*d_k, S=S, 
                              grid_size=8, spline_order=3)
        self.w_ks = AGKANLinear(d_model, n_head*d_k, S=S//2,  # Key使用更小压缩比
                              grid_size=6, spline_order=3)
        self.w_vs = AGKANLinear(d_model, n_head*d_v, S=S,
                              grid_size=8, spline_order=3)
        
        # 注意力计算模块
        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
        
        # 输出融合层
        self.fc = AGKANLinear(n_head*d_v, d_model, S=2*S,  # 增大压缩维度
                            grid_size=6, spline_order=3)
        
        # 标准化与正则化
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 自定义初始化
        self._init_agkan_weights()

    def _init_agkan_weights(self):
        """AGKAN专用权重初始化策略"""
        # 基函数路径初始化
        for m in [self.w_qs, self.w_ks, self.w_vs]:
            nn.init.normal_(m.kan.base_weight, mean=0, std=0.02)
            nn.init.normal_(m.kan.spline_weight, mean=0, std=0.01)
        
        # 注意力路径初始化
        nn.init.orthogonal_(self.w_qs.attn_compress.weight)
        nn.init.kaiming_uniform_(self.w_ks.attn_compress.weight)
        nn.init.xavier_normal_(self.w_vs.attn_recover.weight)
        
        # 门控网络偏置初始化
        nn.init.constant_(self.w_qs.gate[0].bias, 1.0)  # Query偏重KAN路径
        nn.init.constant_(self.w_ks.gate[0].bias, 0.5)  # Key平衡模式
        nn.init.constant_(self.w_vs.gate[0].bias, 2.0)  # Value强偏KAN

    def forward(self, q, k, v, mask=None):
        """
        前向传播流程优化：
        1. 改进维度转换效率
        2. 添加记忆监控接口
        3. 优化残差连接位置
        """
        residual = q  # 保留原始输入用于残差连接
        
        # 前置层标准化
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)
        
        # 投影与维度处理
        batch_size, len_q = q.size(0), q.size(1)
        
        # AGKAN投影处理
        q = self._reshape_projection(self.w_qs, q, self.d_k)
        k = self._reshape_projection(self.w_ks, k, self.d_k)
        v = self._reshape_projection(self.w_vs, v, self.d_v)
        
        # 注意力计算
        output, attn = self.attention(q, k, v, mask=mask)
        
        # 输出重构
        output = output.view(self.n_head, batch_size, len_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous()
        output = output.view(batch_size, len_q, -1)
        
        # 最终投影与残差
        output = self.dropout(self.fc(output))
        return residual + output, attn

    def _reshape_projection(self, layer, x, d_head):
        """优化的维度变换方法"""
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 通过AGKAN投影
        x = layer(x)  # [B, L, n_head*d_head]
        
        # 维度重组
        x = x.view(batch_size, seq_len, self.n_head, d_head)
        return x.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_head)

    def get_gate_statistics(self):
        """获取门控网络状态用于监控"""
        return {
            'query_gate': self.w_qs.gate[0].weight.data.mean().item(),
            'key_gate': self.w_ks.gate[0].weight.data.std().item(),
            'value_gate': self.w_vs.gate[0].bias.data.mean().item()
        }




class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid,fft_conv1d_kernel,fft_conv1d_padding, dropout=0.1):
        super().__init__()


        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=fft_conv1d_kernel[0], padding=fft_conv1d_padding[0])

        # Position-wise.
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=fft_conv1d_kernel[1], padding=fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        # Pre-norm.
        x = self.layer_norm(x)
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = output + residual

        return output