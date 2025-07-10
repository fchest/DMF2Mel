import torch
import torch.nn as nn
import torch.nn.functional as F
from models.kan import KANLinear

class AGKANLinear(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 S=64,          # 外部注意力压缩维度
                 grid_size=8, 
                 spline_order=3):
        super().__init__()
        # 传统KAN组件
        self.kan = KANLinear(in_features, out_features, 
                            grid_size=grid_size, 
                            spline_order=spline_order)
        
        # 外部注意力组件
        self.attn_compress = nn.Linear(in_features, S, bias=False)
        self.attn_recover = nn.Linear(S, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)
        
        # 动态融合门控
        self.gate = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
        
        # 初始化参数
        nn.init.orthogonal_(self.attn_compress.weight)
        nn.init.kaiming_uniform_(self.attn_recover.weight)

    def forward(self, x):
        # 传统KAN路径
        kan_out = self.kan(x)  # [B, L, D_out]
        
        # 注意力路径
        # 压缩阶段
        attn_weights = F.softmax(self.attn_compress(x), dim=-1)  # [B, L, S]
        # 恢复阶段
        attn_out = self.attn_recover(attn_weights)  # [B, L, D_out]
        
        # 动态门控融合
        gate = self.gate(x)  # [B, L, 1]
        output = gate * kan_out + (1 - gate) * attn_out
        
        return self.norm(output)

class AGKAN(nn.Module):
    def __init__(self, layers, S=64):
        super().__init__()
        self.net = nn.Sequential(*[
            AGKANLinear(in_dim, out_dim, S=S) 
            for in_dim, out_dim in zip(layers[:-1], layers[1:])
        ])
        
    def forward(self, x):
        return self.net(x)