''' Define Scale "Scaled Dot-Product Attention" '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    ''' Compute Scaled Dot Product Attention '''
    def __init__(self, d_k,  dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # scores : (batch_size x n_heads x len_q(=len_k) x len_k(=len_q))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask==0, -1e9) # Fills element of self tensor with this value
        
        attn = self.dropout(F.softmax(scores, dim=-1))
        
        context = torch.matmul(attn, V)
        
        return context, attn