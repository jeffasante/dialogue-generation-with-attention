''' Define the sublayers in encoder/decoder layer '''

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer.Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(d_model, n_heads * d_k)
        self.W_K = nn.Linear(d_model, n_heads * d_k)
        self.W_V = nn.Linear(d_model, n_heads * d_v)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        
        self.attention = ScaledDotProductAttention(d_k)
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, Q, K, V, attn_mask):
        # Q: (batch_size x len_q x d_model), K: (batch_size x len_k x d_model, V: (batch_size x len_k x d_model))
        residual, batch_size = Q, Q.size(0)
        
        
        # Pass through the pre-attention projection: (B, S, D) proj-> (B,S,D) split->
        # Transpose for attention dot product:  (B, S, H, W) trans-> (B, H, S, W)
        q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # q: (batch_size x n_heads x len_q x d_k)
        k = self.W_K(K.float()).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # k: (batch_size x n_heads x len_q x d_k)
        v = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2) # v: (batch_size x n_heads x len_q x d_k)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : (batch_size x n_heads x len_q x len_k)
        
        # context: (batch_size x n_heads x len_q x d_v), 
        context, attn = self.attention(q, k, v, attn_mask)#ScaledDotProductAttention.forward(q, k, v, attn_mask)
        
        # Transpose to move the head dimension back: [batch_size x len_q x n_heads x d_v]
        # Combine the last two dimensions to concatenate all the heads together: [batch_size x len_q x (n_heads * d_v)]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * d_v)
        
        output = self.dropout(self.linear(context))
        
        return self.layer_norm(output + residual), attn # output: (batch_size x len_q x d_model)
    
    
    
class PositionWiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module'''
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff) # position-wise
        self.W_2 = nn.Linear(d_ff, d_model) # postion-wise
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, inputs):
        residual = inputs # inputs : (batch_size, len_q, d_model)
        output = self.W_2(self.dropout(F.relu(self.W_1(inputs))))
        output += residual
        
        return self.layer_norm(output)
        

    