''' Define the Layers '''
import torch
import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(nn.Module):
    ''' Compose with two layers'''
    
    def __init__(self, d_model, d_ff, n_heads, d_k, d_v):
        super(EncoderLayer, self).__init__()
                
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff)
        
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        
        return enc_outputs, attn
    
    
class DecoderLayer(nn.Module):
    
    ''' Compose with three layers'''
    
    def __init__(self, d_model, d_ff, n_heads, d_k, d_v):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.dec_enc_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff)
        
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,  dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) 
        
        return dec_outputs, dec_self_attn, dec_enc_attn
    