''' Define the Transformer model '''

import torch
import torch.nn as nn
import numpy as np


# Positional Encoding with sine table algo.

def get_sinusiod_encoding_table(n_position, d_model):
    ''' Sinusoid position encoding table '''
    drop = nn.Dropout(p=0.1)
    
    def get_postion_angle_vec(position):
        
        return [calc_angle(position, hid_j) for hid_j in range(d_model) ]

    def calc_angle(position, hid_idx):
        
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    
    sinusoid_table = torch.tensor([get_postion_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2]) # dim 2i + i

    return drop(sinusoid_table) #sinusoid_table #.unsqueeze(0)

def get_attn_pad_mask(seq_q, seq_k):
    bs, len_q = seq_q.size()
    bs, len_k = seq_k.size()
    
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # batch_size x 1 x len_k(=len_q), one is masking
    
    return pad_attn_mask.expand(bs, len_q, len_k) # batch_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    ''' For masking out the subsequent info'''
    
    attn_shape = seq.size()
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).byte() 
    
    return subsequent_mask
    


class Encoder(nn.Module):
    '''An encoder model with stacked self.attention mechanism. '''
    
    def __init__(self, src_vocab_size, d_model, d_ff, seq_length, n_layers, n_heads, d_k, d_v, dropout=0.1):
        super(Encoder, self).__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
#         print(src_vocab_size)
        self.position_embed = nn.Embedding.from_pretrained(get_sinusiod_encoding_table(src_vocab_size, d_model), freeze=True)
        self.stacked_layers = nn.ModuleList([EncoderLayer(d_model, d_ff, 
                                                          n_heads, d_k, d_v) for _ in range(n_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.seq_len = seq_length
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_length]
        ''' Pass thr input and mask through each layer in turn.'''
        
        # -- Forward
        
        enc_outputs = self.dropout(self.src_embed(enc_inputs)) + self.position_embed(enc_inputs.squeeze()) 
        enc_outputs = self.layer_norm(enc_outputs)
        
#         print(enc_inputs.shape)
        
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        
        enc_self_attns_list = []
        
        # residual connection
        for layer in self.stacked_layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
        
            enc_self_attns_list.append(enc_self_attn)
            
        return enc_outputs, enc_self_attn


        
 
 class Decoder(nn.Module):
    '''A Decoder model with self.attention mechanism. '''
    
    def __init__(self,  response_vocab_size, d_model, seq_length, n_layers, n_heads, d_k, d_v, dropout=0.1):
        super(Decoder, self).__init__()
        self.tgt_embed = nn.Embedding(response_vocab_size, d_model)
        self.position_embed = nn.Embedding.from_pretrained(get_sinusiod_encoding_table(response_vocab_size, d_model), freeze=True)
        self.stacked_layers = nn.ModuleList([DecoderLayer(d_model, d_ff, 
                                                          n_heads, d_k, d_v) for _ in range(n_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.seq_len = seq_length
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_length]
        
        # -- Forward
        dec_outputs = self.dropout(self.tgt_embed(dec_inputs)) + self.position_embed(dec_inputs.squeeze()) 

        dec_outputs = self.layer_norm(dec_outputs)
        
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        
        
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
    
        
        dec_self_attns_list, dec_enc_attns_list = [], []        
        
        for layer in self.stacked_layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             dec_self_attn_mask, dec_enc_attn_mask)
        
            dec_self_attns_list.append(dec_self_attn)
            dec_enc_attns_list.append(dec_enc_attn)
            
            
        return dec_outputs, dec_self_attns_list, dec_enc_attns_list

    



class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    
    def __init__(self, src_vocab_size, d_model, 
                 d_ff, src_len, n_layers, n_heads, d_k, d_v):
        super(Transformer, self).__init__()
        self.encoder= Encoder(
            src_vocab_size, d_model, d_ff, src_len, n_layers, n_heads, d_k, d_v
            )
        
        
        self.decoder= Decoder(
            src_vocab_size, d_model, src_len, n_layers, n_heads, d_k, d_v
            )
        
        # src_len is the same as tgt_len
        self.projection = nn.Linear(d_model, src_len, bias=False)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
                
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size]
        
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
        
