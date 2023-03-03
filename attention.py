import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
  '''
  Multi-Head Attention sub-layer as described in section 3.2.2. Used as part of the Encoder layer.
  '''
  def __init__(self, d_model, h):
    super(MultiHeadAttention, self).__init__()
    self.d_model = d_model # embedding size
    self.h = h # number of heads
    # embedding projection size for query, keys and values vectors
    self.d_q = self.d_k = self.d_v = d_model // h
    # linear projection layers for embeddings
    self.fc_Q = nn.Linear(in_features=d_model, out_features=d_model)
    self.fc_K = nn.Linear(in_features=d_model, out_features=d_model)
    self.fc_V = nn.Linear(in_features=d_model, out_features=d_model)
    # attention function
    self.attention = ScaledDotProductAttention()
    # linear projection layer for attention
    self.fc_mh_out = nn.Linear(in_features=d_model, out_features=d_model)

  def forward(self, Q, K, V, mask=None):
    batch_size = Q.shape[0]
    # linear projection of Q, K and V
    p_Q = self.fc_Q(Q) # [b, sz_q, d_model] -> [b, sz_q, d_model]
    p_K = self.fc_K(K) # [b, sz_k, d_model] -> [b, sz_k, d_model]
    p_V = self.fc_V(V) # [b, sz_v, d_model] -> [b, sz_v, d_model]
    # divide embedding dimension into seperate heads for Q, K, V
    p_Q = p_Q.reshape((batch_size, -1, self.h, self.d_q)) # [b, sz_q, d_model] -> [b, sz_q, h, d_q]
    p_K = p_K.reshape((batch_size, -1, self.h, self.d_k)) # [b, sz_k, d_model] -> [b, sz_k, h, d_k]
    p_V = p_V.reshape((batch_size, -1, self.h, self.d_v)) # [b, sz_v, d_model] -> [b, sz_v, h, d_v]
    # move the head dimension of Q, K and V
    p_Q = p_Q.permute((0, 2, 1, 3)) # [b, sz_q, h, d_q] -> [b, h, sz_q, d_q]
    p_K = p_K.permute((0, 2, 1, 3)) # [b, sz_k, h, d_k] -> [b, h, sz_k, d_k]
    p_V = p_V.permute((0, 2, 1, 3)) # [b, sz_v, h, d_v] -> [b, h, sz_v, d_v
    # calculate the scaled dot product attention for each head in parallel
    mh_out, mh_attn = self.attention(p_Q, p_K, p_V, mask)
    # move the head dimension of the attention weighted values
    mh_out = mh_out.permute((0, 2, 1, 3)) # [b, sz_v, h, d_v] -> [b, sz_v, h, d_v]
    # concatenate heads of attention weighted values
    mh_out = mh_out.reshape((batch_size, -1, self.d_model)) # [b, sz_v, h, d_v] -> [b, sz_v, h * d_v (d_model)]
    # linear projection of attention weighted values
    mh_out = self.fc_mh_out(mh_out) # [b, sz_v, d_model] -> [b, sz_v, d_model]

    return mh_out, mh_attn # multi-head output, multi-head attention weights
  

class ScaledDotProductAttention(nn.Module):
  '''
  Scaled Dot-Product Attention function as described in section 3.2.1. Used as part of the Multi-Head Attention layer.
  '''
  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()
    # calculate attention weights
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, Q, K, V, mask=None):
    # transpose the final 2 dimensions of K to allow multiplication with Q
    K = K.permute(0, 1, 3, 2) # [b, h, sz_k, d_k] -> [b, h, d_k, sz_k]
    # calulate attention matrix between Q and K
    attn = Q.matmul(K) # [b, h, sz_q, d_q] @ [b, h, d_k, sz_k] -> [b, h, sz_q, sz_k]
    # scale attention matrix by factor sqrt(d_k)
    attn = attn / torch.tensor(K.shape[-2])

    # mask out illegal attention value connections
    if mask is not None:
      attn = attn.masked_fill_(mask, -math.inf)

    # convert attention values to weights
    attn = self.softmax(attn)
    # multiply weighted attention with V
    out = attn.matmul(V)

    return out, attn # attention weighted values, attention weights