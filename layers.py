import torch.nn as nn
import attention # attention.py

class Encoder(nn.Module):
  '''
  Encoder as described in section 3.1. Contains multiple encoder layers.
  '''
  def __init__(self, N, d_model, h, d_ff):
    super(Encoder, self).__init__()
    # encoder of N encoder layers
    self.encoder = nn.ModuleList([EncoderLayer(d_model, h, d_ff) for i in range(N)])

  def forward(self, x):
    for layer in self.encoder:
      x = layer(x)
    return x
  
class Decoder(nn.Module):
  '''
  Decoder as described in section 3.1. Contains multiple decoder layers.
  '''
  def __init__(self, N,  d_model, h, d_ff):
    super(Decoder, self).__init__()
    # decoder of N decoder layers
    self.decoder = nn.ModuleList([DecoderLayer(d_model, h, d_ff) for i in range(N)])

  def forward(self, x, encoder_output, mask=None):
    for layer in self.decoder:
      x = layer(x, encoder_output, mask)
    return x
  
class EncoderLayer(nn.Module):
  '''
  Encoder layer as described in section 3.1. Contains the multi-head attention and feed forward network sub-layers.
  '''
  def __init__(self, d_model, h, d_ff):
    super(EncoderLayer, self).__init__()
    # multi-head attention sub-layer
    self.mha = attention.MultiHeadAttention(d_model, h)
    # multi-head attention layer norm
    self.layer_norm_mha = nn.LayerNorm(normalized_shape=d_model)
    # feed forward network sub-layer
    self.ffn = FeedForwardNetwork(d_model, d_ff)
    # feed foward network layer norm
    self.layer_norm_ffn = nn.LayerNorm(normalized_shape=d_model)

  def forward(self, x):
    query = keys = values = x
    mha_out, mha_attn = self.mha(query, keys, values)
    # residual connection and layer norm
    x = self.layer_norm_mha(x + mha_out)
    # feed forward network
    ffn_out = self.ffn(x)
    # residual connection and layer norm
    x = self.layer_norm_ffn(x + ffn_out)
    return x
  
class DecoderLayer(nn.Module):
  '''
  Decoder layer as described in section 3.1. Contains the multi-head attention and feed forward network sub-layers.
  '''
  def __init__(self, d_model, h, d_ff):
    super(DecoderLayer, self).__init__()
    # masked multi-head attention sub-layer
    self.masked_mha = attention.MultiHeadAttention(d_model, h)
    # masked multi-head attention layer norm
    self.layer_norm_masked_mha = nn.LayerNorm(normalized_shape=d_model)

    # multi-head attention sub-layer
    self.mha = attention.MultiHeadAttention(d_model, h)
    # multi-head attention layer norm
    self.layer_norm_mha = nn.LayerNorm(normalized_shape=d_model)

    # feed forward network sub-layer
    self.ffn = FeedForwardNetwork(d_model, d_ff)
    # feed foward network layer norm
    self.layer_norm_ffn = nn.LayerNorm(normalized_shape=d_model)

  def forward(self, x, encoder_output, mask=None):
    # masked multi-head attention
    query = keys = values = x
    masked_mha_out, masked_mha_attn = self.masked_mha(query, keys, values, mask)
    # residual connection and layer norm
    x = self.layer_norm_masked_mha(x + masked_mha_out)

    # multi-head attention
    query = x
    keys = values = encoder_output
    mha_out, mha_attn = self.mha(query, keys, values)
    # residual connection and layer norm
    x = self.layer_norm_mha(x + mha_out)

    # feed forward network
    ffn_out = self.ffn(x)
    # residual connection and layer norm
    x = self.layer_norm_ffn(x + ffn_out)

    return x
  
class FeedForwardNetwork(nn.Module):
    '''
    Position-wise Feed Forward Network sub-layer as described in section 3.3. Used as part of the Encoder layer.
    '''
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        # feed forward network layers
        self.fc_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.fc_2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc_2(self.relu(self.fc_1(x)))