import torch
from torch import nn

class PositionalEncoding(nn.Module):
  '''
  Positional Encoding as described in section 3.5.
  '''
  def __init__(self, d_model):
    super(PositionalEncoding, self).__init__()
    # embedding size
    self.d_model = d_model
    # 2i / d_model
    self.exp = torch.arange(start=0, end=self.d_model, step=2, dtype=torch.float32) / self.d_model
    # 10000
    self.base = torch.full(size=(self.exp.shape[-1],), fill_value=10000.0, dtype=torch.float32)
    # 10000 ^ (2i / d_model)
    self.denominator = torch.pow(self.base, self.exp)

  def forward(self, x):
    # input sequence size
    sz_x = x.shape[-2]
    # initialise positional encoding for each sequence position
    pe = torch.zeros(size=(sz_x, self.d_model))
    
    # calculate positional encoding for each position in the input sequence
    for pos in range(sz_x):
      pe[pos, 0::2] = torch.sin(self.denominator) # PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
      pe[pos, 1::2] = torch.cos(self.denominator) # PE(pos, 2i+1)   = cos(pos / 10000^(2i / d_model))

    # combine input embedding and positional encoding
    x = x + pe
    return x