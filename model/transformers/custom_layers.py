import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple
from einops import repeat, rearrange


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args):
        return args[0]


class Linear(nn.Linear):
    """
    Bias-Switching Linear layer
    """
    def __init__(self, n_bias_sets=0, additional_bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is None:
            n_bias_sets = 0

        self.n_bias_sets = n_bias_sets
        self.additional_bias = additional_bias
        self.bias2 = 0
        if self.n_bias_sets > 0:
            if self.additional_bias:
                self.bias2 = nn.Parameter(self.bias.data.clone()[None, None])

            assert self.bias is not None
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_bias_sets).contiguous())

    def forward(self, input, b_idx=None):
        if self.n_bias_sets > 0:
            assert b_idx is not None
            output = F.linear(input, self.weight, None)
            if b_idx.ndim == 1:
                if input.shape[0] == b_idx.shape[0]:
                        bias = self.bias[b_idx][:, None]
                elif input.shape[1] == b_idx.shape[0]: # default time arrangement (B=T=1)
                    bias = self.bias[b_idx][None, :]
                else:   # B or T > 1 in time attn operations 
                    BHWT, N, _ = output.shape
                    bias_split_n = len(b_idx) // N
                    bias_split_size = BHWT // bias_split_n 
                    bias_splits = []
                    for i in range(bias_split_n):
                        b = b_idx[i*N]
                        bias_split = self.bias[b].repeat(bias_split_size, N, 1)
                        bias_splits.append(bias_split)
                    bias = torch.concat(bias_splits)
            else:
                bias_mh = torch.stack(self.bias.split(self.bias.shape[1] // b_idx.shape[1], dim=1), 0)
                bias = torch.einsum('bhn,hnd->bhd', b_idx, bias_mh)
                bias = rearrange(bias, 'B h d -> B 1 (h d)')

            return output + bias + self.bias2
        else:
            return F.linear(input, self.weight, self.bias + self.bias2 if self.bias is not None else None)


class LayerNorm(nn.LayerNorm):
    """
    Bias-Switching LayerNorm
    """
    def __init__(self, n_bias_sets=0, additional_bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is None:
            n_bias_sets = 0
        
        self.n_bias_sets = n_bias_sets
        self.additional_bias = additional_bias
        self.bias2 = 0
        if self.n_bias_sets > 0:
            if self.additional_bias:
                self.bias2 = nn.Parameter(self.bias.data.clone())

            assert self.elementwise_affine
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_bias_sets).contiguous())

    def forward(self, input, b_idx=None):
        if self.n_bias_sets > 0:
            assert b_idx is not None
            output = F.layer_norm(input, self.normalized_shape, self.weight, None, self.eps)
            if b_idx.ndim == 1:
                if input.shape[0] == b_idx.shape[0]:
                    bias = self.bias[b_idx][:, None]
                elif input.shape[1] == b_idx.shape[0]: # default time arrangement (B=T=1)
                    bias = self.bias[b_idx][None, :]
                else:   # B or T > 1 in time attn operations 
                    BHWT, N, _ = output.shape
                    bias_split_n = len(b_idx) // N
                    bias_split_size = BHWT // bias_split_n 
                    bias_splits = []
                    for i in range(bias_split_n):
                        b = b_idx[i*N]
                        bias_split = self.bias[b].repeat(bias_split_size, N, 1)
                        bias_splits.append(bias_split)
                    bias = torch.concat(bias_splits)
            else:
                bias_mh = torch.stack(self.bias.split(self.bias.shape[1] // b_idx.shape[1], dim=1), 0)
                bias = torch.einsum('bhn,hnd->bhd', b_idx, bias_mh)
                bias = rearrange(bias, 'B h d -> B 1 (h d)')
            return output + bias
        else:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias + self.bias2 if self.bias is not None else None, self.eps)


class Mlp(nn.Module):
    """
    Bias-Switching MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., n_bias_sets=0, additional_bias=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = Linear(n_bias_sets, additional_bias, in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = Linear(n_bias_sets, additional_bias, hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, b_idx=None):
        x = self.fc1(x, b_idx=b_idx)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x, b_idx=b_idx)
        x = self.drop2(x)
        return x


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            input = module(*inputs)
        return input
