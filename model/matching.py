import torch
import torch.nn as nn
import math
from einops import rearrange


class CrossAttention(nn.Module):
    '''
    Multi-Head Cross-Attention layer for Matching
    '''
    def __init__(self, dim_q, dim_v, dim_o, num_heads=4, act_fn=nn.GELU,
                 dr=0.1, pre_ln=True, ln=True, residual=True, dim_k=None):
        super().__init__()
        
        if dim_k is None:
            dim_k = dim_q
        
        # heads and temperature
        self.num_heads = num_heads
        self.dim_split_q = dim_q // num_heads
        self.dim_split_v = dim_o // num_heads
        self.temperature = math.sqrt(dim_o)
        self.residual = residual
        
        # projection layers
        self.fc_q = nn.Linear(dim_q, dim_q, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_q, bias=False)
        self.fc_v = nn.Linear(dim_v, dim_o, bias=False)
        self.fc_o = nn.Linear(dim_o, dim_o, bias=False)
        
        # nonlinear activation and dropout
        self.activation = act_fn()
        self.attn_dropout = nn.Dropout(dr)
        
        # layernorm layers
        if pre_ln:
            if dim_q == dim_k:
                self.pre_ln_q = self.pre_ln_k = nn.LayerNorm(dim_q)
            else:
                self.pre_ln_q = nn.LayerNorm(dim_q)
                self.pre_ln_k = nn.LayerNorm(dim_k)
        else:
            self.pre_ln_q = self.pre_ln_k = nn.Identity()
        self.ln = nn.LayerNorm(dim_o) if ln else nn.Identity()
        
    def compute_attention_scores(self, Q, K, mask=None, **kwargs):
        # pre-layer normalization
        Q = self.pre_ln_q(Q)
        K = self.pre_ln_k(K)
        
        # lienar projection
        Q = self.fc_q(Q)
        K = self.fc_k(K)

        # split into multiple heads
        Q_ = torch.cat(Q.split(self.dim_split_q, 2), 0)
        K_ = torch.cat(K.split(self.dim_split_q, 2), 0)

        # scaled dot-product attention with mask and dropout
        A = Q_.bmm(K_.transpose(1, 2)) / self.temperature
        A = A.clip(-1e4, 1e4)
        if mask is not None:
            A.masked_fill(mask, -1e38)
        A = A.softmax(dim=2)
        if mask is not None:
            A.masked_fill(mask, 0)
        A = self.attn_dropout(A)
        
        return A
    
    def project_values(self, V):
        # linear projection
        O = self.fc_v(V)
        
        # residual connection with non-linearity
        if self.residual:
            O = O + self.activation(self.fc_o(O))
        else:
            O = self.fc_o(O)
            
        return O

    def forward(self, Q, K, V, mask=None, get_attn_map=False, Q2=None, K2=None):
        # pre-layer normalization
        Q = self.pre_ln_q(Q)
        K = self.pre_ln_k(K)
        if Q2 is not None:
            assert K2 is not None
            Q2 = self.pre_ln_q(Q2)
            K2 = self.pre_ln_k(K2)
        
        # lienar projection
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)
        if Q2 is not None:
            Q2 = self.fc_q(Q2)
            K2 = self.fc_k(K2)

        # split into multiple heads
        Q_ = torch.cat(Q.split(self.dim_split_q, 2), 0)
        K_ = torch.cat(K.split(self.dim_split_q, 2), 0)
        V_ = torch.cat(V.split(self.dim_split_v, 2), 0)
        if Q2 is not None:
            Q2_ = torch.cat(Q2.split(self.dim_split_q, 2), 0)
            K2_ = torch.cat(K2.split(self.dim_split_q, 2), 0)
            Q_ = torch.cat([Q_, Q2_], 2)
            K_ = torch.cat([K_, K2_], 2)
        
        # scaled dot-product attention with mask and dropout
        L = Q_.bmm(K_.transpose(1, 2)) / self.temperature
        L = L.clip(-1e4, 1e4)
        
        # mask
        if mask is not None:
            mask = mask.transpose(1, 2).expand_as(L)
        
        if mask is not None:
            L.masked_fill(mask, -1e38)
            
        A = L.softmax(dim=2)
        if mask is not None:
            A.masked_fill(mask, 0)
        A = self.attn_dropout(A)
        
        # apply attention to values
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        
        # layer normalization
        O = self.ln(O)
        
        # residual connection with non-linearity
        if self.residual:
            O = O + self.activation(self.fc_o(O))
        else:
            O = self.fc_o(O)
            
        if get_attn_map:
            return O, A
        else:
            return O
        

class MatchingModule(nn.Module):
    '''
    Matching Module of VTM
    '''
    def __init__(self, dim_w, dim_z, n_heads=4, n_levels=4):
        super().__init__()
        self.matching = nn.ModuleList([CrossAttention(dim_w, dim_z, dim_z, num_heads=n_heads)
                                       for _ in range(n_levels)])
        self.n_levels = n_levels
            
    def forward(self, W_Qs, W_Ss, Z_Ss):
        if isinstance(W_Qs, tuple):
            W_Qs, W_Q2s = W_Qs
            W_Ss, W_S2s = W_Ss
        else:
            W_Q2s = W_S2s = None

        B, T, N = W_Qs[0].shape[:3]
        Z_Qs = []
        for level in range(self.n_levels):
            Q = rearrange(W_Qs[level][:, :, :, 1:], 'B T N n d -> (B T) (N n) d')
            K = rearrange(W_Ss[level][:, :, :, 1:], 'B T N n d -> (B T) (N n) d')
            V = rearrange(Z_Ss[level][:, :, :, 1:], 'B T N n d -> (B T) (N n) d')
            if W_Q2s is not None:
                Q2 = rearrange(W_Q2s[level][:, :, :, 1:], 'B T N n d -> (B T) (N n) d')
                K2 = rearrange(W_S2s[level][:, :, :, 1:], 'B T N n d -> (B T) (N n) d')
            else:
                Q2 = K2 = None
            O = self.matching[level](Q, K, V, Q2=Q2, K2=K2)
            Z_Q = rearrange(O, '(B T) (N n) d -> B T N n d', B=B, T=T, N=N)
            Z_Qs.append(Z_Q)
        
        return Z_Qs