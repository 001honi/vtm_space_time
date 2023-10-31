""" BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
Model from official source: https://github.com/microsoft/unilm/tree/master/beit
and
https://github.com/microsoft/unilm/tree/master/beit2
@inproceedings{beit,
title={{BEiT}: {BERT} Pre-Training of Image Transformers},
author={Hangbo Bao and Li Dong and Songhao Piao and Furu Wei},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=p-BhZSz59o4}
}
@article{beitv2,
title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
year={2022},
eprint={2208.06366},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
At this point only the 1k fine-tuned classification weights and model configs have been added,
see original source above for pre-training models and procedure.
Modifications by / Copyright 2021 Ross Wightman, original copyrights below
"""
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from timm.models.layers import PatchEmbed, DropPath, trunc_normal_
from .registry import register_model
from .vision_transformer import checkpoint_filter_fn
from .custom_layers import Identity, Linear, LayerNorm, Mlp
from einops import rearrange


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'beit_base_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth'),
    'beit_base_patch16_384': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_384_pt22k_ft22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_base_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D',
        # url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth',
        num_classes=21841,
    ),
    'beit_large_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22kto1k.pth'),
    'beit_large_patch16_384': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_384_pt22k_ft22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_large_patch16_512': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_512_pt22k_ft22kto1k.pth',
        input_size=(3, 512, 512), crop_pct=1.0,
    ),
    'beit_large_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k.pth',
        num_classes=21841,
    ),

    'beitv2_base_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21kto1k.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_base_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k.pth',
        num_classes=21841,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_base_patch16_224_in22k_ft21k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth',
        num_classes=21841,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21kto1k.pth',
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth',
        num_classes=21841,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_384': _cfg(
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384)
    ),
    'beitv2_large_patch16_416': _cfg(
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 416, 416)
    ),
}


def gen_relative_position_index(window_size: Tuple[int, int]) -> torch.Tensor:
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(torch.meshgrid(
        [torch.arange(window_size[0]),
         torch.arange(window_size[1])], indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(size=(window_area + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, n_bias_sets=0, additional_bias=False, qkv_bitfit=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(n_bias_sets, additional_bias, dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias2 = self.v_bias2 = 0
            if n_bias_sets > 0 and qkv_bitfit:
                self.q_bias = nn.Parameter(torch.zeros(n_bias_sets, all_head_dim))
                self.register_buffer('k_bias', torch.zeros(n_bias_sets, all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(n_bias_sets, all_head_dim))
                if additional_bias:
                    self.q_bias2 = nn.Parameter(torch.zeros(1, 1, all_head_dim))
                    self.register_buffer('k_bias2', torch.zeros(1, 1, all_head_dim), persistent=False)
                    self.v_bias2 = nn.Parameter(torch.zeros(1, 1, all_head_dim))
                else:
                    self.k_bias2 = 0
            else:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.k_bias2 = 0
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.n_bias_sets = n_bias_sets
        self.qkv_bitfit = qkv_bitfit

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            self.register_buffer("relative_position_index", gen_relative_position_index(window_size))
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(n_bias_sets, additional_bias, all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None, b_idx=None):
        B, N, C = x.shape

        if self.n_bias_sets > 0 and self.qkv_bitfit:
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=None)
            if self.q_bias is not None:
                assert b_idx is not None
                if b_idx.ndim == 1:
                    if x.shape[0] == b_idx.shape[0]:
                        qkv_bias = torch.cat((self.q_bias[b_idx][:, None], self.k_bias[b_idx][:, None], self.v_bias[b_idx][:, None]), 2) 
                    elif x.shape[1] == b_idx.shape[0]:
                        qkv_bias = torch.cat((self.q_bias[b_idx][None, :], self.k_bias[b_idx][None, :], self.v_bias[b_idx][None, :]), 2)
                    else:
                        BHWT, N, _ = x.shape
                        bias_split_n = len(b_idx) // N
                        bias_split_size = BHWT // bias_split_n 
                        qkv_bias_splits = []
                        for i in range(bias_split_n):
                            b = b_idx[i*N]
                            q_bias = self.q_bias[b].repeat(bias_split_size, N, 1)
                            k_bias = self.k_bias[b].repeat(bias_split_size, N, 1)
                            v_bias = self.v_bias[b].repeat(bias_split_size, N, 1)
                            qkv_bias_split = torch.cat((q_bias,k_bias,v_bias),2)
                            qkv_bias_splits.append(qkv_bias_split)
                        qkv_bias = torch.concat(qkv_bias_splits)

                else:
                    q_bias_mh = torch.stack(self.q_bias.split(self.q_bias.shape[1] // b_idx.shape[1], dim=1), 0)
                    k_bias_mh = torch.stack(self.k_bias.split(self.k_bias.shape[1] // b_idx.shape[1], dim=1), 0)
                    v_bias_mh = torch.stack(self.v_bias.split(self.v_bias.shape[1] // b_idx.shape[1], dim=1), 0)
                    q_bias = rearrange(torch.einsum('bhn,hnd->bhd', b_idx, q_bias_mh), 'B h d -> B 1 (h d)') + self.q_bias2
                    k_bias = rearrange(torch.einsum('bhn,hnd->bhd', b_idx, k_bias_mh), 'B h d -> B 1 (h d)') + self.k_bias2
                    v_bias = rearrange(torch.einsum('bhn,hnd->bhd', b_idx, v_bias_mh), 'B h d -> B 1 (h d)') + self.v_bias2
                    qkv_bias = torch.cat((q_bias, k_bias, v_bias), 2)

                qkv = qkv + qkv_bias
        else:
            qkv_bias = torch.cat((self.q_bias + self.q_bias2, self.k_bias + self.k_bias2, self.v_bias + self.v_bias2)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            if attn.shape[-1] == self._get_rel_pos_bias().shape[-1]: #FIXME 5-Crop interpolation
                # if self._get_rel_pos_bias().shape[-1] != 2:
                attn = attn + self._get_rel_pos_bias()
        if shared_rel_pos_bias is not None:
            attn = attn + shared_rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x, b_idx=b_idx)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            window_size=None, attn_head_dim=None, n_bias_sets=0, additional_bias=False,
            qkv_bitfit=True, time_attn=0):
        super().__init__()
        self.time_attn = time_attn
        self.norm1 = norm_layer(n_bias_sets, additional_bias, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            window_size=window_size, attn_head_dim=attn_head_dim, n_bias_sets=n_bias_sets, additional_bias=additional_bias, qkv_bitfit=qkv_bitfit)
        # Initialize time-layers for time-attention
        # Note: layer_name + ‘0’ or ‘_t’ indicates time layers
        if time_attn:
            self.norm0 = norm_layer(n_bias_sets, additional_bias, dim)
            self.attn_t = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            window_size=(time_attn-1,1), attn_head_dim=attn_head_dim, n_bias_sets=n_bias_sets, additional_bias=additional_bias, qkv_bitfit=qkv_bitfit)
            self.temp_fc = Linear(n_bias_sets, additional_bias, dim, dim)
            self.time_gate = nn.Parameter(torch.zeros(1,time_attn,dim))
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(n_bias_sets, additional_bias, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, n_bias_sets=n_bias_sets, additional_bias=additional_bias)

        if init_values:
            self.gamma_0 = nn.Parameter(init_values * torch.ones(dim)) if time_attn else None
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None, b_idx=None, vtm_shape=None):
        if self.gamma_1 is None:
            if self.time_attn:
                B,T,N,C,H,W = vtm_shape
                x = x[:, 1:] # Nx196x768 ; B=T=1
                x = rearrange(x, '(B T N) k d -> (B T k) N d',B=B,T=T,N=N) # Nx196x768 -> 196xNx768 ; B=T=1
                x = rearrange(x, '(B T k) N d -> (B T) (k N) d',B=B,T=T,N=N) # 196xNx768 -> 1x392x768 ; B=T=1
                x = rearrange(x, '(B T) (H W N) d -> (B T H W) N d',B=B,T=T,N=N,H=H,W=W) # 1x392x768 -> 196xNx768 ; B=T=1
                x_res = self.drop_path(self.attn_t(self.norm0(x, b_idx), b_idx=b_idx)) # 196xNx768 ; B=T=1
                x_res = self.temp_fc(x_res, b_idx)
                # Suppress time attn in early iterations by gating
                # x_res = x_res * self.time_gate if x_res.shape[1] == self.time_gate.shape[1] else x_res * self.time_gate[:,0,:]
                x = x + x_res
                # [Uncomment only one] ---------------------------------------------
                # x = x + self.temp_fc(x, b_idx)
                x = rearrange(x, '(B T k) N d -> (B T N) k d',B=B,T=T,N=N) # 196xNx768 -> Nx196x768 ; B=T=1 
                # ------------------------------------------------------------------
                # Temporal FC (Type 0) (Old)
                # x = rearrange(x, '(B T H W) N d -> (B T) (H W N) d',B=B,T=T,N=N,H=H,W=W) # 196xNx768 -> 1x392x768 ; B=T=1
                # x = x + self.temp_fc(x, b_idx)
                # x = rearrange(x, '(B T) (H W N) d -> (B T N) (H W) d',B=B,T=T,N=N,H=H,W=W) # 1x392x768 -> Nx196x768 ; B=T=1
                # ------------------------------------------------------------------
                # Temporal FC (Type I)
                # x = x + self.temp_fc(x, b_idx) # 196xNx768 ; B=T=1
                # x = rearrange(x, '(B T k) N d -> (B T N) k d',B=B,T=T,N=N) # 196xNx768 -> Nx196x768 ; B=T=1 
                # ------------------------------------------------------------------
                # Temporal FC (Type II)
                # x = rearrange(x, '(B T k) N d -> (B T N) k d',B=B,T=T,N=N) # 196xNx768 -> Nx196x768 ; B=T=1 
                # x = x + self.temp_fc(x, b_idx)
                init_cls_token = x[:,0,:].unsqueeze(1)
                cls_token = init_cls_token.repeat(1, 1, 1)
                x = torch.cat((cls_token, x), dim=1)

            x = x + self.drop_path(self.attn(self.norm1(x, b_idx), shared_rel_pos_bias=shared_rel_pos_bias, b_idx=b_idx))
            x = x + self.drop_path(self.mlp(self.norm2(x, b_idx), b_idx))
       
        else:
            if self.time_attn:
                B,T,N,C,H,W = vtm_shape
                x = x[:, 1:] # Nx196x768 ; B=T=1
                x = rearrange(x, '(B T N) k d -> (B T k) N d',B=B,T=T,N=N) # Nx196x768 -> 196xNx768 ; B=T=1
                x = rearrange(x, '(B T k) N d -> (B T) (k N) d',B=B,T=T,N=N) # 196xNx768 -> 1x392x768 ; B=T=1
                x = rearrange(x, '(B T) (H W N) d -> (B T H W) N d',B=B,T=T,N=N,H=H,W=W) # 1x392x768 -> 196xNx768 ; B=T=1
                x_res = self.drop_path(self.attn_t(self.norm0(x, b_idx), b_idx=b_idx)) # 196xNx768 ; B=T=1
                x_res = self.temp_fc(x_res, b_idx)
                # Suppress time attn in early iterations by gating
                # x_res = x_res * self.time_gate if x_res.shape[1] == self.time_gate.shape[1] else x_res * self.time_gate[:,0,:]
                x = x + x_res
                # [Uncomment only one] ---------------------------------------------
                # x = x + self.temp_fc(x, b_idx)
                x = rearrange(x, '(B T k) N d -> (B T N) k d',B=B,T=T,N=N) # 196xNx768 -> Nx196x768 ; B=T=1 
                # ------------------------------------------------------------------
                # Temporal FC (Type 0) (Old)
                # x = rearrange(x, '(B T H W) N d -> (B T) (H W N) d',B=B,T=T,N=N,H=H,W=W) # 196xNx768 -> 1x392x768 ; B=T=1
                # x = x + self.temp_fc(x, b_idx)
                # x = rearrange(x, '(B T) (H W N) d -> (B T N) (H W) d',B=B,T=T,N=N,H=H,W=W) # 1x392x768 -> Nx196x768 ; B=T=1
                # ------------------------------------------------------------------
                # Temporal FC (Type I)
                # x = x + self.temp_fc(x, b_idx) # 196xNx768 ; B=T=1
                # x = rearrange(x, '(B T k) N d -> (B T N) k d',B=B,T=T,N=N) # 196xNx768 -> Nx196x768 ; B=T=1 
                # ------------------------------------------------------------------
                # Temporal FC (Type II)
                # x = rearrange(x, '(B T k) N d -> (B T N) k d',B=B,T=T,N=N) # 196xNx768 -> Nx196x768 ; B=T=1 
                # x = x + self.temp_fc(x, b_idx)
                init_cls_token = x[:,0,:].unsqueeze(1)
                cls_token = init_cls_token.repeat(1, 1, 1)
                x = torch.cat((cls_token, x), dim=1)

            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x, b_idx), shared_rel_pos_bias=shared_rel_pos_bias, b_idx=b_idx))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x, b_idx), b_idx))          

        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.register_buffer("relative_position_index", gen_relative_position_index(window_size))

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area + 1, self.window_area + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class Beit(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='avg',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
            attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(LayerNorm, eps=1e-6),
            init_values=None, use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
            head_init_scale=0.001, n_bias_sets=0, additional_bias=False, n_feature_levels=1, qkv_bitfit=True,
            n_interaction_blocks=0, interaction_type='global', interaction_bias_sets=0, time_attn=0):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.grad_checkpointing = False
        self.time_attn = time_attn

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.grid_size, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if time_attn:
            self.time_embed = nn.Parameter(torch.zeros(1, time_attn, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.grid_size if use_rel_pos_bias else None,
                n_bias_sets=n_bias_sets, additional_bias=additional_bias, qkv_bitfit=qkv_bitfit, time_attn=time_attn)
            for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = Identity() if use_fc_norm else norm_layer(n_bias_sets, additional_bias, embed_dim)
        self.fc_norm = norm_layer(0, False, embed_dim) if use_fc_norm else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        # initialization of temporal fc weights (TimeSformer'21)
        if time_attn:
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temp_fc.weight, 0)
                      nn.init.constant_(m.temp_fc.bias, 0)
                    i += 1

        self.feature_blocks = [level * (len(self.blocks) // n_feature_levels) - 1 for level in range(1, n_feature_levels + 1)]

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed|rel_pos_bias',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, b_idx=None, interaction_mask=None, feature_idxs=None, vtm_shape=None):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.time_attn:
            B,T,N,C,H,W = vtm_shape
            W = H = H // self.patch_embed.patch_size[0]
            vtm_shape = (B,T,N,C,H,W)
            # Separate CLS token
            x = x[:, 1:]
            x = rearrange(x, '(B T N) k d -> (B T k) N d',B=B,T=T,N=N) # Nx196x768 -> 196xNx768 ; B=T=1 
            # Interpolation, in case embedding size mismatch in the inference
            if x.shape[1] != self.time_embed.shape[1]:
                x = x + F.interpolate(self.time_embed.transpose(1,2),
                    size=(x.shape[1]), mode='nearest').transpose(1, 2)
            else:
                x = x + self.time_embed
            x = self.time_drop(x)

            x = rearrange(x, '(B T k) N d -> (B T N) k d',B=B,T=T,N=N) # 196xNx768 -> Nx196x768 ; B=T=1 
            # Add CLS token 
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if feature_idxs is not None:
            features = []
        if isinstance(b_idx, list):
            b_idx_list = b_idx
            b_idx = b_idx_list.pop(0)
        else:
            b_idx_list = None
        for i, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                assert b_idx is None
                x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
            else:
                x = blk(x, shared_rel_pos_bias=rel_pos_bias, b_idx=b_idx, vtm_shape=vtm_shape)
            if feature_idxs is not None and i in feature_idxs:
                features.append(x)
                if i == max(feature_idxs):
                    break
                if b_idx_list is not None:
                    b_idx = b_idx_list.pop(0)

        if feature_idxs is not None:
            return features
        else:
            x = self.norm(x, b_idx)
            return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.fc_norm is not None:
            x = x[:, 1:].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x, b_idx=None):
        x = self.forward_features(x, b_idx=b_idx)
        x = self.forward_head(x)
        return x


def _beit_checkpoint_filter_fn(state_dict, model):
    if 'module' in state_dict:
        # beit v2 didn't strip module
        state_dict = state_dict['module']
    return checkpoint_filter_fn(state_dict, model)


def _create_beit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Beit models.')

    model = build_model_with_cfg(
        Beit, variant, pretrained,
        # FIXME an updated filter fn needed to interpolate rel pos emb if fine tuning to diff model sizes
        pretrained_filter_fn=_beit_checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def beit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_base_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_base_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    if kwargs['img_size'] == 64:
        model_kwargs = dict(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_512(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beit_large_patch16_512', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beit_large_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_base_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_base_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_base_patch16_224_in22k_ft21k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_base_patch16_224_in22k_ft21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beitv2_large_patch16_416(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beitv2_large_patch16_416', pretrained=pretrained, **model_kwargs)
    return model
