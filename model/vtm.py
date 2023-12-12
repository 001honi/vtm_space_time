import torch
import torch.nn as nn
import random

from .encoder import ViTEncoder #, get_teachers
from .decoder import DPTDecoder
from .matching import MatchingModule

from dataset.unified_dataset import Unified
from dataset.taskonomy import Taskonomy


class VTM(nn.Module):
    '''
    Visual Token Matching
    '''
    def __init__(self, config, n_tasks):
        super().__init__()
        if config.time_attn:
            self.time_embed = nn.Parameter(torch.zeros(1, config.time_attn, 768))
        else:
            self.time_embed = None

        self.image_encoder = ViTEncoder(config.image_encoder, pretrained=(config.stage == 0 and
                                                                          not (config.continue_mode or
                                                                               config.resolution_finetune_mode)),
                                        in_chans=3, n_bias_sets=n_tasks, n_levels=config.n_levels,
                                        drop_path_rate=config.image_encoder_drop_path_rate,
                                        qkv_bitfit=getattr(config, 'qkv_bitfit', True),
                                        additional_bias=getattr(config, 'additional_bias', False),
                                        time_attn=config.time_attn,
                                        img_size=config.img_size)
        
        self.label_encoder = ViTEncoder(config.label_encoder, pretrained=False, in_chans=1, n_bias_sets=0,
                                        n_levels=config.n_levels,
                                        drop_path_rate=config.label_encoder_drop_path_rate,
                                        time_attn=config.time_attn,
                                        img_size=config.img_size)

        self.matching_module = MatchingModule(self.image_encoder.backbone.embed_dim,
                                              self.label_encoder.backbone.embed_dim,
                                              config.n_attn_heads, n_levels=config.n_levels)
        
        self.label_decoder = DPTDecoder(self.label_encoder.grid_size, self.label_encoder.backbone.embed_dim,
                                        hidden_features=[min(config.decoder_features*(2**i), 1024)
                                                         if config.n_levels == 4
                                                         else min(config.decoder_features*(2**(i//2)), 1024)
                                                         for i in range(config.n_levels)],
                                        deconv_head=config.deconv_head,
                                        time_attn=config.time_attn)
        
        
    def bias_parameters(self):
        # bias parameters for similarity adaptation
        for p in self.image_encoder.bias_parameters():
            yield p

    def bias_parameter_names(self):
        names = [f'image_encoder.{name}' for name in self.image_encoder.bias_parameter_names()]

        return names

    def additional_bias_parameters(self):
        # bias parameters for similarity adaptation
        for p in self.image_encoder.additional_bias_parameters():
            yield p

    def additional_bias_parameter_names(self):
        names = [f'image_encoder.{name}' for name in self.image_encoder.additional_bias_parameter_names()]

        return names

    def head_parameters(self):
        return self.label_decoder.head.parameters()

    def pretrained_parameters(self):
        return self.image_encoder.parameters()
    
    def scratch_parameters(self):
        modules = [self.label_encoder, self.matching_module, self.label_decoder]
        for module in modules:
            for p in module.parameters():
                yield p

    def label_decoder_parameters(self):
        modules = [
            self.label_decoder.resamplers,
            self.label_decoder.projectors,
            self.label_decoder.fusion_blocks,
            self.label_decoder.head
        ]
        for module in modules:
            for p in module.parameters():
                yield p
    
    def time_parameters(self):
        modules = [self.image_encoder, self.label_encoder, self.label_decoder]; i=0
        for module in modules:
            for name, p in module.time_parameters():
                # print(i, name); i+=1
                yield p 

    def forward(self, X_S, Y_S, X_Q, t_idx=None):
        # encode query input, support input and output
        if isinstance(X_S, tuple):
            X_S1, X_S2 = X_S
            X_Q1, X_Q2 = X_Q
            W_Qs = self.image_encoder(X_Q1, t_idx), self.image_encoder(X_Q2, t_idx)
            W_Ss = self.image_encoder(X_S1, t_idx), self.image_encoder(X_S2, t_idx)
        else:
            W_Qs = self.image_encoder(X_Q.float(), t_idx, shared_time_embed=self.time_embed)
            W_Ss = self.image_encoder(X_S.float(), t_idx, shared_time_embed=self.time_embed)

        Z_Ss = self.label_encoder(Y_S.float(),shared_time_embed=self.time_embed)

        # mix support output by matching
        Z_Q_preds = self.matching_module(W_Qs, W_Ss, Z_Ss)
        
        # decode support output
        Y_Q_pred = self.label_decoder(Z_Q_preds)
        
        return Y_Q_pred


class DPT(nn.Module):
    '''
    Dense Prediction Transformer
    '''
    def __init__(self, config, n_tasks):
        super().__init__()
        self.image_encoder = ViTEncoder(config.image_encoder, pretrained=(config.stage == 0 and
                                                                          not (config.continue_mode or
                                                                               config.resolution_finetune_mode)),
                                        n_bias_sets=0,
                                        in_chans=3,
                                        drop_path_rate=config.image_encoder_drop_path_rate)
        self.label_decoder = DPTDecoder(self.image_encoder.grid_size, self.image_encoder.backbone.embed_dim,
                                        hidden_features=[min(config.decoder_features*n, 1024) for n in range(1, 5)],
                                        deconv_head=config.deconv_head,
                                        out_chans=n_tasks)

    def pretrained_parameters(self):
        return self.image_encoder.parameters()

    def scratch_parameters(self):
        return self.label_decoder.parameters()

    def forward(self, X):
        # encode
        Zs = self.image_encoder(X[None, None])

        # cut off cls token
        Zs = [Z[:, :, :, 1:] for Z in Zs]

        # decode
        Y_pred = self.label_decoder(Zs)[0, 0]
        
        return Y_pred

    

def get_model(config, verbose=False):

    # set number of tasks for bitfit
    if getattr(config, 'bitfit', True):
        if config.stage == 0 and config.model == 'VTM':
            n_tasks = len(Unified.TASKS)
        else:
            if config.channel_idx < 0:
                if config.dataset == 'taskonomy':
                    n_tasks = len(Taskonomy.TASK_GROUP_DICT[config.task])
                elif config.task == 'pose_6d':
                    n_tasks = 9
                elif config.task == 'flow':
                    n_tasks = 2
                elif config.task == 'derain':
                    n_tasks = 3
                elif config.task == 'semseg':
                    n_tasks = 1
                elif config.task == 'animalkp':
                    n_tasks = 17    
                else:
                    n_tasks = 1
            else:
                n_tasks = 1
    else:
        n_tasks = 0

    if config.model == 'VTM':
        model = VTM(config, n_tasks)
        if verbose:
            print(f'Registered VTM with {n_tasks} task-specific bias parameters.')
    elif config.model == 'DPT':
        model = DPT(config, n_tasks)

    return model