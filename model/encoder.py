import torch.nn as nn
from einops import rearrange, repeat

from .transformers.factory import create_model
from .transformers.custom_layers import Identity
# from .dpt.models import DPTSegmentationModel, DPTDepthModel

        
class ViTEncoder(nn.Module):
    '''
    Vision Transformer encoder wrapper for VTM
    '''
    def __init__(self, backbone, pretrained, in_chans, n_bias_sets, n_levels=4, qkv_bitfit=True, 
                 additional_bias=False, time_attn=0, img_size=224, **kwargs):
        super().__init__()
        self.backbone = create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            global_pool='',
            num_classes=0,
            n_bias_sets=n_bias_sets,
            additional_bias=additional_bias,
            qkv_bitfit=qkv_bitfit,
            time_attn=time_attn,
            img_size=img_size,
            **kwargs
        )
        self.grid_size = self.backbone.patch_embed.grid_size
        self.backbone.norm = Identity()
        self.qkv_bitfit = qkv_bitfit
        self.n_levels = n_levels
        self.feature_idxs = [level * (len(self.backbone.blocks) // self.n_levels) - 1
                             for level in range(1, self.n_levels + 1)]
    
    def bias_parameters(self):
        for name, p in self.backbone.named_parameters():
            if (name[-4:] == 'bias' and p.ndim == 2 and
                    (self.qkv_bitfit or name[-6:] not in ['q_bias', 'k_bias', 'v_bias'])):
                yield p
    
    def additional_bias_parameters(self):
        for name, p in self.backbone.named_parameters():
            if name[-5:] == 'bias2':
                yield p

    def bias_parameter_names(self):
        names = []
        for name, p in self.backbone.named_parameters():
            if (name[-4:] == 'bias' and p.ndim == 2 and
                    (self.qkv_bitfit or name[-6:] not in ['q_bias', 'k_bias', 'v_bias'])):
                names.append(f'backbone.{name}')
        return names

    def additional_bias_parameter_names(self):
        names = []
        for name, p in self.backbone.named_parameters():
            if name[-5:] == 'bias2':
                names.append(f'backbone.{name}')
        return names

    def time_parameters(self):
        for name, p in self.backbone.named_parameters():
            if (name.find('time')    != -1 or
                name.find('attn_t')  != -1 or
                name.find('norm0')   != -1 or 
                name.find('ls0')     != -1 or
                name.find('gamma_0') != -1):
                
                yield name, p
            
    def forward(self, x, t_idx=None, shared_time_embed=None):
        '''
        [input]
            x: (B, T, N, C, H, W)
            b_idx: None or (B, T)
        [output]
            features: dict of (B, T, N, hw+1, d)
        '''
        B, T, N = x.shape[:3]; vtm_shape = x.shape
        
        # flatten tensors
        x = rearrange(x, 'B T N C H W -> (B T N) C H W').contiguous()

        # repeat task index for shots
        if t_idx is not None:
            if isinstance(t_idx, list):
                t_idx = [repeat(t, 'B T ... -> (B T N) ...', N=N) for t in t_idx]
            else:
                t_idx = repeat(t_idx, 'B T -> (B T N)', N=N)

        features = self.backbone.forward_features(x, b_idx=t_idx, feature_idxs=self.feature_idxs, vtm_shape=vtm_shape, shared_time_embed=shared_time_embed)
        features = [rearrange(feat, '(B T N) n d -> B T N n d', B=B, T=T, N=N) for feat in features]

        return features


# def get_teachers(config, verbose=False):
#     teachers = []
#     n_pseudo_tasks = 0
#     for teacher_encoder in config.teacher_encoder:
#         if teacher_encoder.split('_')[0] == 'vit':
#             teacher = create_model(
#                 teacher_encoder,
#                 pretrained=True,
#                 in_chans=3,
#                 global_pool='',
#                 num_classes=0,
#             ).eval()
#             teacher.distill_types = ['attention', 'feature', 'attention-sparse', 'feature-sparse']
#             teacher.n_distill_tasks = [teacher.num_heads, teacher.embed_dim, teacher.num_heads, teacher.embed_dim]
#             n_pseudo_tasks += 2*(teacher.num_heads + teacher.embed_dim)

#         elif teacher_encoder.split('_')[0] == 'dpt':
#             if teacher_encoder.split('_')[1] == 'depth':
#                 teacher = DPTDepthModel(
#                     path='model/pretrained_checkpoints/dpt_large-midas-2f21e586.pt',
#                     backbone="vitl16_384",
#                     non_negative=True,
#                     enable_attention_hooks=False,
#                 ).eval()
#             elif teacher_encoder.split('_')[1] == 'segmentation':
#                 teacher = DPTSegmentationModel(
#                     150,
#                     path='model/pretrained_checkpoints/dpt_hybrid-ade20k-53898607.pt',
#                     backbone="vitb_rn50_384",
#                 ).eval()
#             else:
#                 raise ValueError(f'Invalid model name for DPT: {teacher_encoder}')
            
#             teacher.distill_types = ['feature']
#             teacher.n_distill_tasks = [3*teacher.features]
#             n_pseudo_tasks += (3*teacher.features)
        
#         else:
#             raise NotImplementedError(f'Invalid teacher encoder: {teacher_encoder}')

#         teacher.name = teacher_encoder
#         teachers.append(teacher)
#         if verbose:
#             print('Registered teacher encoder: ', teacher_encoder)

#     teachers = nn.ModuleList(teachers)
#     for teacher in teachers:
#         # freeze teacher parameters
#         for param in teacher.parameters():
#             param.requires_grad = False
#     return teachers, n_pseudo_tasks