import torch
import os 
os.environ["CUDA_VISIBLE_DEVICES"]=""

# VTM Base ckpt
path = "base_ckpt/TRAIN/taskonomy224_fold4_base.ckpt"
checkpoint = torch.load(path,map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']

# # VTM Time ckpt
# path = "experiments/TRAIN/MIDAIR-128/ps8-bs1-100-space/VTM_unified/checkpoints/last.ckpt"
# checkpoint = torch.load(path,map_location=torch.device('cpu'))
# model_state_dict = checkpoint['state_dict']

# Keys to be removed
# =============================================================================
remove_keys = [
    'model.label_backbone.backbone.pretrained.model.head.weight',
    'model.label_backbone.backbone.pretrained.model.head.bias',
    'model.label_backbone.backbone.pretrained.model.norm.weight',
    'model.label_backbone.backbone.pretrained.model.norm.bias'
    ]

for key in remove_keys:
    del state_dict[key]
# =============================================================================

new_state_dict = {}
for name, param in state_dict.items():

    if 'image_backbone.backbone.beit' in name:
        model_name = name.replace('image_backbone.backbone.beit',
                                  'image_encoder.backbone')
        new_state_dict[model_name] = state_dict[name].clone()

    if 'label_backbone.backbone.pretrained.model' in name:
        model_name = name.replace('label_backbone.backbone.pretrained.model',
                                  'label_encoder.backbone')
        new_state_dict[model_name] = state_dict[name].clone()
    
    if 'matching_module' in name:
        new_state_dict[name] = state_dict[name].clone()
    
    # print(f"{name:<80} {param.shape}")

torch.save(new_state_dict,"base_ckpt/TRAIN/taskonomy224_fold4.ckpt")