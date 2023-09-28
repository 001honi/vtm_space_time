from skimage import color
import numpy as np
from einops import rearrange
from train.loss import convert_to_multiclass

import torch

from dataset.utils import flow_to_color
               

def visualize_batch(X=None, Y=None, M=None, Y_preds=None, channels=None, postprocess_fn=None, mask_type='noise'):
    '''
    Visualize a global batch consists of N-shot images and labels for T channels.
    It is assumed that images are shared by all channels, thus convert channels into RGB and visualize at once.
    '''
    
    vis = []
    
    # shape check
    assert X is not None or Y is not None or Y_preds is not None
    
    # visualize image
    if X is not None:
        img = X.cpu().float()
        vis.append(img)
    else:
        img = None
        
    # flatten labels and masks
    Ys = []
    Ms = []
    if Y is not None:
        Ys.append(Y)
        Ms.append(M)
    if Y_preds is not None:
        if isinstance(Y_preds, torch.Tensor):
            Ys.append(Y_preds)
            Ms.append(None)
        elif isinstance(Y_preds, (tuple, list)):
            for Y_pred in Y_preds:
                Ys.append(Y_pred)
                Ms.append(None)
        else:
            ValueError(f'unsupported predictions type: {type(Y_preds)}')

    # visualize labels
    if len(Ys) > 0:
        for Y in Ys:
            Y = Y.cpu().float()

            if channels is None:
                channels = list(range(Y.size(1)))

            label = Y[:, channels]

            # fill masked region with random noise
            if M is not None:
                assert Y.shape == M.shape, (Y.shape, M.shape)
                M = M.cpu().float()
                if mask_type == 'noise':
                    vis_mask = torch.rand_like(label)
                elif mask_type == 'zeros':
                    vis_mask = torch.zeros_like(label)
                elif mask_type == 'halfs':
                    vis_mask = torch.ones_like(label) * 0.5
                elif mask_type == 'ones':
                    vis_mask = torch.ones_like(label)
                else:
                    vis_mask = label
                label = torch.where(M[:, channels].bool(),
                                    label,
                                    vis_mask)

            if postprocess_fn is not None:
                label = postprocess_fn(label, img)
            
            label = visualize_label_as_rgb(label)
            vis.append(label)

    vis = torch.cat(vis)
    vis = vis.float().clip(0, 1)
    
    return vis


def postprocess_depth(label, img=None):
    label = 0.6*label + 0.4
    label = torch.exp(label * np.log(2.0**16.0)) - 1.0
    label = torch.log(label) / 11.09
    label = (label - 0.64) / 0.18
    label = (label + 1.) / 2
    label = (label*255).byte().float() / 255.
    return label


def postprocess_semseg(label, img=None, fixed_colors=True):
    COLORS = ('red', 'blue', 'yellow', 'magenta', 
              'green', 'indigo', 'darkorange', 'cyan', 'pink', 
              'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
              'purple', 'darkviolet')
    
    if label.ndim == 4:
        label = label.squeeze(1)
    
    label_vis = []
    if img is not None:
        for img_, label_ in zip(img, label):
            if fixed_colors:
                for c in range(len(COLORS)+1):
                    label_[0, c] = c

            label_vis.append(torch.from_numpy(color.label2rgb(label_.round().numpy(),
                                                              image=img_.permute(1, 2, 0).numpy(),
                                                              colors=COLORS,
                                                              kind='overlay')).permute(2, 0, 1))
    else:
        for label_ in label:
            if fixed_colors:
                for c in range(len(COLORS)+1):
                    label_[0, c] = c

            label_vis.append(torch.from_numpy(color.label2rgb(label_.round().numpy(),
                                                              colors=COLORS,
                                                              kind='overlay')).permute(2, 0, 1))
    
    label = torch.stack(label_vis)
    
    return label


def postprocess_flow(Y, img=None, base_size=(256, 256)):
    Y = Y * 2 - 1
    vis = Y * rearrange(torch.tensor(base_size, device=Y.device),
                                'c -> 1 c 1 1')
    vis = rearrange(vis, 'b c h w -> b h w c').numpy()
    vis = torch.stack([torch.from_numpy(flow_to_color(_Y)) for _Y in vis]).float() / 255
    vis = rearrange(vis, 'b h w c -> b c h w')
    
    return vis


def postprocess_kp_multiclass(label, img=None):
    '''
    For given input Y: (n, 1, h, w), make a stack of RGB color images.
    The color depends on Y's value (0, 1/17, ... , 1).
    '''
    assert label.ndim == 4, label.shape
    color_palette = [(255,255,255)] + [(i,j,k) for i in [0, 127, 255] for j in [0, 127, 255] for k in [0, 127, 255]][:-1]
    color_palette = torch.tensor(color_palette).float()
    color_palette = color_palette / 255.
    label = convert_to_multiclass(label, n_classes=17)
    label = label.long()
    label = color_palette[label]
    return label.permute(0, 3, 1, 2)


def visualize_label_as_rgb(label):
    if label.size(1) == 1:
        label = label.repeat(1, 3, 1, 1)
    elif label.size(1) == 2:
        label = torch.cat((label, torch.zeros_like(label[:, :1])), 1)
    elif label.size(1) == 5:
        label = torch.stack((
            label[:, :2].mean(1),
            label[:, 2:4].mean(1),
            label[:, 4]
        ), 1)
    elif label.size(1) != 3:
        assert NotImplementedError
        
    return label

def visualize_depth_plasma(label, cmap='plasma'):
    assert label.ndim == 4, label.shape
    import matplotlib.pyplot as plt
    label = label.squeeze(1)
    cmap = plt.get_cmap(cmap)
    # label = np.log10(91.45 - label.detach().cpu().numpy())
    # label = label / np.log10(90.45)
    label = 1 - label.cpu().numpy()
                     
    coloredDepth = cmap(label)[:, :, :, :3]
    # coloredDepth = cmap(label.detach().cpu().numpy())[:, :, :, :3]
    coloredDepth = rearrange(coloredDepth, 'b h w c -> b c h w')
        
    return coloredDepth
