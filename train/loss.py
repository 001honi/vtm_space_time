import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce

from dataset.utils import calc_psnr, calc_ssim, get_depth_metric, get_modes, get_y_channel
from .miou_fss import AverageMeter, Evaluator
from PIL import Image
from torchvision.transforms import ToTensor

from dataset.unified_dataset import Unified
from dataset.coco import COCO
from dataset.utils import crop_arrays
from dataset.dataloader_factory import base_sizes
import os

SEMSEG_IDXS = [Unified.TASKS.index(task) for task in Unified.TASKS_CATEGORICAL + \
               [('coco', task) for task in COCO.TASKS_KP_HUMAN]] # Add this to train human keypoint with loss_seg


def generate_semseg_mask(t_idx):
    '''
    Generate binary mask whether the task is semantic segmentation (1) or not (0).
    '''
    semseg_mask = torch.zeros_like(t_idx, dtype=bool)
    for semseg_idx in SEMSEG_IDXS:
        semseg_mask = torch.logical_or(semseg_mask, t_idx == semseg_idx)

    return semseg_mask


def multimodal_softmax_loss(Y_src, Y_tgt, M, n_classes, dim=1, temp=1.0):
    Y_cs_src = []
    Y_cs_tgt = []
    for c in range(1 + n_classes):
        val = c / n_classes
        Y_cs_src.append(-(Y_src - val).pow(2))
        Y_cs_tgt.append(-(Y_tgt - val).pow(2))
    Y_cs_src = torch.cat(Y_cs_src, dim=dim)
    Y_cs_tgt = torch.cat(Y_cs_tgt, dim=dim)
    Y_m_tgt = torch.argmax(Y_cs_tgt, dim=dim)
    loss = F.cross_entropy(Y_cs_src / temp, Y_m_tgt, reduction='none')
    loss = (M * loss).mean()

    return loss


def multimodal_mse_loss(Y_src, Y_tgt, M, n_classes, dim=1, temp=1.0):
    Y_cs_src = []
    Y_cs_tgt = []
    for c in range(1 + n_classes):
        val = c / n_classes
        Y_cs_src.append(-(Y_src - val).pow(2))
        Y_cs_tgt.append(-(Y_tgt - val).pow(2))
    Y_cs_src = torch.cat(Y_cs_src, dim=dim)
    Y_cs_tgt = torch.cat(Y_cs_tgt, dim=dim)
    loss = F.mse_loss(Y_cs_src / temp, Y_cs_tgt / temp, reduction='none')
    loss = (M * loss).mean()

    return loss


def hybrid_loss(Y_src, Y_tgt, M, t_idx, pseudo_idxs=None, pseudo_loss_coef=0.1, pseudo_info=None, config=None):
    '''
    Compute l1 loss for continuous tasks and bce loss for semantic segmentation.
    [loss_args]
        Y_src: unnormalized prediction of shape (B, T, N, 1, H, W)
        Y_tgt: normalized GT of shape (B, T, N, 1, H, W)
        M    : mask for loss computation of shape (B, T, N, 1, H, W)
        t_idx: task index of shape (B, T)
        pseudo_idxs: pseudo task index of shape (B, T)
        pseudo_info: a tuple of (teacher name, distill type)
    '''
    # prediction loss
    loss_bce = F.binary_cross_entropy_with_logits(Y_src, Y_tgt, reduction='none')
    loss_l1 = F.l1_loss(Y_src.sigmoid(), Y_tgt, reduction='none')
    loss_spatial = spatial_softmax_loss(Y_src, Y_tgt, M, reduce='none') # B T N C
    loss_spatial = rearrange(loss_spatial, 'B T N C -> (B T) N C 1 1') / Y_src.shape[-1] / Y_src.shape[-2] # B T N C -> (B T) N C

    # loss masking
    loss_bce = rearrange((M * loss_bce), 'B T ... -> (B T) ...')
    loss_l1 = rearrange((M * loss_l1), 'B T ... -> (B T) ...')
    t_idx = rearrange(t_idx, 'B T -> (B T)')

    # create loss masks
    semseg_mask = generate_semseg_mask(t_idx)
    semseg_mask = rearrange(semseg_mask, 'B -> B 1 1 1 1').float()
    # pseudo loss masks
    pseudo_l1_mask = torch.zeros_like(semseg_mask)
    pseudo_bce_mask = torch.zeros_like(semseg_mask)
    pseudo_ssl_mask = torch.zeros_like(semseg_mask)
    
    if pseudo_idxs is not None:
        first, second, *_ = pseudo_info[0].split('_')
        if config.dpt_seg_bce and first == 'dpt' and second == 'segmentation': # use bce loss instead.
            pseudo_bce_mask = rearrange(pseudo_idxs, 'B T -> (B T) 1 1 1 1').float()
        elif first == 'dpt':
            if pseudo_info[2] == 0:
                pseudo_bce_mask = rearrange(pseudo_idxs, 'B T -> (B T) 1 1 1 1').float()
            else:
                pseudo_l1_mask = rearrange(pseudo_idxs, 'B T -> (B T) 1 1 1 1').float()
        else:
            if 'sparse' in pseudo_info[1]:
                pseudo_ssl_mask = rearrange(pseudo_idxs, 'B T -> (B T) 1 1 1 1').float()
            else:
                pseudo_l1_mask = rearrange(pseudo_idxs, 'B T -> (B T) 1 1 1 1').float()

    new_coco = COCO.NEW_COCO
    if new_coco:
        ssl_mask = torch.zeros_like(t_idx, dtype=bool)
        for kp_semantic_task in COCO.TASKS_KP_SEMANTIC:
            ssl_idx = Unified.TASKS.index(('coco', kp_semantic_task))
            ssl_mask = torch.logical_or(ssl_mask, t_idx == ssl_idx)
        ssl_mask = rearrange(ssl_mask, 'B -> B 1 1 1 1').float()
    else:
        ssl_mask = torch.zeros_like(semseg_mask)

    # compute loss
    loss_cat = semseg_mask * (1 - ssl_mask) * loss_bce
    loss_ssl = semseg_mask * ssl_mask * loss_spatial
    # loss_cat += loss_ssl.mean() # add ssl loss to categorical loss.
    loss_con = (1 - semseg_mask) * (1 - pseudo_l1_mask - pseudo_bce_mask - pseudo_ssl_mask) * loss_l1
    loss_psd_bce = (1 - semseg_mask) * pseudo_bce_mask * loss_bce
    loss_psd_l1 = (1 - semseg_mask) * pseudo_l1_mask * loss_l1
    loss_psd_ssl = (1 - semseg_mask) * pseudo_ssl_mask * loss_spatial
    loss_psd = loss_psd_bce + loss_psd_l1 + loss_psd_ssl

    loss = ((loss_cat + loss_ssl.mean())+ loss_con + pseudo_loss_coef * loss_psd).mean()

    log_loss_cat = loss_cat.detach().mean() if (semseg_mask * (1 - ssl_mask)).sum() > 0 else None
    log_loss_ssl = loss_ssl.detach().mean() if (semseg_mask * ssl_mask).sum() > 0 else None
    log_loss_con = loss_con.detach().mean() if ((1 - semseg_mask) * (1 - pseudo_l1_mask - pseudo_bce_mask)).sum() > 0 else None
    log_loss_psd = loss_psd.detach().mean() if ((1 - semseg_mask) * (pseudo_l1_mask + pseudo_bce_mask)).sum() > 0 else None
    log_loss_psd_bce = loss_psd_bce.detach().mean() if ((1 - semseg_mask) * (pseudo_bce_mask)).sum() > 0 else None
    log_loss_psd_l1 = loss_psd_l1.detach().mean() if ((1 - semseg_mask) * (pseudo_l1_mask)).sum() > 0 else None
    log_loss_psd_ssl = loss_psd_ssl.detach().mean() if ((1 - semseg_mask) * (pseudo_ssl_mask)).sum() > 0 else None

    loss_dict = {'loss': loss,
                 'loss_cat': log_loss_cat,
                 'loss_ssl': log_loss_ssl,
                 'loss_con': log_loss_con,
                 'loss_psd': log_loss_psd,
                 'loss_psd_bce': log_loss_psd_bce,
                 'loss_psd_l1': log_loss_psd_l1,
                 'loss_psd_ssl': log_loss_psd_ssl,
                 }
    
    return loss_dict
    

def spatial_normalize(x):
    x_min = reduce(x, '... h w -> ... 1 1', 'min')
    x_max = reduce(x, '... h w -> ... 1 1', 'max')
    x = (x - x_min) / (x_max - x_min)
    return x


@torch.no_grad()
def preprocess_pseudo_batch(train_data, config, teachers):
    X, Y, M, t_idx = train_data
    B, T, N = X.shape[:3]
    X = rearrange(X, 'B T N C H W -> (B T) N C H W')
    Y = rearrange(Y, 'B T N C H W -> (B T) N C H W')
    M = rearrange(M, 'B T N C H W -> (B T) N C H W')
    t_idx = rearrange(t_idx, 'B T -> (B T)')
    pseudo_idxs = (t_idx < 0)

    if pseudo_idxs.float().sum() > 0:
        # select images for pseudo tasks
        X_pseudo = rearrange(X[pseudo_idxs], 'P N C H W -> (P N) C H W')

        # choose teacher
        t_idx_offset = len(Unified.TASKS)
        teacher_idx = random.choice(range(len(teachers)))
        for i in range(teacher_idx):
            for n_distill_tasks in teachers[i].n_distill_tasks:
                t_idx_offset += n_distill_tasks
        teacher = teachers[teacher_idx]
        
        # choose distill type
        distill_type = random.choice(teacher.distill_types)

        # choose block for pseudo tasks
        if teacher.name.split('_')[0] == 'vit':
            block_idx = random.randint(config.distill_start_block, len(teacher.blocks) - 1)
        elif teacher.name.split('_')[0] == 'dpt':
            block_idx = random.randint(0, 2)
            t_idx_offset += (block_idx*teacher.features)
        
        # create pseudo label
        if distill_type in ['attention', 'attention-sparse']:
            Y_pseudo = teacher.extract_attention_map(X_pseudo, block_idx)
        elif distill_type in ['feature', 'feature-sparse']:
            Y_pseudo = teacher.extract_feature_map(X_pseudo, block_idx)
        else:
            raise NotImplementedError(f'Distill type {distill_type} not implemented.')
        Y_pseudo = rearrange(Y_pseudo, '(P N) C H W -> P N C H W', N=N)

        # sample pseudo task channels
        pseudo_channels = random.sample(range(Y_pseudo.shape[2]), Y_pseudo.shape[0])

        # select pseudo task channels
        Y_pseudo = torch.stack([Y_pseudo[i, :, c:c+1, :, :] for i, c in enumerate(pseudo_channels)])

        # upsample pseudo label
        Y_pseudo = rearrange(Y_pseudo, 'P N C H W -> (P N) C H W')
        cast = False
        if Y_pseudo.dtype != torch.float32:
            dtype = Y_pseudo.dtype
            cast = True
            Y_pseudo = Y_pseudo.float()
        Y_pseudo = F.interpolate(Y_pseudo, X_pseudo.shape[-2:], mode='bilinear', align_corners=False)
        if cast:
            Y_pseudo = Y_pseudo.to(dtype)
        Y_pseudo = rearrange(Y_pseudo, '(P N) C H W -> P N C H W', N=N)

        # normalize pseudo label
        if distill_type == 'attention':
            Y_pseudo = spatial_normalize(Y_pseudo)
        elif distill_type == 'feature':
            Y_pseudo = Y_pseudo.sigmoid()
        elif distill_type == 'attention-sparse':
            Y_pseudo = (Y_pseudo / 0.2 - 4).sigmoid()
        elif distill_type == 'feature-sparse':
            Y_pseudo = (Y_pseudo / 0.01 - 200).sigmoid()
        else:
            raise NotImplementedError

        # create pseudo mask
        M_pseudo = torch.ones_like(Y_pseudo)

        # update data
        Y[pseudo_idxs] = Y_pseudo.to(Y.dtype)
        M[pseudo_idxs] = M_pseudo.to(M.dtype)
        t_idx[pseudo_idxs] = torch.tensor(pseudo_channels, device=t_idx.device) + t_idx_offset
        pseudo_info = (teacher.name, distill_type, block_idx)
    else:
        pseudo_idxs = None
        pseudo_info = None

    X = rearrange(X, '(B T) N C H W -> B T N C H W', B=B, T=T)
    Y = rearrange(Y, '(B T) N C H W -> B T N C H W', B=B, T=T)
    M = rearrange(M, '(B T) N C H W -> B T N C H W', B=B, T=T)
    t_idx = rearrange(t_idx, '(B T) -> B T', B=B, T=T)
    if pseudo_idxs is not None:
        pseudo_idxs = rearrange(pseudo_idxs, '(B T) -> B T', B=B, T=T)

    train_data = (X, Y, M, t_idx)

    return train_data, pseudo_idxs, pseudo_info


def preprocess_dynamic_support(train_data, config, dynamic_support):
    X, Y, M = train_data
    X, Y, M = X[:config.shot], Y[:config.shot], M[:config.shot]
    X_support, Y_support, M_support = dynamic_support
    X_support, Y_support, M_support = crop_arrays(X_support, Y_support, M_support,
                                                  base_size=base_sizes[config.img_size],
                                                  crop_size=config.img_size,
                                                  random=True)
    X = torch.cat([X, X_support], dim=0)
    Y = torch.cat([Y, Y_support], dim=0)
    M = torch.cat([M, M_support], dim=0)
    train_data = (X, Y, M)
    
    return train_data
 

def compute_loss(model, train_data, config, n_classes=None, teachers=None, dynamic_support=None):
    if config.model == 'VTM':
        if teachers is not None:
            train_data, pseudo_idxs, pseudo_info = preprocess_pseudo_batch(train_data, config, teachers)
        else:
            pseudo_idxs = pseudo_info = None
        if dynamic_support is not None:
            train_data = preprocess_dynamic_support(train_data, config, dynamic_support)
        return compute_episodic_loss(model, train_data, config, n_classes=n_classes, pseudo_idxs=pseudo_idxs, pseudo_info=pseudo_info)

def compute_episodic_loss(model, train_data, config, n_classes=None, pseudo_idxs=None, pseudo_info=None):
    '''
    Compute episodic training loss for VTM.
    [train_data, from taskonomy]
        X    : input image of shape (B, T, N, 3, H, W)
        Y    : output label of shape (B, T, N, 1, H, W)
        M    : output mask of shape (B, T, N, 1, H, W)
        t_idx: task index of shape (B, T)
    [train_data, from downstream]
        X    : input image of shape (N, 3, H, W)
        Y    : output label of shape (N, T, H, W)
        M    : output mask of shape (N, T, H, W)
    [train_data, from patched data like LoveDA]
        X    : input image of shape (N, P, 3, H, W)
        Y    : output label of shape (N, P, T, H, W)
        M    : output mask of shape (N, P, T, H, W)
    '''
    if config.stage == 0 or config.stage == 3:
        X, Y, M, t_idx = train_data
    else:
        X, Y, M = train_data
        if config.task == 'flow':
            X, X2 = X

        # patch processing for loveda
        if config.task == 'semseg' and config.dataset == 'loveda':
            X = rearrange(X, 'N P C H W -> (N P) C H W', P=X.size(1))
            Y = rearrange(Y, 'N P T H W -> (N P) T H W', P=Y.size(1))
            M = rearrange(M, 'N P T H W -> (N P) T H W', P=M.size(1))

        assert X.ndim == 4
        assert Y.ndim == 4
        # multi-class processing for davis2017
        if config.task == 'mvos':
            assert n_classes is not None
            dtype = Y.dtype
            Y = F.one_hot((Y*n_classes).long())
            class_perm = torch.randperm(n_classes + 1)
            Y = torch.argmax(Y[..., class_perm], dim=-1).to(dtype=dtype)
            Y = Y / n_classes

        X = repeat(X, '(B N) C H W -> B T N C H W', B=1, T=Y.size(1))
        Y = rearrange(Y, '(B N) T H W -> B T N 1 H W', B=1)
        M = rearrange(M, '(B N) T H W -> B T N 1 H W', B=1)
        t_idx = repeat(torch.tensor(list(range(Y.size(1))), device=X.device), 'T -> B T', B=len(X))
        if config.task == 'flow':
            X2 = repeat(X2, '(B N) C H W -> B T N C H W', B=1, T=Y.size(1))

    # split the batches into support and query
    X_S, X_Q = X.split(math.ceil(X.size(2) / 2), dim=2)
    Y_S, Y_Q = Y.split(math.ceil(Y.size(2) / 2), dim=2)
    M_S, M_Q = M.split(math.ceil(M.size(2) / 2), dim=2)
    if config.task == 'flow':
        X2_S, X2_Q = X2.split(math.ceil(X2.size(2) / 2), dim=2)
        X_S = (X_S, X2_S)
        X_Q = (X_Q, X2_Q)

    # ignore masked region in support label
    if config.task == 'depth' and config.dataset == 'eigen':
        M_S = (Y_S > 0)
    Y_S_in = torch.where(M_S.bool(), Y_S, torch.ones_like(Y_S) * config.mask_value)

    # compute loss for query images
    Y_Q_pred = model(X_S, Y_S_in, X_Q, t_idx=t_idx)
    if config.stage == 0 or config.stage == 3:
        return hybrid_loss(Y_Q_pred, Y_Q, M_Q, t_idx,
                           pseudo_idxs=pseudo_idxs, pseudo_loss_coef=config.distill_weight, pseudo_info=pseudo_info, config=config)
    else:
        if config.task == 'mvos' and getattr(config, 'multimodal_softmax_loss', False):
            assert n_classes is not None
            loss = multimodal_softmax_loss(Y_Q_pred, Y_Q, M_Q, n_classes, temp=config.temp_msl)
        elif config.task == 'mvos' and getattr(config, 'multimodal_mse_loss', False):
            assert n_classes is not None
            loss = multimodal_mse_loss(Y_Q_pred, Y_Q, M_Q, n_classes, temp=config.temp_msl)
        elif config.task == 'animalkp' and getattr(config, 'spatial_softmax_loss', False):
            loss = spatial_softmax_loss(Y_Q_pred, Y_Q, M_Q)
        elif config.task == 'animalkp' and getattr(config, 'mse_loss', False):
            loss = (M_Q * F.mse_loss(Y_Q_pred, Y_Q, reduction='none')).mean()
        elif config.task in ['segment_semantic', 'vos', 'pose_6d', 'ds', 'mvos', 'sod', 'animalkp', 'semseg', 'derain']:
            if config.normalized_bce:
                eps = 1e-10
                Y_Q_pred = Y_Q_pred / (eps + reduce(Y_Q_pred, 'B T N 1 H W -> B T N 1 1 1', 'sum'))
                Y_Q = Y_Q / (eps + reduce(Y_Q, 'B T N 1 H W -> B T N 1 1 1', 'sum'))
            if config.channel_ce:
                loss = (M_Q * F.cross_entropy(Y_Q_pred, Y_Q.argmax(dim=1), reduction='none')).mean()
            else:
                loss = (M_Q * F.binary_cross_entropy_with_logits(Y_Q_pred, Y_Q, reduction='none')).mean()
        elif config.task in ['derain', 'flow'] or (config.task == 'depth' and config.dataset == 'eigen'):
            loss = (M_Q * F.l1_loss(Y_Q_pred.sigmoid(), Y_Q, reduction='none')).sum() / M_Q.sum()
        else:
            raise NotImplementedError(config.task)
    
        return {'loss': loss}


def spatial_softmax_loss(Y_pred, Y, M, reduce='mean'):
    '''
    Compute spatial softmax loss for AnimalKP.
    '''
    M = rearrange(M, 'B T N C H W -> B (H W) T N C')
    Y_pred = rearrange(Y_pred, 'B T N C H W -> B (H W) T N C')
    Y = rearrange(Y, 'B T N C H W -> B (H W) T N C')
    loss = F.cross_entropy(Y_pred*M, Y*M, reduction='none')
    if reduce == 'mean':
        loss = loss.mean()
    return loss


def compute_supervised_loss(model, train_data, config):
    '''
    Compute supervised training loss for DPT.
    [train_data]
        X    : input image of shape (B, 3, H, W)
        Y    : output label of shape (B, T, H, W)
        M    : output mask of shape (B, T, H, W)
    '''
    X, Y, M = train_data
    Y_pred = model(X)
    if config.task in ['segment_semantic', 'vos', 'pose_6d', 'ds']:
        loss = (M * F.cross_entropy(Y_pred, Y, reduction='none')).mean()
    else:
        loss = (M * F.l1_loss(Y_pred.sigmoid(), Y, reduction='none')).mean()
    
    return loss


def normalize_tensor(input_tensor, dim):
    '''
    Normalize Euclidean vector.
    '''
    norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
    zero_mask = (norm == 0)
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out


def compute_metric(Y, Y_pred, M, task, dataset, evaluator=None, stage=0, n_classes=None, Y_ori_paths=None, crop_not_resize=False, X=None, aux=None):
    '''
    Compute evaluation metric for each task.
    '''
    # Mean Angle Error
    if task == 'normal':
        pred = normalize_tensor(Y_pred, dim=1)
        gt = normalize_tensor(Y, dim=1)
        deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
        metric = (M[:, 0] * deg_diff).mean()
        
    # Mean IoU for binary segmentation
    elif 'segment_semantic' in task or task in ['vos', 'ds', 'semseg']:
        assert evaluator is not None

        if stage == 0:
            area_inter, area_union = Evaluator.classify_prediction(Y_pred.clone().float(), Y.float().round())
            assert 'segment_semantic' in task
            semseg_class = int(task.split('_')[-1])
            class_id = torch.tensor([evaluator.semseg_classes.index(semseg_class)]*len(Y_pred), device=Y.device)
            area_inter = area_inter.to(Y.device)
            area_union = area_union.to(Y.device)
            evaluator.update(area_inter, area_union, class_id)
        elif n_classes is not None:
            for c in range(n_classes):
                area_inter, area_union = Evaluator.classify_prediction((Y_pred == c + 1).float(),
                                                        (Y.float().round() == c + 1).float())
                class_id = torch.tensor([c]*len(Y_pred), device=Y.device)

                area_inter = area_inter.to(Y.device)
                area_union = area_union.to(Y.device)
                evaluator.update(area_inter, area_union, class_id)

        else:
            if task == 'ds' and Y_pred.shape[-2:] != (512, 512):
                Y = F.interpolate(Y.float(), (512, 512), mode='nearest')
                Y_pred = F.interpolate(Y_pred.float(), (512, 512), mode='nearest')
            area_inter, area_union = Evaluator.classify_prediction(Y_pred.clone().float(), Y.float().round())
            class_id = torch.tensor([0]*len(Y_pred), device=Y.device) # use 0 for all classes
            area_inter = area_inter.to(Y.device)
            area_union = area_union.to(Y.device)
            evaluator.update(area_inter, area_union, class_id)
        
        metric = 0

    elif task == 'mvos':
        if stage == 0:
            raise NotImplementedError
        else:
            assert n_classes is not None
            for c in range(n_classes):
                area_inter, area_union = Evaluator.classify_prediction((Y_pred == c + 1).float(),
                                                                       (Y.float().round() == c + 1).float())
                class_id = torch.tensor([c]*len(Y_pred), device=Y.device)

                area_inter = area_inter.to(Y.device)
                area_union = area_union.to(Y.device)
                evaluator.update(area_inter, area_union, class_id)
                
            metric = 0

    elif task == 'sod':
        if stage == 1:
            # Just compute the MAE
            mae = torch.mean(torch.abs(Y_pred - Y)).item()
            metric = mae
        elif stage == 2:
            assert evaluator is not None
            metric = torch.zeros(4, device=Y.device)
            for i in range(Y_pred.size(0)):
                y_ori_path = Y_ori_paths[i]
                if not os.path.exists(y_ori_path):
                    y_ori_path = y_ori_path.replace('/common_datasets', '/root/DATA')
                y_ori = Image.open(y_ori_path).convert('L')
                w, h = y_ori.size # PIL image size is (w, h)
                # make tensor from PIL image y_ori and load to device
                y_ori = ToTensor()(y_ori).to(Y_pred.device)

                # Resize the prediction to the original size
                pred = F.interpolate(Y_pred[i:i+1], size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
                mae, max_f, avg_f, s_score = evaluator.cal_total_metrics(pred=pred, mask=y_ori)
                metric += torch.tensor([mae, max_f, avg_f, s_score], device=metric.device)
            metric = metric / Y_pred.size(0)
        
    # BCE Loss
    elif task == 'animalkp':
        # This is a real loss used in training
        # metric = (M * F.binary_cross_entropy_with_logits(Y_pred, Y, reduction='none')).mean()
        
        # Y and Y_pred is B x 1 x H x W
        metric = 0
        assert len(Y.shape) == 4 and Y.size(1) == 1
        # Test-like evaluation metric -> distance between modes, when best match is done.
        real_modes = [(Y[b][0] == 1.0).float().nonzero().cpu() for b in range(len(Y))]
        for b in range(len(Y_pred)):
            gt = np.array(real_modes[b]) # (num_gt, 2)
            dets, scores = get_modes(Y_pred[b], return_scores=True)
            dets = np.array(dets) # (num_detection, 2)
            scores = np.array(scores) # (num_detection,)
            assert len(gt) > 0, "GT without any points detected. Check the class dict again."
            assert len(dets) > 0, "No detection. Check the get_modes function again."

            # find the closest gt for each detection 
            # use np broadcasting here
            dists = np.linalg.norm(dets[:, None] - gt[None], axis=-1) # (num_detection, num_gt)
            # sum up all minimum distance for detection
            metric += np.sum(np.min(dists, axis=1))

    elif task == 'derain':
        assert X is not None
        Y_pred = X - Y_pred # get the derained image
        Y = X - Y # get the derained image
        # PSNR only for validation, PSNR+SSIM for test
        psnr = 0
        ssim = 0
        for b in range(len(Y_pred)):
            first, second = Y_pred[b].cpu(), Y[b].cpu()
            first = first.float()
            second = second.float()
            first = get_y_channel(first, 1.)
            second = get_y_channel(second, 1.)
            psnr += calc_psnr(first, second, 1, 1, cal_type='')
            if stage == 2:
                ssim += calc_ssim(first, second, R=1)
        psnr /= len(Y_pred)
        ssim /= len(Y_pred)
        if stage == 1:
            return psnr
        return torch.tensor([psnr, ssim], device=Y.device)

    elif task == 'depth' and dataset == 'eigen':
        full_X, full_Y, full_M = aux
        assert len(full_Y.shape) == 4 and full_Y.size(1) == 1
        # Unnomalize the depth
        height, width = full_Y.shape[-2:]
        Y_pred = F.interpolate(Y_pred, size=(height, width), mode='bilinear', align_corners=False)

        # roll back log scale
        Y_pred = (np.log(23153) - np.log(506)) * Y_pred + np.log(506)
        Y_pred = Y_pred.exp() / 256.
        Y_pred = torch.clamp(Y_pred, max=80)
        full_Y = torch.clamp(full_Y, max=80)
        metrics = []
        for b in range(len(Y_pred)):
            pred, gt, mask = Y_pred[b][0].cpu().float().numpy(), full_Y[b][0].cpu().float().numpy(), full_M[b][0].cpu().float().numpy()
            # Garg crop ECCV16
            crop = np.array([0.40810811 * height,  0.99189189 * height, 0.03594771 * width,   0.96405229 * width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            metric = get_depth_metric(gt[mask], pred[mask])
            metrics.append(metric)
        metrics = torch.tensor(metrics, device=Y.device)
        metrics = metrics.mean(axis=0)

        return metrics

    # Mean Squared Error
    else:
        metric = (M * F.mse_loss(Y, Y_pred, reduction='none').pow(0.5)).mean()
        
    return metric


def convert_to_multiclass(Y, n_classes):
    Y_cs = []
    for c in range(1 + n_classes):
        val = c / n_classes
        Y_cs.append(-(Y - val).pow(2))
    Y_cs = torch.cat(Y_cs, dim=1)
    Y_m = torch.argmax(Y_cs, dim=1)

    return Y_m
