import json
from zipfile import ZipFile
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
import torch.distributed as dist
from einops import rearrange, repeat, reduce
import os
# from xtcocotools.cocoeval import COCOeval
# from xtcocotools.coco import COCO
from PIL import Image
import math

from model.vtm import get_model

from dataset.dataloader_factory import get_train_loader, get_eval_loader, get_validation_loaders, generate_support_data, base_sizes
from dataset.utils import get_modes, mix_overlapped_crop, modes_to_array, ret_to_coco, simple_mix_overlapped_crop, to_device, mix_fivecrop, crop_arrays, dense_crf
from dataset.unified_dataset import test_dataset_dict
from .duts_eval import DUTS_Evaluator

from .optim import get_optimizer
from .loss import compute_loss, compute_metric, convert_to_multiclass
from .visualize import visualize_batch, postprocess_depth, postprocess_semseg, postprocess_flow, postprocess_kp_multiclass, visualize_depth_plasma, visualize_label_as_rgb
from .miou_fss import AverageMeter


class LightningTrainWrapper(pl.LightningModule):
    def __init__(self, config, verbose=True):
        '''
        Pytorch lightning wrapper for Visual Token Matching.
        '''
        super().__init__()

        # load model.
        self.model = get_model(config, verbose=verbose)
        self.config = config
        self.verbose = verbose
        self.teachers = None # disable 

        # tools for validation.
        self.crop = T.Compose([
            T.FiveCrop(config.img_size),
            T.Lambda(lambda crops: torch.stack([crop for crop in crops]))
        ])
        if self.config.model == 'VTM':
            if not (self.config.stage == 0 and self.config.no_eval):
                self.support_data = self.load_support_data()
            if self.config.stage == 1:
                for attn in self.model.matching_module.matching:
                    attn.attn_dropout.p = self.config.attn_dropout

        if self.config.dataset == 'davis2017':
            n_classes = torch.load('dataset/meta_info/davis2017_n_objects.pth')
            self.n_classes = n_classes[self.config.class_name]
        elif self.config.multichannel:
            if self.config.dataset == 'loveda':
                self.n_classes = 7
            elif self.config.dataset == 'ap10k':
                self.n_classes = 17
            else:
                self.n_classes = None            
        else:
            self.n_classes = None
        self.topil = T.ToPILImage()

        # save hyper=parameters
        self.save_hyperparameters()

        # load train data to prevent redundant loading
        if config.stage == 0 and not getattr(self.config, 'no_train', False):
            self.train_data = get_train_loader(self.config, verbose=(self.verbose and verbose))

        self.dynamic_support = None

        if config.precision == 'fp16':
            self.model = self.model.half()

    def load_support_data(self, data_path='support_data.pth'):
        '''
        Load support data for validation.
        '''
        data_path = data_path.replace('.pth', f'_{self.config.img_size}.pth') # temporary fix for using various resolution
        # another temporary fix for old/new coco
        from dataset.coco import COCO as FOOBAR # avoid name conflict
        if FOOBAR.NEW_COCO:
            data_path = data_path.replace('.pth', '_new_coco.pth')

        if self.config.dataset != 'taskonomy':
            data_path = data_path.replace('.pth', f'_{self.config.dataset}.pth')

        if self.config.stage == 0:
            # generate support data if not exists.
            support_data = generate_support_data(self.config, data_path=data_path, verbose=self.verbose)
        else:
            task = f'{self.config.task}_{self.config.channel_idx}' if self.config.task == 'segment_semantic' else self.config.task
            support_data = {task: get_train_loader(self.config, verbose=False, get_support_data=True)}

        if self.verbose:
            print('loaded support data')
        
        # convert to proper precision
        if self.config.precision == 'fp16':
            support_data = to_device(support_data, dtype=torch.half)
        elif self.config.precision == 'bf16':
            support_data = to_device(support_data, dtype=torch.bfloat16)
        
        return support_data

    def configure_optimizers(self):
        '''
        Prepare optimizer and lr scheduler.
        '''
        optimizer, self.lr_scheduler = get_optimizer(self.config, self.model)
        return optimizer
    
    def train_dataloader(self, verbose=True):
        '''
        Prepare training loader.
        '''
        if not getattr(self.config, 'no_train', False):
            if self.config.stage == 0:
                train_loader = DataLoader(self.train_data, batch_size=(self.config.global_batch_size // torch.cuda.device_count()),
                                        shuffle=False, pin_memory=True, persistent_workers=(self.config.num_workers > 0),
                                        drop_last=True, num_workers=self.config.num_workers)
            else:
                train_loader = get_train_loader(self.config, verbose=(self.verbose and verbose))

            return train_loader
    
    def val_dataloader(self, verbose=True):
        '''
        Prepare validation loaders.
        '''
        if (not self.config.no_eval):
            # use external data from validation split
            if self.config.stage != 1 or self.config.use_valid:
                val_loaders, loader_tag = get_validation_loaders(self.config, verbose=(self.verbose and verbose))
                self.valid_tasks = list(val_loaders.keys())
                self.valid_tag = loader_tag
                return list(val_loaders.values())
            
            # use second half of support data as validation query
            else:
                assert self.config.shot > 1
                class SubQueryDataset:
                    def __init__(self, data):
                        self.data = data
                        self.n_query = self.data[0].shape[2] // 2
                    
                    def __len__(self):
                        return self.n_query
                    
                    def __getitem__(self, idx):
                        return (self.data[0][0, 0, self.n_query+idx],
                                self.data[1][0, :, self.n_query+idx, 0],
                                self.data[2][0, :, self.n_query+idx, 0])
                    
                valid_task = list(self.support_data.keys())[0]
                dset = SubQueryDataset(self.support_data[valid_task][:3])
                self.valid_tasks = [valid_task]
                self.valid_tag = 'mtest_support'
                    
                return torch.utils.data.DataLoader(dset, shuffle=False, batch_size=len(dset))
    
    def test_dataloader(self, verbose=True):
        '''
        Prepare test loaders.
        '''
        test_loader = get_eval_loader(self.config, self.config.task, split=self.config.test_split,
                                      channel_idx=self.config.channel_idx, verbose=(self.verbose and verbose))
        
        return test_loader
        
    def forward(self, *args, **kwargs):
        '''
        Forward data to model.
        '''
        return self.model(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        '''
        A single training iteration.
        '''
        # forward model and compute loss.
        if self.config.stage == 0:
            loss_dict = compute_loss(self.model, batch, self.config, n_classes=self.n_classes, teachers=self.teachers)
            if self.config.learning_to_bias:
                self.model.ema_image_encoder.update()
                self.model.ema_label_encoder.update()
        elif self.config.stage == 3:
            loss_dict = compute_loss(self.model, batch, self.config, n_classes=self.n_classes, teachers=self.teachers)
        else:
            loss_dict = compute_loss(self.model, batch, self.config, n_classes=self.n_classes, dynamic_support=self.dynamic_support)

        # schedule learning rate.
        self.lr_scheduler.step(self.global_step + self.config.schedule_from)

        if self.config.stage == 0:
            tag = ''
        elif self.config.stage == 1:
            if self.config.dataset == 'taskonomy':
                if self.config.task == 'segment_semantic':
                    tag = f'_segment_semantic_{self.config.channel_idx}'
                else:
                    tag = f'_{self.config.task}'
            elif self.config.dataset in ['davis2016', 'davis2017', 'loveda', 'ap10k']:
                tag = f'_{self.config.dataset}_{self.config.task}_{self.config.class_name}'
            elif self.config.dataset in ['isic2018', 'duts', 'kittiflow', 'rain100l', 'rain100h', 'eigen']:
                tag = f'_{self.config.dataset}_{self.config.task}'
            else:
                raise NotImplementedError
        elif self.config.stage == 3:
            tag = f'_{self.config.dataset}_pseudo'
        
        loss = loss_dict.pop('loss')
        # log losses and learning rate.
        log_dict = {
            f'training/loss{tag}': loss.detach(),
            f'training/lr{tag}': self.lr_scheduler.lr,
            'step': self.global_step,
        }
        for key, value in loss_dict.items():
            if value is not None:
                log_dict[f'training/{key}{tag}'] = value

        self.log_dict(
            log_dict,
            logger=True,
            on_step=True,
            sync_dist=True,
        )

        return loss
    
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def inference(self, X, task, **kwargs):
        if self.config.dataset == 'unified':
            task_name = task[1]
        else:
            task_name = task

        if self.config.dataset == 'loveda':
            X = rearrange(X, 'N P C H W -> (N P) C H W') # P is number of patches per image
        elif self.config.task == 'flow':
            X, X2 = X # X2 is the second image

        if self.config.model == 'VTM':
            # support data
            X_S, Y_S, M_S, t_idx = to_device(self.support_data[task], X.device)
            if task == 'animalkp':
                X_S = X_S[:10]
                Y_S = Y_S[:10]
                M_S = M_S[:10]
            t_idx = t_idx.long()
            T = Y_S.size(1)

            if self.dynamic_support is not None:
                assert self.config.task != 'flow'
                X_S_dynamic, Y_S_dynamic, M_S_dynamic = self.dynamic_support
                X_S_dynamic, Y_S_dynamic, M_S_dynamic = crop_arrays(X_S_dynamic, Y_S_dynamic, M_S_dynamic,
                                                                    base_size=base_sizes[self.config.img_size],
                                                                    crop_size=self.config.img_size,
                                                                    random=False)
                X_S = torch.cat([X_S, rearrange(X_S_dynamic, 'N C H W -> 1 1 N C H W')], dim=2)
                Y_S = torch.cat([Y_S, rearrange(Y_S_dynamic, 'N T H W -> 1 T N 1 H W')], dim=2)
                M_S = torch.cat([M_S, rearrange(M_S_dynamic, 'N T H W -> 1 T N 1 H W')], dim=2)

            # five-crop query images to 224 x 224 and reshape for matching
            if X.shape[-2:] != (self.config.img_size, self.config.img_size):
                X_crop = repeat(self.crop(X), 'F B C H W -> 1 T (F B) C H W', T=T)
                if self.config.task == 'flow':
                    X2_crop = repeat(self.crop(X2), 'F B C H W -> 1 T (F B) C H W', T=T)
                    X_crop = (X_crop, X2_crop)
            else:
                X_crop = repeat(X, 'B C H W -> 1 T B C H W', T=T)
                if self.config.task == 'flow':
                    X2_crop = repeat(X2, 'B C H W -> 1 T B C H W', T=T)
                    X_crop = (X_crop, X2_crop)

            if self.config.task == 'depth' and self.config.dataset == 'eigen':
                M_S = (Y_S > 0)

            # predict labels on each crop
            Y_S_in = torch.where(M_S.bool(), Y_S, torch.ones_like(Y_S) * self.config.mask_value)

            if T > 5:
                assert self.config.task != 'flow'
                Y_pred_crop = []
                chunk_size = 2
                for X_S_, Y_S_in_, X_crop_, t_idx_ in zip(X_S.split(chunk_size, dim=1),
                                                          Y_S_in.split(chunk_size, dim=1),
                                                          X_crop.split(chunk_size, dim=1),
                                                          t_idx.split(chunk_size, dim=1)):
                    Y_pred_crop_ = self.model(X_S_, Y_S_in_, X_crop_, t_idx=t_idx_, **kwargs)
                    Y_pred_crop.append(Y_pred_crop_)
                Y_pred_crop = torch.cat(Y_pred_crop, dim=1)
            else:
                Y_pred_crop = self.model(X_S, Y_S_in, X_crop, t_idx=t_idx, **kwargs)

            if 'segment_semantic' not in task_name and task_name not in ['vos', 'pose_6d', 'ds', 'sod', 'semseg']:
                if self.config.spatial_softmax_loss:
                    H, W = Y_pred_crop.shape[-2:]
                    Y_pred_crop = rearrange(Y_pred_crop, '1 T N C H W -> 1 T N C (H W)')
                    Y_pred_crop = F.softmax(Y_pred_crop, dim=-1)
                    Y_pred_crop = rearrange(Y_pred_crop, '1 T N C (H W) -> 1 T N C H W', H=H, W=W)
                    Y_pred_crop = Y_pred_crop / (1e-18 + reduce(Y_pred_crop, '1 T N C H W -> 1 T N C 1 1', 'max'))
                else:
                    Y_pred_crop = Y_pred_crop.sigmoid()

            # remix the cropped predictions into a whole prediction
            if X.shape[-2:] != (self.config.img_size, self.config.img_size):
                Y_pred_crop = rearrange(Y_pred_crop, '1 T (F B) 1 H W -> F B T H W', F=5)
                if self.config.task == 'flow':
                    Y_pred = mix_fivecrop(Y_pred_crop, base_size=X.size(-1), crop_size=X_crop[0].size(-1))
                else:
                    Y_pred = mix_fivecrop(Y_pred_crop, base_size=X.size(-1), crop_size=X_crop.size(-1))
            else:
                Y_pred = rearrange(Y_pred_crop, '1 T B 1 H W -> B T H W')

        elif self.config.model == 'DPT':
            if X.shape[-2:] != (self.config.img_size, self.config.img_size):
                X_crop = rearrange(self.crop(X), 'F B C H W -> (F B) C H W')
            else:
                X_crop = X
            Y_pred_crop = self.model(X_crop)
            
            if X.shape[-2:] != (self.config.img_size, self.config.img_size):
                Y_pred_crop = rearrange(Y_pred_crop, '(F B) T H W -> F B T H W', F=5)
                if 'segment_semantic' not in task_name and task_name not in ['vos', 'pose_6d', 'ds']:
                    Y_pred_crop = Y_pred_crop.sigmoid()
                Y_pred = mix_fivecrop(Y_pred_crop, base_size=X.size(-1), crop_size=X_crop.size(-1))
            else:
                Y_pred_crop = Y_pred_crop.sigmoid()

        return Y_pred
    
    def register_evaluators(self):
        # register evaluators
        if self.config.stage == 0:
            if self.config.dataset == 'unified':
                evaluators = {}
                dset_names = set(dset_name for dset_name, _ in self.valid_tasks)
                for dset_name in dset_names:
                    if 'categorical' in test_dataset_dict[dset_name]:
                        Dataset = test_dataset_dict[dset_name]['base']
                        evaluators[dset_name] = AverageMeter(class_ids_interest=range(len(Dataset.CLASS_IDXS_VAL)),
                                                             semseg_classes=Dataset.CLASS_IDXS_VAL,
                                                             device=self.device)
                    else:
                        evaluators[dset_name] = None
                self.evaluator = evaluators
            else:
                self.evaluator = AverageMeter(range(len(test_dataset_dict['taskonomy']['base'].CLASS_IDXS)),
                                              semseg_classes=test_dataset_dict['taskonomy']['base'].CLASS_IDXS,
                                              device=self.device)
        else:
            if self.config.dataset in ['davis2017']:
                self.evaluator = AverageMeter(range(self.n_classes), device=self.device)
            elif self.config.dataset == 'duts':
                self.evaluator = None # We will use only MAE for validation, thus no need to initialize evaluator.
            elif self.config.dataset == 'loveda' and self.config.multichannel:
                self.evaluator = AverageMeter(range(self.n_classes), device=self.device)
            else:
                self.evaluator = AverageMeter(0, device=self.device)

    def on_validation_start(self):
        self.register_evaluators()
        return super().on_validation_start()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Evaluate few-shot performance on validation dataset.
        '''
        if self.config.dataset == 'unified':
            task = self.valid_tasks[dataloader_idx]
            dset_name, task_name = task
        else:
            task = self.valid_tasks[dataloader_idx]
            dset_name = None
            task_name = task
        
        # query data
        if len(batch) == 3:
            X, Y, M = batch # Normal use case
            aux = None
        elif len(batch) == 4:
            X, Y, M, aux = batch # when additional informaion is needed.
        else:
            raise ValueError(f'Invalid batch: {len(batch)}')

        # support data
        Y_pred = self.inference(X.float(), task) # B T H W
        # Y_pred = self.inference(X, task) # B T H W

        # discretization for semantic segmentation
        if 'segment_semantic' in task_name or task_name in ['vos', 'ds', 'sod']:
            Y_pred = (Y_pred.sigmoid() > self.config.semseg_threshold).float()
        elif task_name == 'semseg':
            if self.config.crop_not_resize:

                Y_pred = rearrange(Y_pred, '(B L) t h w -> B (t h w) L', L=9)
                Y_pred = simple_mix_overlapped_crop(Y_pred) # B C H W, H=W=512
                # interp to Y size
                Y_pred = F.interpolate(Y_pred, size=Y.shape[-2:], mode='nearest')
                # also mix X for visualization (this is not gt but resized gt)
                X = rearrange(X, 'B L t h w -> B (t h w) L', L=9)
                X = simple_mix_overlapped_crop(X.float()).to(X.dtype) # B C H W, H=W=512
                M = rearrange(M, 'B L t h w -> B (t h w) L', L=9)
                M = simple_mix_overlapped_crop(M) # B C H W, H=W=512

            if self.config.multichannel:
                Y_pred = torch.argmax(Y_pred, dim=1)
                Y = torch.argmax(Y, dim=1)
            else:
                Y_pred = (Y_pred.sigmoid() > self.config.semseg_threshold).float()
        elif task == 'mvos':
            Y = convert_to_multiclass(Y, self.n_classes)
            Y_pred = convert_to_multiclass(Y_pred, self.n_classes)
            M = M[:, 0]
        elif task == 'flow':
            X, X2 = X

        # compute evaluation metric
        if self.config.dataset == 'unified':
            metric = compute_metric(Y, Y_pred, M, task_name, self.config.dataset, self.evaluator[dset_name], self.config.stage, self.n_classes)
        else:
            metric = compute_metric(Y, Y_pred, M, task_name, self.config.dataset, self.evaluator, self.config.stage, self.n_classes, crop_not_resize=self.config.crop_not_resize, X=X, aux=aux)
        metric *= len(Y)
        
        # visualize first batch
        if batch_idx == 0:
            X_vis = rearrange(self.all_gather(X), 'G B ... -> (B G) ...')
            Y_vis = rearrange(self.all_gather(Y), 'G B ... -> (B G) ...')
            M_vis = rearrange(self.all_gather(M), 'G B ... -> (B G) ...')
            Y_pred_vis = rearrange(self.all_gather(Y_pred), 'G B ... -> (B G) ...')
            if self.config.dataset == 'eigen':
                X_vis = rearrange(self.all_gather(aux[0]), 'G B ... -> (B G) ...')
                Y_vis = rearrange(self.all_gather(aux[1]), 'G B ... -> (B G) ...')
                M_vis = rearrange(self.all_gather(aux[2]), 'G B ... -> (B G) ...')
                Y_pred_vis = F.interpolate(Y_pred_vis, size=X_vis.shape[-2:], mode='bilinear', align_corners=False)
                # Since eigen is big, we only visualize first 5 images.
                X_vis = X_vis[:5]
                Y_vis = Y_vis[:5]
                M_vis = M_vis[:5]
                Y_pred_vis = Y_pred_vis[:5]
            elif self.config.dataset == 'loveda':
                # again to one hot
                if self.config.multichannel:
                    Y_vis = F.one_hot(Y_vis, num_classes=self.n_classes).permute(0, 3, 1, 2)
                    Y_pred_vis = F.one_hot(Y_pred_vis, num_classes=self.n_classes).permute(0, 3, 1, 2)

                # resize to 512 since the image is large
                Y_pred_vis = F.interpolate(Y_pred_vis.float(), size=X_vis.shape[-2:], mode='nearest')
                Y_vis = F.interpolate(Y_vis.float(), size=X_vis.shape[-2:], mode='nearest')
                M_vis = F.interpolate(M_vis.float(), size=X_vis.shape[-2:], mode='nearest')
            elif task == 'flow':
                X2_vis = rearrange(self.all_gather(X2), 'G B ... -> (B G) ...')
                X_vis = torch.cat([X_vis, X2_vis])

            if self.config.dataset in ['rain100l', 'rain100h']:
                vis_batch_diff = (X_vis, Y_vis, M_vis, Y_pred_vis)
                self.vis_images(vis_batch_diff, task_name+'_diff', dset_name=dset_name)
                vis_batch_derain = (X_vis, X_vis - Y_vis, M_vis, X_vis - Y_pred_vis)
                self.vis_images(vis_batch_derain, task_name, dset_name=dset_name)
            else:
                vis_batch = (X_vis, Y_vis, M_vis, Y_pred_vis)
                self.vis_images(vis_batch, task_name, dset_name=dset_name)

        if self.config.dynamic_support:
            if self.current_epoch > 0 and (self.current_epoch % self.config.dynamic_support_interval == 0):
                add_idx = self.current_epoch // self.config.dynamic_support_interval
                target_batch_idx = add_idx // self.config.eval_batch_size
                if batch_idx == target_batch_idx:
                    target_idx = add_idx % self.config.eval_batch_size
                    X_support_add = X[target_idx][None]
                    if task == 'mvos':
                        Y_pred_support_add = Y_pred[target_idx][None, None] / self.n_classes
                        M_pred_support_add = M[target_idx][None, None]
                    else:
                        Y_pred_support_add = Y_pred[target_idx][None]
                        M_pred_support_add = M[target_idx][None]
                    if self.dynamic_support is None:
                        self.dynamic_support = (X_support_add, Y_pred_support_add, M_pred_support_add)
                    else:
                        X_support, Y_pred_support, M_support = self.dynamic_support
                        self.dynamic_support = (torch.cat([X_support, X_support_add]),
                                                torch.cat([Y_pred_support, Y_pred_support_add]),
                                                torch.cat([M_support, M_pred_support_add]))

        return metric, torch.tensor(len(X), device=self.device)
    
    def on_test_start(self) -> None:
        if self.config.stage == 0:
            self.evaluator = AverageMeter(range(len(test_dataset_dict['taskonomy']['base'].CLASS_IDXS)),
                                          semseg_classes=test_dataset_dict['taskonomy']['base'].CLASS_IDXS,
                                          device=self.device)
        else:
            if self.config.dataset == 'duts':
                self.evaluator = DUTS_Evaluator(device=self.device)
            elif self.config.dataset == 'loveda':
                self.evaluator = None
                self.loveda_base_log_dir = os.path.join(self.config.result_dir, f'{self.config.dataset}_{self.config.task}_results_shot:{self.config.shot}{self.config.result_postfix}')
                self.loveda_ret_dir = os.path.join(self.loveda_base_log_dir, 'ret')
                os.makedirs(self.loveda_ret_dir, exist_ok=True)
                if not self.config.multichannel:
                    self.loveda_result_dict = {} # {imgname:{patch_num:prediction}}
                    self.semseg_classes = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural']
                    self.class_id = self.semseg_classes.index(self.config.class_name) + 1
                    self.loveda_logit_dir = os.path.join(self.loveda_base_log_dir, 'logits', str(self.class_id - 1)) # should change this to str(self.class_id)
                    os.makedirs(self.loveda_logit_dir, exist_ok=True)
            elif self.config.dataset == 'ap10k':
                self.ap10k_classes = ['left_eye', 'right_eye', 'nose', 'neck', 'root_of_tail', 'left_shoulder',
                                      'left_elbow', 'left_front_paw', 'right_shoulder', 'right_elbow', 'right_front_paw',
                                      'left_hip', 'left_knee', 'left_back_paw', 'right_hip', 'right_knee', 'right_back_paw']
                self.ap10k_base_dir = os.path.join(
                    self.config.result_dir, f'{self.config.dataset}_{self.config.task}_results_shot:{self.config.shot}{self.config.result_postfix}')
                self.ap10k_save_path = os.path.join(self.ap10k_base_dir, 'pred', f'{self.config.class_name}.pth')
                self.ap10k_res_dict = {}
                root = self.config.path_dict[self.config.dataset]
                self.ann_path = os.path.join(root, 'annotations', f'ap10k-test-split1.json')
                # self.ann_path = os.path.join(root, 'annotations', f'ap10k-val-split1.json')
                os.makedirs(os.path.dirname(self.ap10k_save_path), exist_ok=True)
            else:
                self.evaluator = AverageMeter(0, device=self.device)

            if getattr(self.config, 'dynamic_support', False):
                support_path = os.path.join('experiments', self.config.save_dir, self.config.exp_name, self.config.exp_subname,
                                            'checkpoints'', dynamic_support.pth')
                if os.path.exists(support_path):
                    self.dynamic_support = torch.load(support_path)
                    print(f'Loaded dynamic support from {support_path}')

        return super().on_test_start()
        
    def validation_epoch_end(self, validation_step_outputs):
        '''
        Aggregate losses of all validation datasets and log them into tensorboard.
        '''
        if len(self.valid_tasks) == 1:
            validation_step_outputs = (validation_step_outputs,)
        avg_loss = []
        log_dict = {'step': self.global_step}

        for task, losses_batch in zip(self.valid_tasks, validation_step_outputs):
            N_total = sum([losses[1] for losses in losses_batch])
            loss_pred = sum([losses[0] for losses in losses_batch])
            N_total = self.all_gather(N_total).sum()
            loss_pred = self.all_gather(loss_pred).sum(dim=0)

            loss_pred = loss_pred / N_total

            # log task-specific errors
            if self.config.dataset == 'taskonomy':
                if 'segment_semantic' in task:
                    if self.config.stage > 0 or test_dataset_dict['taskonomy']['base'].TASKS_SEMSEG.index(task) == 0:
                        self.evaluator.intersection_buf = reduce(self.all_gather(self.evaluator.intersection_buf),
                                                                        'G ... -> ...', 'sum')
                        self.evaluator.union_buf = reduce(self.all_gather(self.evaluator.union_buf),
                                                                'G ... -> ...', 'sum')

                        loss_pred = 1 - self.evaluator.compute_iou()[0]

                        if self.config.stage == 0:
                            tag = f'{self.valid_tag}/segment_semantic_pred'
                        else:
                            tag = f'{self.valid_tag}/segment_semantic_{self.config.channel_idx}_pred'

                        log_dict[tag] = loss_pred
                        avg_loss.append(loss_pred)
                else:
                    log_dict[f'{self.valid_tag}/{task}_pred'] = loss_pred
                    avg_loss.append(loss_pred)

            elif self.config.dataset in ['davis2016', 'davis2017', 'loveda']:
                self.evaluator.intersection_buf = reduce(self.all_gather(self.evaluator.intersection_buf),
                                                                'G ... -> ...', 'sum')
                self.evaluator.union_buf = reduce(self.all_gather(self.evaluator.union_buf),
                                                        'G ... -> ...', 'sum')

                loss_pred = 1 - self.evaluator.compute_iou()[0]
                log_dict[f'{self.valid_tag}/{self.config.dataset}_{task}_{self.config.class_name}_pred'] = loss_pred
                avg_loss.append(loss_pred)

            elif self.config.dataset == 'isic2018':
                self.evaluator.intersection_buf = reduce(self.all_gather(self.evaluator.intersection_buf),
                                                                'G ... -> ...', 'sum')
                self.evaluator.union_buf = reduce(self.all_gather(self.evaluator.union_buf),
                                                        'G ... -> ...', 'sum')
                intersection = self.evaluator.intersection_buf.float()
                union = self.evaluator.union_buf.float()

                dsc = 2*intersection / torch.max(torch.stack([union + intersection, self.evaluator.ones]), dim=0)[0]
                loss_pred = 1 - dsc[1, 0]

                log_dict[f'{self.valid_tag}/{self.config.dataset}_{task}_pred'] = loss_pred
                avg_loss.append(loss_pred)

            elif self.config.dataset == 'ap10k':
                log_dict[f'{self.valid_tag}/{self.config.dataset}_{task}_{self.config.class_name}_pred'] = loss_pred
                avg_loss.append(loss_pred)

            elif self.config.dataset in ['rain100l', 'rain100h']:
                log_dict[f'{self.valid_tag}/{self.config.dataset}_{task}_pred'] = 1 - loss_pred / 100 # psnr
                avg_loss.append(loss_pred)

            elif self.config.dataset == 'eigen':
                # loss_pred: silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3
                # use Abs Rel.
                loss_pred = loss_pred[1]
                log_dict[f'{self.valid_tag}/{self.config.dataset}_{task}_pred'] = loss_pred # psnr
                avg_loss.append(loss_pred)

            elif self.config.dataset == 'unified':
                dset_name, task_name = task
                # semseg
                if 'segment_semantic' in task_name:
                    semseg_classes = test_dataset_dict[dset_name]['base'].TASKS_SEMSEG
                    if self.config.stage > 0 or semseg_classes.index(task_name) == 0:
                        self.evaluator[dset_name].intersection_buf = reduce(self.all_gather(self.evaluator[dset_name].intersection_buf),
                                                                        'G ... -> ...', 'sum')
                        self.evaluator[dset_name].union_buf = reduce(self.all_gather(self.evaluator[dset_name].union_buf),
                                                                'G ... -> ...', 'sum')

                        loss_pred = 1 - self.evaluator[dset_name].compute_iou()[0]

                        if self.config.stage == 0:
                            tag = f'{self.valid_tag}/{dset_name}_segment_semantic_pred'
                        else:
                            tag = f'{self.valid_tag}/{dset_name}_segment_semantic_{self.config.channel_idx}_pred'

                        log_dict[tag] = loss_pred
                        avg_loss.append(loss_pred)
                # other
                else:
                    log_dict[f'{self.valid_tag}/{dset_name}_{task_name}_pred'] = loss_pred
                    avg_loss.append(loss_pred)

            else:
                log_dict[f'{self.valid_tag}/{self.config.dataset}_{task}_pred'] = loss_pred
                avg_loss.append(loss_pred)

        # log task-averaged error
        if self.config.stage == 0:
            avg_loss = sum(avg_loss) / len(avg_loss)
            log_dict[f'summary/{self.valid_tag}_pred'] = avg_loss

        self.log_dict(
            log_dict,
            logger=True,
            rank_zero_only=True
        )
        
        # reset miou evaluator
        if self.evaluator is not None:
            if self.config.dataset == 'unified':
                for evaluator in self.evaluator.values():
                    if evaluator is not None:
                        evaluator.reset()
            else:
                self.evaluator.reset()

    def test_step(self, batch, batch_idx):
        '''
        Evaluate few-shot performance on test dataset.
        '''
        if self.config.task == 'segment_semantic':
            task = f'segment_semantic_{self.config.channel_idx}'
        else:
            task = self.config.task

        # query data
        if len(batch) == 3:
            X, Y, M = batch # Normal use case
            aux = None
        elif len(batch) == 4: # duts, ap10k
            X, Y, M, aux = batch # when additional informaion is needed.
        else:
            raise ValueError(f'Invalid batch: {len(batch)}')

        # support data
        if X.shape[-2:] != base_sizes[self.config.img_size]:
            dtype = X.dtype
            cast = False
            if dtype != torch.float:
                X = X.float()
                cast = True
            X = F.interpolate(X, base_sizes[self.config.img_size], mode='bilinear', align_corners=False)
            if cast:
                X = X.to(dtype)

        Y_pred = self.inference(X, task)

        # discretization for semantic segmentation
        if 'segment_semantic' in task or task == 'ds':
            Y_pred = (Y_pred.sigmoid() > self.config.semseg_threshold).float()
        elif task in ['pose_6d', 'sod', 'semseg']:
            Y_pred = Y_pred.sigmoid()
        # No thresholding for sod here: we will threshold with several values in evaluator.
        # Also no thresholding for semseg here: we need to use logit for final prediction.

        # compute evaluation metric
        if self.config.dataset in ['taskonomy', 'isic2018', 'rain100l', 'rain100h', 'eigen']:
            metric = compute_metric(Y, Y_pred, M, task, self.config.dataset, self.evaluator, self.config.stage, X=X, aux=aux)
            metric *= len(X)

            return metric, len(X)
        
        # save prediction as images
        elif self.config.dataset in ['davis2016', 'davis2017']:
            save_root = os.path.join(self.config.result_dir,
                                     f'{self.config.dataset}_{self.config.task}_results_shot:{self.config.shot}{self.config.result_postfix}')
            os.makedirs(save_root, exist_ok=True)
            save_dir = os.path.join(save_root, self.config.class_name)
            os.makedirs(save_dir, exist_ok=True)

            if self.config.task == 'vos':
                # logits to probability
                if getattr(self.config, 'dense_crf', False):
                    Y_pred = Y_pred.sigmoid()
                    Y_pred = torch.where(Y_pred >= self.config.semseg_threshold,
                                         0.5 + 0.5 * (Y_pred - self.config.semseg_threshold) / (1 - self.config.semseg_threshold),
                                         0.5 * Y_pred / self.config.semseg_threshold)
                    Y_pred = torch.stack([dense_crf(X[i], Y_pred[i]) for i in range(len(X))])[:, None]
                else:
                    Y_pred = Y_pred.sigmoid()
            
                # upsampling
                Y_pred = F.interpolate(Y_pred, (480, 854), mode='bilinear', align_corners=False)

                # thresholding
                Y_pred = (Y_pred > self.config.semseg_threshold).float()

            elif task == 'mvos':
                if self.config.class_name == 'bike-packing':
                    target_size = (480, 910)
                elif self.config.class_name == 'shooting':
                    target_size = (480, 1152)
                else:
                    target_size = (480, 854)
                Y_pred = F.interpolate(Y_pred, target_size, mode='bilinear', align_corners=False)
                Y_pred = convert_to_multiclass(Y_pred, self.n_classes)

            for i in range(len(Y_pred)):
                if self.config.dataset == 'davis2016':
                    save_image(Y_pred[i], os.path.join(save_dir, f'{batch_idx*self.config.eval_batch_size+i:05d}.png'))
                else:
                    img = self.topil(Y_pred[i].cpu().byte())
                    img.save(os.path.join(save_dir, f'{batch_idx*self.config.eval_batch_size+i:05d}.png'))

        elif self.config.dataset == 'duts':
            # aux is paths
            Y_ori_paths = [path.replace('/common_datasets', self.config.path_dict['duts'].replace('/duts', ''))
                           for path in aux]
            metric = compute_metric(Y, Y_pred, M, task, self.config.dataset, self.evaluator, self.config.stage, Y_ori_paths=Y_ori_paths)
            metric *= len(X)

            return metric, len(X)
        elif self.config.dataset == 'loveda':
            if self.config.crop_not_resize:
                Y_pred = rearrange(Y_pred, '(B L) t h w -> B (t h w) L', L=9)
                Y_pred = simple_mix_overlapped_crop(Y_pred) # B C H W, H=W=512

            # we get save paths as y, so just save the prediction
            Y_pred = Y_pred.sigmoid().cpu()
            for i in range(len(Y)):
                if self.config.multichannel:
                    ret = F.interpolate(Y_pred[i].unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False).squeeze(0)
                    ret = ret.argmax(1).long()
                    img = self.topil(ret.cpu().to(torch.uint8))
                    img.save(os.path.join(self.loveda_ret_dir, Y[i]))
                else:
                    torch.save(Y_pred[i], os.path.join(self.loveda_logit_dir, Y[i]))
        elif self.config.dataset == 'ap10k':
            # save the prediction as a mode for each imgId.
            # imgIds are stored in aux.
            for i in range(len(aux)):
                imgId = aux[i].item()
                modes, scores = get_modes(Y_pred[i], return_scores=True, top_one=self.config.top_one)
                self.ap10k_res_dict[imgId] = (modes, scores)
        else:
            raise NotImplementedError
    
    def test_epoch_end(self, test_step_outputs):
        if self.config.dataset == 'taskonomy':
            # append test split to save_postfix
            log_name = f'result{self.config.save_postfix}_split:{self.config.test_split}{self.config.result_postfix}.pth'
            log_path = os.path.join(self.config.result_dir, log_name)
            
            if self.config.task == 'segment_semantic':
                # metric = self.evaluator.compute_iou()[0].cpu().item()
                torch.save(self.evaluator, log_path)
            else:
                N_total = sum([losses[1] for losses in test_step_outputs])
                metric = sum([losses[0] for losses in test_step_outputs])
                N_total = self.all_gather(N_total).sum()
                metric = self.all_gather(metric).sum(dim=0)
                metric = metric / N_total
                metric = metric.cpu().item()
                if self.local_rank == 0:
                    torch.save(metric, log_path)
        
        elif self.config.dataset == 'isic2018':
            log_path = os.path.join(self.config.result_dir, f'{self.config.dataset}_{self.config.task}_results_shot:{self.config.shot}{self.config.result_postfix}.pth')
            self.evaluator.intersection_buf = reduce(self.all_gather(self.evaluator.intersection_buf),
                                                                'G ... -> ...', 'sum')
            self.evaluator.union_buf = reduce(self.all_gather(self.evaluator.union_buf),
                                                        'G ... -> ...', 'sum')
            intersection = self.evaluator.intersection_buf.float()
            union = self.evaluator.union_buf.float()
            dsc = 2*intersection / torch.max(torch.stack([union + intersection, self.evaluator.ones]), dim=0)[0]
            metric = dsc[1, 0].item()

            if self.local_rank == 0:
                torch.save(metric, log_path)

        elif self.config.dataset in ['davis2016', 'davis2017']:
            pass

        elif self.config.dataset == 'duts':
            log_path = os.path.join(self.config.result_dir, f'{self.config.dataset}_{self.config.task}_results_shot:{self.config.shot}{self.config.result_postfix}.pth')
            N_total = sum([losses[1] for losses in test_step_outputs])
            metric = sum([losses[0] for losses in test_step_outputs])
            assert metric.size(0) == 4
            N_total = self.all_gather(N_total).sum()
            metric = self.all_gather(metric).sum(dim=0)
            metric /= N_total
            metric = metric.cpu()
            metric_dict = {'MAE': metric[0],'Max.F': metric[1],'Avg.F': metric[2],'S-measure': metric[3]}
            if self.local_rank == 0:
                torch.save(metric_dict, log_path)
        
        elif self.config.dataset == 'loveda':
            # gather the logits only when all the classes are finished
            if self.local_rank == 0 and not self.config.multichannel:
                

                # if self.config.multichannel:
                #     # zip images in self.loveda_ret_dir to a zip file
                #     img_list = os.listdir(self.loveda_ret_dir)
                #     img_list.sort()
                #     with ZipFile(os.path.join(self.loveda_base_log_dir, 'ret.zip'), 'w') as zipObj:
                #         for img in img_list:
                #             zipObj.write(os.path.join(self.loveda_ret_dir, img))
                #     print(f'Final prediction results were saved at {os.path.join(self.loveda_base_log_dir, "ret.zip")}')            
                # else:
                from glob import glob
                from tqdm import tqdm
                all_logits_paths = glob(os.path.join(self.loveda_base_log_dir, "logits", "*", "*.pth"))
                all_logits_paths.sort(key=lambda x: x.split("/")[-1])
                if len(all_logits_paths) == 7 * 1796:                    
                    assert (len(all_logits_paths) % 7) == 0, f'loveda has 7 classes but got total {len(all_logits_paths)} logits'
                    for i in tqdm(range(len(all_logits_paths)//7), desc='Gathering logits'):
                        *_, name = all_logits_paths[7*i].split("/")
                        name = name.split("_")[1]
                        logits = [torch.load(all_logits_paths[7*i])]
                        for j in range (1, 7):
                            assert name in all_logits_paths[7*i+j], 'Result logits are not correctly ordered.'
                            logits.append(torch.load(all_logits_paths[7*i+j]))
                        logits = torch.cat(logits, dim=0)
                        logits = F.interpolate(logits.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False).squeeze(0)
                        ret = torch.argmax(logits, dim=0).long()
                        img = self.topil(ret.cpu().to(torch.uint8))
                        img.save(os.path.join(self.loveda_ret_dir, name + ".png"))
                    print(f'Final prediction results were saved at {self.loveda_ret_dir}')            
                    parsed = self.loveda_base_log_dir.split("/")
                    exp_name = parsed[-4]
                    sname = parsed[-3]
                    # parsed[-2] == 'logs'
                    shot_info = parsed[-1]
                    if 'shot' in shot_info:
                        shot_info = shot_info[shot_info.rfind('shot'):]
                    else:
                        shot_info = ''
                    result_path = os.path.join(self.config.result_dir , '_'.join([exp_name, sname, shot_info, 'result.zip']))
                    img_list = os.listdir(self.loveda_ret_dir)
                    img_list.sort()
                    with ZipFile(result_path, 'w') as zipObj:
                        for img in img_list:
                            zipObj.write(os.path.join(self.loveda_ret_dir, img), arcname=img)
        
        elif self.config.dataset == 'ap10k':
            # save the prediction as {imgId: modes} where modes = List[List[tuple]]
            all_res_dicts = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_res_dicts, self.ap10k_res_dict) # this gathers all res dict to all_res_dicts
            ret_dict = {}
            for each in all_res_dicts:
                ret_dict.update(each)
            if self.local_rank == 0:
                torch.save(ret_dict, self.ap10k_save_path)
            if len(os.listdir(os.path.join(self.ap10k_base_dir, 'pred'))) == 17 and self.local_rank == 0:
                # get all result dict
                all_imgIds = list(ret_dict.keys())
                res_dicts = [torch.load(os.path.join(self.ap10k_base_dir, 'pred', f'{cls_name}.pth'))
                             for cls_name in self.ap10k_classes] # each res_dict is imgId: List[tuple(x,y)]]
                # aggregate results from res_dict
                result = {}
                for imgId in all_imgIds:
                    all_kp_modes = []
                    all_kp_scores = []
                    for res_dict in res_dicts:
                        modes, scores = res_dict[imgId] # modes: List[Tuple[int, int]], scores: List[float]
                        all_kp_modes.append(modes)
                        all_kp_scores.append(scores)
                    arr, score = modes_to_array(all_kp_modes, all_kp_scores, max_detection=1 if self.config.top_one else 20)
                    # arr: 17 x max_det x 3 | score: max_det | max_det defaults 20
                    result[imgId] = (arr, score)

                gt_coco = COCO(self.ann_path)
                ret_coco = ret_to_coco(result, gt_coco, base_sizes[self.config.img_size][0])
                # save coco result
                with open(os.path.join(self.ap10k_base_dir, 'ret_coco.json'), 'w') as f:
                    json.dump(ret_coco, f)
                # evaluate
                res_file = os.path.join(self.ap10k_base_dir, 'ret_coco.json')
                gt_coco.imgs = {k:v for k,v in gt_coco.imgs.items() if k in all_imgIds}
                coco_det = gt_coco.loadRes(res_file)
                sigmas = [
                    0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
                    0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089
                ]
                sigmas = np.array(sigmas)
                coco_eval = COCOeval(gt_coco, coco_det, 'keypoints', sigmas=sigmas)
                coco_eval.params.useSegm = None
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                stats_names = [
                    'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
                    'AR .75', 'AR (M)', 'AR (L)'
                ]
                info_str = list(zip(stats_names, coco_eval.stats))
                # save info_str
                torch.save(info_str, os.path.join(self.ap10k_base_dir, 'result.pth'))
        elif self.config.dataset in ['rain100l', 'rain100h']:
            log_path = os.path.join(self.config.result_dir, f'{self.config.dataset}_{self.config.task}_results_shot:{self.config.shot}{self.config.result_postfix}.pth')
            N_total = sum([losses[1] for losses in test_step_outputs])
            metric = torch.zeros(2, device=self.device)
            for losses in test_step_outputs:
                metric += losses[0]
            metric /= N_total
            metric = metric.cpu()
            metric_dict = {'PSNR': metric[0], 'SSIM': metric[1]}
            torch.save(metric_dict, log_path)
        elif self.config.dataset == 'eigen':
            log_path = os.path.join(self.config.result_dir, f'{self.config.dataset}_{self.config.task}_results_shot:{self.config.shot}{self.config.result_postfix}.pth')
            N_total = sum([losses[1] for losses in test_step_outputs])
            metric = sum([losses[0] for losses in test_step_outputs])
            N_total = self.all_gather(N_total).sum()
            metric = self.all_gather(metric).sum(dim=0)
            metric = metric.cpu() / N_total.cpu()

            name = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
            assert len(name) == len(metric)
            metric_dict = {k:v for k, v in zip(name, metric)}
            if self.local_rank == 0:
                torch.save(metric_dict, log_path)

        else:
            raise NotImplementedError
    
    @pl.utilities.rank_zero_only
    def vis_images(self, batch, task, dset_name=None, **kwargs):
        '''
        Visualize query prediction into tensorboard.
        '''
        X, Y, M, Y_pred = batch
        
        # set task-specific post-processing function for visualization
        vis_aux = None
        normalize = False
        mask_type = 'none'
        if task == 'depth_zbuffer':
            postprocess_fn = postprocess_depth
        elif 'segment_semantic' in task or task in ['vos', 'mvos', 'ds', 'sod']:
            postprocess_fn = postprocess_semseg
        elif task == 'semseg':
            postprocess_fn = postprocess_semseg
            if Y_pred.shape[1] > 1:
                vis_aux = []
                for c in range(1, Y_pred.shape[1]):
                    vis_aux.append(visualize_batch(X, Y[:,c:c+1,...], M[:,c:c+1,...], Y_pred[:,c:c+1,...], mask_type=mask_type, postprocess_fn=postprocess_fn))
                vis_aux = torch.cat(vis_aux, dim=0)
                Y_pred = Y_pred[:,0:1,...]
                Y = Y[:,0:1,...]
                M = M[:,0:1,...]

        elif task == 'flow':
            mask_type = 'halfs'
            postprocess_fn = postprocess_flow
            vis_aux = postprocess_flow(Y_pred.cpu())
        elif task == 'depth' and self.config.dataset == 'eigen':
            X = X.float()
            Y = Y.float()
            M = M.float()
            Y_pred = Y_pred.float()
            Y_pred = torch.where(Y_pred == 0, torch.ones_like(Y_pred), Y_pred)
            vis_aux = visualize_depth_plasma(Y_pred.cpu())

            Y = ((Y*256 + 1).log() - math.log(506)) / (math.log(23153) - math.log(506))
            Y = torch.where(M.bool(), Y, torch.ones_like(Y))
            Y_pred = torch.where(M.bool(), Y_pred, torch.ones_like(Y_pred))

            Y = visualize_depth_plasma(Y.cpu())
            Y_pred = visualize_depth_plasma(Y_pred.cpu())

            vis_aux = torch.from_numpy(vis_aux)
            Y = torch.from_numpy(Y)
            Y_pred = torch.from_numpy(Y_pred)

            postprocess_fn = None
            M = None

        elif task == 'keypoints_multiclass':
            postprocess_fn = postprocess_kp_multiclass
        # elif task == 'derain':
        #     Y_pred = X - Y_pred # visualize derained image
        #     Y = X - Y # visualize ground truth derained image
        #     postprocess_fn = None
        else:
            postprocess_fn = None
        
        if task in ['animalkp'] or any(map(task.startswith, ['keypoints_semantic', 'keypoints_human'])):
            normalize = True 
            
        # visualize batch
        vis = visualize_batch(X, Y, M, Y_pred, postprocess_fn=postprocess_fn, mask_type=mask_type, **kwargs)
        if vis_aux is not None:
            vis = torch.cat((vis, vis_aux))
        vis = make_grid(vis, nrow=len(Y), normalize=normalize, scale_each=True)

        if self.config.dataset == 'taskonomy':
            vis_tag = f'{self.valid_tag}/{task}'
        elif self.config.dataset in ['davis2016', 'davis2017', 'loveda', 'ap10k']:
            vis_tag = f'{self.valid_tag}/{self.config.dataset}_{task}_{self.config.class_name}'
        elif self.config.dataset == 'unified':
            if 'segment_semantic' in task:
                class_idx = int(task.split('_')[-1])
                class_name = test_dataset_dict[dset_name]['base'].CLASS_IDX_TO_NAME[class_idx]
                vis_tag = f'{self.valid_tag}/{dset_name}_segment_semantic_{class_name}'
            elif 'keypoints_semantic' in task:
                class_idx = int(task.split('_')[-1])
                class_name = test_dataset_dict[dset_name]['base'].KP_IDX_TO_NAME[class_idx]
                vis_tag = f'{self.valid_tag}/{dset_name}_keypoints_semantic_{class_name}'
            else:
                vis_tag = f'{self.valid_tag}/{dset_name}_{task}'
        elif self.config.dataset in ['isic2018', 'duts', 'kittiflow', 'rain100l', 'rain100h', 'eigen']:
            vis_tag = f'{self.valid_tag}/{self.config.dataset}_{task}'
        else:
            raise NotImplementedError
        self.logger.experiment.add_image(vis_tag, vis, self.global_step)

    def on_fit_end(self):
        if self.config.stage == 1 and self.dynamic_support is not None:
            if self.local_rank == 0:
                torch.save(self.dynamic_support, os.path.join(self.config.ckpt_dir, 'dynamic_support.pth'))
                print(f'Saved dynamic support to {os.path.join(self.config.ckpt_dir, "dynamic_support.pth")}')
                
        return super().on_fit_end()
