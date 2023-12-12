import torch
from torch.utils.data import DataLoader
import os
from einops import rearrange, repeat

from .unified_dataset import Unified, UnifiedTrainDataset, test_dataset_dict
from .benchmark_dataset import BenchmarkDataset
from .utils import crop_arrays
from .downstream import DownstreamFinetuneDataset, DAVIS2017FinetuneDataset, EigenFinetuneDataset, ISIC2018FinetuneDataset, \
    DUTSTestDataset, LoveDAFinetuneDataset, KittiFlowFinetuneDataset, KittiFlowFinetuneDataset, AP10KFinetuneDataset, RainFinetuneDataset,  \
    DownstreamUnsupervisedDataset

base_sizes = {
    128: (128, 128),  ### FOR FTUNE
    # 128: (256, 256), #MIDAIR
    224: (256, 256),
    384: (448, 448),
    416: (480, 480),
    512: (592, 592),
}


data_configs = {
    'davis2016': {
        'img_ext': '.jpg',
        'lbl_ext': '.png',
        'train_files': False,
        'sequential': True,
    },
    'davis2017': {
        'img_ext': '.jpg',
        'lbl_ext': '.png',
        'train_files': False,
        'sequential': True,
    },
    'isic2018': {
        'img_ext': '.jpg',
        'lbl_ext': '.png',
        'train_files': False,
        'sequential': False,
    },
    'duts': {
        'img_ext': '.jpg',
        'lbl_ext': '.png',
        'train_files': True,
        'sequential': False,
    },
    'loveda': {
        'img_ext': '.png',
        'lbl_ext': '.png',
        'train_files': False,
        'sequential': False,
    },
    'ap10k': {
        'img_ext': 'jpg',
        'lbl_ext': '',
        'train_files': False,
        'sequential': False,
    },
    'rain100l': {
        'img_ext': 'rain-',
        'lbl_ext': 'diff-',
        'train_files': True,
        'sequential': False,
    },
    'rain100h': {
        'img_ext': 'rain-',
        'lbl_ext': 'diff-',
        'train_files': True,
        'sequential': False,
    },
    'eigen': {
        'img_ext': '',
        'lbl_ext': '',
        'train_files': True,
        'sequential': False,
    },
}


def get_train_loader(config, pin_memory=True, verbose=True, get_support_data=False, return_dset=False):
    '''
    Load training dataloader.
    '''
    # set dataset size
    if get_support_data:
        dset_size = config.shot
    elif config.no_eval:
        dset_size = config.n_steps*config.global_batch_size
    else:
        dset_size = config.val_iter*config.global_batch_size

    # compute common arguments
    common_kwargs = {
        'base_size': base_sizes[config.img_size],
        'crop_size': (config.img_size, config.img_size),
        'seed': config.seed + int(os.environ.get('LOCAL_RANK', 0)),
        'precision': config.precision,
        'sample_by_seq': config.sample_by_seq,
        'sample_skip': config.sample_skip
    }

    # create dataset for episodic training
    if config.stage == 0:
        if config.benchmark_mode:
            train_data = BenchmarkDataset(shot=config.shot, dset_size=dset_size)
        else:
            train_data = UnifiedTrainDataset(
                config=config,
                shot=config.shot,
                dset_size=dset_size,
                verbose=verbose,
                unary_augmentation=config.unary_augmentation,
                binary_augmentation=config.binary_augmentation,
                path_dict=config.path_dict,
                split='train',
                **common_kwargs
            )
        return train_data
    
    # create dataset for domain adaptation
    elif config.stage == 3:
        train_data = DownstreamUnsupervisedDataset(
            dset_size=dset_size,
            shot=config.shot,
            base_size=base_sizes[config.img_size],
            crop_size=(config.img_size, config.img_size),
            dataset=config.dataset,
            data_root=config.path_dict[config.dataset],
            img_ext=data_configs[config.dataset]['img_ext'],
            lbl_ext=data_configs[config.dataset]['lbl_ext'],
            class_name=config.class_name,
            precision=config.precision,
        )

    # create dataset for fine-tuning
    else:
        if config.task in ['', 'all']:
            raise ValueError("task should be specified for fine-tuning")

        if config.dataset == 'kittiflow':
            train_data = KittiFlowFinetuneDataset(
                split='train',
                dset_size=dset_size,
                shot=config.shot,
                support_idx=config.support_idx,
                channel_idx=config.channel_idx,
                root_dir=config.path_dict[config.dataset],
                base_size=base_sizes[config.img_size],
                img_size=(config.img_size, config.img_size),
                precision=config.precision,
                fix_seed=False,
                sequential=True
            )
        else:
            specific_kwargs = {}
            if config.dataset == 'davis2017':
                FinetuneDataset = DAVIS2017FinetuneDataset
                specific_kwargs['permute_classes'] = config.permute_classes
            elif config.dataset == 'isic2018':
                FinetuneDataset = ISIC2018FinetuneDataset
            elif config.dataset == 'loveda':
                crop_not_resize = config.crop_not_resize
                FinetuneDataset = lambda **kwargs: LoveDAFinetuneDataset(multichannel= config.multichannel, crop_not_resize=crop_not_resize, **kwargs)
            elif config.dataset == 'ap10k':
                FinetuneDataset = lambda **kwargs: AP10KFinetuneDataset(skip_crowd=config.skip_crowd, **kwargs) 
            elif config.dataset in ['rain100l', 'rain100h']:
                FinetuneDataset = RainFinetuneDataset
            elif config.dataset == 'eigen':
                FinetuneDataset = EigenFinetuneDataset
            else:
                FinetuneDataset = DownstreamFinetuneDataset
            train_data = FinetuneDataset(
                split='train',
                dset_size=dset_size,
                shot=config.shot,
                support_idx=config.support_idx,
                channel_idx=config.channel_idx,
                base_size=base_sizes[config.img_size],
                img_size=(config.img_size, config.img_size),
                dataset=config.dataset,
                data_root=config.path_dict[config.dataset],
                img_ext=data_configs[config.dataset]['img_ext'],
                lbl_ext=data_configs[config.dataset]['lbl_ext'],
                class_name=config.class_name,
                train_files=data_configs[config.dataset]['train_files'],
                sequential=data_configs[config.dataset]['sequential'],
                precision=config.precision,
                fix_seed=False,
                **specific_kwargs,
            )

        if get_support_data:
            train_data.fix_seed = True
            support_loader = DataLoader(train_data, batch_size=config.shot, shuffle=False, drop_last=False, num_workers=0)
            for support_data in support_loader:
                break
            
            X, Y, M = support_data

            N = config.shot
            if config.dataset == 'loveda':
                N = N * X.size(1)
                X = rearrange(X, 'B P C H W -> (B P) C H W')
                Y = rearrange(Y, 'B P C H W -> (B P) C H W')
                M = rearrange(M, 'B P C H W -> (B P) C H W')
            # elif config.dataset == 'kittiflow':
            #     X, X2 = X

            assert X.ndim == 4
            assert Y.ndim == 4
            X = repeat(X, '(B N) C H W -> B T N C H W', N=N, T=Y.size(1))
            Y = rearrange(Y, '(B N) T H W -> B T N 1 H W', N=N)
            M = rearrange(M, '(B N) T H W -> B T N 1 H W', N=N)
            t_idx = repeat(torch.tensor(list(range(Y.size(1))), device=X.device), 'T -> B T', B=len(X))
            # if config.dataset == 'kittiflow':
            #     X2 = repeat(X2, '(B N) C H W -> B T N C H W', N=N, T=Y.size(1))
            #     X = (X, X2)
            support_data = X, Y, M, t_idx

            return support_data
        
    if return_dset:
        return train_data
  
    # create training loader.
    train_loader = DataLoader(train_data, batch_size=(config.global_batch_size // torch.cuda.device_count()),
                              shuffle=False, pin_memory=pin_memory, persistent_workers=True,
                              drop_last=True, num_workers=config.num_workers)
        
    return train_loader


def get_eval_loader(config, task, split='valid', channel_idx=-1, pin_memory=True, verbose=True):
    '''
    Load evaluation dataloader.
    '''
    # no crop for evaluation.
    crop_size = base_size = base_sizes[config.img_size]
            
    # evaluate some subset or the whole data.
    if config.n_eval_batches > 0:
        dset_size = config.n_eval_batches * config.eval_batch_size
    else:
        dset_size = -1

    if config.stage == 0:
        raise NotImplementedError
    else:
        if config.dataset == 'kittiflow':
            eval_data = KittiFlowFinetuneDataset(
                split=split,
                dset_size=dset_size,
                shot=config.shot,
                support_idx=config.support_idx,
                channel_idx=config.channel_idx,
                root_dir=config.path_dict[config.dataset],
                base_size=base_size,
                img_size=crop_size,
                precision=config.precision,
                fix_seed=True,
            )
        else:
            specific_kwargs = {}
            if config.dataset == 'davis2017':
                FinetuneDataset = DAVIS2017FinetuneDataset
                specific_kwargs['permute_classes'] = config.permute_classes
                specific_kwargs['n_vis_size'] = config.eval_batch_size
            elif config.dataset == 'isic2018':
                FinetuneDataset = ISIC2018FinetuneDataset
            elif config.dataset == 'duts' and split == 'test':
                FinetuneDataset = DUTSTestDataset
            elif config.dataset == 'loveda':
                FinetuneDataset = lambda **kwargs: LoveDAFinetuneDataset(multichannel=config.multichannel, **kwargs) 
            elif config.dataset == 'ap10k':
                FinetuneDataset = lambda **kwargs: AP10KFinetuneDataset(skip_crowd=config.skip_crowd, **kwargs) 
            elif config.dataset in ['rain100l', 'rain100h']:
                FinetuneDataset = RainFinetuneDataset
            elif config.dataset == 'eigen':
                FinetuneDataset = EigenFinetuneDataset
            else:
                FinetuneDataset = DownstreamFinetuneDataset
            eval_data = FinetuneDataset(
                split=split,
                dset_size=dset_size,
                shot=config.shot,
                support_idx=config.support_idx,
                channel_idx=config.channel_idx,
                base_size=base_size,
                img_size=crop_size,
                dataset=config.dataset,
                data_root=config.path_dict[config.dataset],
                img_ext=data_configs[config.dataset]['img_ext'],
                lbl_ext=data_configs[config.dataset]['lbl_ext'],
                class_name=config.class_name,
                train_files=data_configs[config.dataset]['train_files'],
                sequential=data_configs[config.dataset]['sequential'],
                precision=config.precision,
                fix_seed=True,
                **specific_kwargs,
            )

    # create dataloader.
    eval_loader = DataLoader(eval_data, batch_size=(config.eval_batch_size // torch.cuda.device_count()),
                             shuffle=False, pin_memory=pin_memory,
                             drop_last=False, num_workers=0)
    
    return eval_loader


def get_validation_loaders(config, verbose=True):
    '''
    Load validation loaders (of unseen images) for training tasks.
    '''
    if config.stage == 0:
        valid_loaders = {}
        loader_tag = 'mtrain_valid'

        # no crop for evaluation.
        crop_size = base_size = base_sizes[config.img_size]
            
        # evaluate some subset or the whole data.
        if config.n_eval_batches > 0:
            dset_size = config.n_eval_batches * config.eval_batch_size
        else:
            dset_size = -1

        for dataset_name, task_group in Unified.TASK_GROUP_NAMES:
            if getattr(config, dataset_name, False):
                if task_group in ['segment_semantic', 'keypoints_semantic']:
                    class_indices = test_dataset_dict[dataset_name]['base'].CLASS_IDXS_VAL
                    if task_group == 'keypoints_semantic':
                        assert dataset_name == 'coco'
                        class_indices = test_dataset_dict[dataset_name]['categorical'].KP_IDXS
                    for c in class_indices:
                        TestDataset = test_dataset_dict[dataset_name]['categorical']
                        eval_data = TestDataset(
                            class_id=c,
                            task_group=task_group,
                            dset_size=dset_size,
                            path_dict=config.path_dict,
                            split='valid',
                            base_size=base_size,
                            crop_size=crop_size,
                            precision=config.precision,
                        )
                        # create dataloader.
                        eval_loader = DataLoader(eval_data, batch_size=(config.eval_batch_size // torch.cuda.device_count()),
                                                shuffle=False, pin_memory=True, drop_last=False, num_workers=0)
                        valid_loaders[(dataset_name, f'{task_group}_{c}')] = eval_loader
                        if verbose:
                            print(f'Loaded {dataset_name} {task_group} {c} validation loader.')
                else:
                    task_type = Unified.TASK_GROUP_TYPE[(dataset_name, task_group)]
                    TestDataset = test_dataset_dict[dataset_name][task_type]
                    eval_data = TestDataset(
                        task_group=task_group,
                        dset_size=dset_size,
                        path_dict=config.path_dict,
                        split='valid',
                        base_size=base_size,
                        crop_size=crop_size,
                        precision=config.precision,
                    )
                    # create dataloader.
                    eval_loader = DataLoader(eval_data, batch_size=(config.eval_batch_size // torch.cuda.device_count()),
                                                shuffle=False, pin_memory=True, drop_last=False, num_workers=0)
                    valid_loaders[(dataset_name, task_group)] = eval_loader
                    if verbose:
                        print(f'Loaded {dataset_name} {task_group} validation loader.')
    elif config.stage == 1:
        if config.task in ['', 'all']:
            raise ValueError("task should be specified for fine-tuning")
        train_tasks = [config.task]
        loader_tag = 'mtest_valid'

        valid_loaders = {}
        for task in train_tasks:
            valid_loaders[task] = get_eval_loader(config, task, 'valid', verbose=verbose)

    else:
        raise NotImplementedError
    
    return valid_loaders, loader_tag


def disassemble_batch(batch, base_size=None, crop_size=None):
    X, Y, M = batch 
    T = Y.size(1)
    X = repeat(X, 'N C H W -> 1 T N C H W', T=T)
    Y = rearrange(Y, 'N T H W -> 1 T N 1 H W')
    M = rearrange(M, 'N T H W -> 1 T N 1 H W')
    if base_size is not None and crop_size is not None:
        X, Y, M = crop_arrays(X, Y, M, base_size=base_size, crop_size=crop_size, random=False)
    
    return X, Y, M


def generate_support_data(config, data_path, split='train', support_idx=0, verbose=True):
    '''
    Generate support data for all tasks.
    '''
    if os.path.exists(data_path):
        support_data = torch.load(data_path)
    else:
        support_data = {}
    
    modified = False
    base_size = crop_size = base_sizes[config.img_size]

    for dataset_name, task_group in Unified.TASK_GROUP_NAMES:
        if getattr(config, dataset_name, False):
            if task_group in ['segment_semantic', 'keypoints_semantic']:
                class_indices = test_dataset_dict[dataset_name]['base'].CLASS_IDXS_VAL
                if task_group == 'keypoints_semantic':
                    assert dataset_name == 'coco'
                    class_indices = test_dataset_dict[dataset_name]['categorical'].KP_IDXS
                for c in class_indices:
                    if (dataset_name, f'{task_group}_{c}') not in support_data:
                        TestDataset = test_dataset_dict[dataset_name]['categorical']
                        dset = TestDataset(
                            class_id=c,
                            task_group=task_group,
                            dset_size=config.shot*(support_idx + 1),
                            path_dict=config.path_dict,
                            split='valid',
                            base_size=base_size,
                            crop_size=crop_size,
                            precision=config.precision,
                        )
                        dloader = DataLoader(dset, batch_size=config.shot, shuffle=False, num_workers=0)
                        for idx, batch in enumerate(dloader):
                            if idx == support_idx:
                                break
                        
                        t_idx = torch.tensor([[Unified.TASKS.index((dataset_name, f'{task_group}_{c}'))]])
                        batch = (*disassemble_batch(batch, base_size, config.img_size), t_idx)
                        support_data[(dataset_name, f'{task_group}_{c}')] = batch
                        if verbose:
                            print(f'Generated {dataset_name} {task_group} {c} support data.')
                        modified = True
            else:
                if (dataset_name, task_group) not in support_data:
                    task_type = Unified.TASK_GROUP_TYPE[(dataset_name, task_group)]
                    TestDataset = test_dataset_dict[dataset_name][task_type]
                    dset = TestDataset(
                        task_group=task_group,
                        dset_size=config.shot*(support_idx + 1),
                        path_dict=config.path_dict,
                        split='valid',
                        base_size=base_size,
                        crop_size=crop_size,
                        precision=config.precision,
                    )
                    dloader = DataLoader(dset, batch_size=config.shot, shuffle=False, num_workers=0)
                    for idx, batch in enumerate(dloader):
                        if idx == support_idx:
                            break
                    
                    t_idx = torch.tensor([[Unified.TASKS.index((dataset_name, f'{task_group}_{c}'))
                                        for c in range(len(TestDataset.TASK_GROUP_DICT[task_group]))]])
                    batch = (*disassemble_batch(batch, base_size, config.img_size), t_idx)
                    support_data[(dataset_name, task_group)] = batch
                    if verbose:
                        print(f'Generated {dataset_name} {task_group} support data.')
                    modified = True

    if modified:
        torch.save(support_data, data_path)
            
    return support_data
