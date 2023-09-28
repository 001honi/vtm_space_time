import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO

import os
import sys
import shutil
import random
import tqdm
import math

import numpy as np
from scipy import interpolate
import torch
import torch.nn.functional as F

from .trainer import LightningTrainWrapper
from dataset.unified_dataset import Unified


def configure_experiment(config, model):
    # set seeds
    set_seeds(config.seed, config.debug_mode)
    
    # set directories
    log_dir, save_dir = set_directories(config,
                                        exp_name=config.exp_name,
                                        exp_subname=(config.exp_subname if config.stage >= 1 else ''),
                                        create_save_dir=(config.stage != 2))

    # create lightning callbacks, logger, and checkpoint plugin
    if config.stage != 2:
        callbacks = set_callbacks(config, save_dir, config.monitor, ptf=config.save_postfix)
        logger = CustomTBLogger(log_dir, name='', version='', default_hp_metric=False)
    else:
        callbacks = set_callbacks(config, save_dir)
        logger = None
    
    # create profiler
    profiler = pl.profilers.PyTorchProfiler(log_dir) if config.profile_mode else None
        
    # parse precision
    precision = int(config.precision.strip('fp')) if config.precision in ['fp16', 'fp32'] else config.precision
        
    # choose accelerator
    strategy = set_strategy(config, precision)

    # choose plugins
    if config.stage == 1 and config.strategy == 'ddp':
        if config.l2b_freeze_bias:
            save_names = ['model.t_idx']
        else:
            save_names = [f'model.{name}' for name in model.model.bias_parameter_names()]
        plugins = [CustomCheckpointIO(save_names)]
    elif config.stage == 3 and config.strategy == 'ddp':
        save_names = [f'model.{name}' for name in model.model.additional_bias_parameter_names()]
        plugins = [CustomCheckpointIO(save_names)]
    else:
        plugins = None
    
    return logger, log_dir, save_dir, callbacks, profiler, precision, strategy, plugins


def set_seeds(seed, debug_mode=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if debug_mode:
        torch.use_deterministic_algorithms(True)


def set_directories(config, root_dir='experiments', exp_name='', log_dir='logs', save_dir='checkpoints',
                    create_log_dir=True, create_save_dir=True, dir_postfix='', exp_subname=''):
    # make an experiment name
    if exp_name == '':
        if config.task == '':
            exp_name = config.exp_name = f'{config.model}_fold:{config.task_fold}{config.name_postfix}'
        else:
            exp_name = config.exp_name = f'{config.model}_task:{config.task}{config.name_postfix}'
    
    # create the root directory
    os.makedirs(root_dir, exist_ok=True)

    # set logging directory
    if create_log_dir:
        os.makedirs(os.path.join(root_dir, config.log_dir), exist_ok=True)
        if config.resolution_finetune_mode:
            dir_postfix = f'_to{config.img_size}'
        log_root = os.path.join(root_dir, config.log_dir, exp_name + dir_postfix)
        os.makedirs(log_root, exist_ok=True)
        if exp_subname != '':
            log_root = os.path.join(log_root, exp_subname)
            os.makedirs(log_root, exist_ok=True)
        log_dir = os.path.join(log_root, log_dir)

        # reset the logging directory if exists
        if config.stage == 0 and os.path.exists(log_dir) and not (config.continue_mode or config.skip_mode):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = None

    # set saving directory
    if create_save_dir:
        save_root = os.path.join(root_dir, config.save_dir, exp_name + dir_postfix)
        if exp_subname != '':
            save_root = os.path.join(save_root, exp_subname)
        save_dir = os.path.join(save_root, save_dir)

        # create the saving directory if checkpoint doesn't exist or in skipping mode,
        # otherwise ask user to reset it
        if config.stage == 0 and os.path.exists(save_dir) and int(os.environ.get('LOCAL_RANK', 0)) == 0:
            if config.continue_mode:
                print(f'resume from checkpoint ({exp_name})')
            elif config.skip_mode:
                print(f'skip the existing checkpoint ({exp_name})')
                sys.exit()
            elif config.debug_mode or config.reset_mode:
                print(f'remove existing checkpoint ({exp_name})')
                shutil.rmtree(save_dir)
            else:
                while True:
                    print(f'redundant experiment name! ({exp_name}) remove existing checkpoints? (y/n)')
                    inp = input()
                    if inp == 'y':
                        shutil.rmtree(save_dir)
                        break
                    elif inp == 'n':
                        print('quit')
                        sys.exit()
                    else:
                        print('invalid input')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    return log_dir, save_dir


def set_strategy(config, precision):
    if config.strategy == 'ddp':
        strategy = pl.strategies.DDPStrategy()
    elif config.strategy == 'deepspeed':
        strategy = pl.strategies.DeepSpeedStrategy(offload_optimizer=(config.optimizer == 'cpuadam'),
                                                   precision_plugin=pl.plugins.precision.DeepSpeedPrecisionPlugin(precision))
    else:
        strategy = None
        
    return strategy


def set_callbacks(config, save_dir, monitor=None, ptf=''):
    callbacks = [
        CustomProgressBar(),
    ]
    if ((not config.no_eval) and
        monitor is not None and
        config.early_stopping_patience > 0):
        callbacks.append(CustomEarlyStopping(monitor=monitor, mode="min", patience=config.early_stopping_patience))

    if not config.no_save and save_dir is not None:
        # step checkpointing
        if config.stage == 0:
            if not config.temporary_checkpointing:
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=save_dir,
                    filename='step:{step:06d}' + ptf,
                    auto_insert_metric_name=False,
                    every_n_epochs=5,
                    save_top_k=-1,
                    save_last=False,
                )
            else:
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=save_dir,
                    filename='step:{step:06d}' + ptf,
                    auto_insert_metric_name=False,
                    every_n_epochs=5,
                    save_top_k=2,
                    save_last=False,
                    monitor='step',
                    mode='max'
                )
            checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
            callbacks.append(checkpoint_callback)

        # last checkpointing
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            filename=f'last{ptf}',
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_top_k=1,
            save_last=False,
            monitor='epoch',
            mode='max',
        )
        checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
        callbacks.append(checkpoint_callback)
        
        # best checkpointing
        if not (config.no_eval or monitor is None):
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename=f'best{ptf}',
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_top_k=1,
                save_last=False,
                monitor=monitor,
                mode='min',
            )
            checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
            callbacks.append(checkpoint_callback)
            
    return callbacks


def get_ckpt_path(load_dir, exp_name, load_step, exp_subname='', save_postfix='', reduced=False):
    if load_step == 0:
        ckpt_name = f'best{save_postfix}.ckpt'
    elif load_step < 0:
        ckpt_name = f'last{save_postfix}.ckpt'
    else:
        ckpt_name = f'step:{load_step:06d}.ckpt'
    if reduced:
        ckpt_name = ckpt_name.replace('.ckpt', '.pth')
        
    load_path = os.path.join('experiments', load_dir, exp_name, exp_subname, 'checkpoints', ckpt_name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"checkpoint ({load_path}) does not exists!")
            
    return load_path


def update_legacy_config(config):
    # update legacy arguments
    if getattr(config, 'n_channel_interaction_blocks', None) is None:
        config.n_channel_interaction_blocks = 0
    if getattr(config, 'channel_interaction_type', None) is None:
        config.channel_interaction_type = 'global'
    if getattr(config, 'post_channel_interaction', None) is None:
        config.post_channel_interaction = False
    if getattr(config, 'interaction_drop', None) is None:
        config.interaction_drop = -1
    if getattr(config, 'decoder_features', None) is None:
        config.decoder_features = 96
    if getattr(config, 'deconv_head', None) is None:
        config.deconv_head = False
    if getattr(config, 'trivial_augmentation', None) is None:
        config.trivial_augmentation = False
    if getattr(config, 'order_mixup', None) is None:
        config.order_mixup = False
    if getattr(config, 'bitfit_init', None) is None:
        config.bitfit_init = 'default'
    if getattr(config, 'multimodal_softmax_loss', None) is None:
        config.multimodal_softmax_loss = False 
    if getattr(config, 'multimodal_mse_loss', None) is None:
        config.multimodal_mse_loss = False 
    if getattr(config, 'temp_msl', None) is None:
        config.temp_msl = 1.0 
    if getattr(config, 'n_levels', None) is None:
        config.n_levels = 4 
    if getattr(config, 'normalized_bce', None) is None:
        config.normalized_bce = False
    if getattr(config, 'crop_not_resize', None) is None:
        config.crop_not_resize = False
    if getattr(config, 'skip_crowd', None) is None:
        config.skip_crowd = False
    if getattr(config, 'spatial_softmax_loss', None) is None:
        config.spatial_softmax_loss = False
    if getattr(config, 'mse_loss', None) is None:
        config.mse_loss = False
    if getattr(config, 'dpt_seg_bce', None) is None:
        config.dpt_seg_bce = False
    if getattr(config, 'top_one', None) is None:
        config.top_one = False
    if getattr(config, 'bitfit', None) is None:
        config.bitfit = True
    if getattr(config, 'dynamic_support', None) is None:
        config.dynamic_support = False
    if getattr(config, 'learning_to_bias', None) is None:
        config.learning_to_bias = False
    if getattr(config, 'l2b_pre_projection', None) is None:
        config.l2b_pre_projection = False
    if getattr(config, 'l2b_freeze_bias', None) is None:
        config.l2b_freeze_bias = False
    if getattr(config, 'permute_classes', None) is None:
        config.permute_classes = False
    if getattr(config, 'task_sampling_weight', None) is None:
        config.task_sampling_weight = [1.0, 10.0, 20.0, 10.0]
    if getattr(config, 'additional_bias', None) is None:
        config.additional_bias = False
    if getattr(config, 'multichannel', None) is None:
        config.multichannel = False
    if getattr(config, 'channel_ce', None) is None:
        config.channel_ce = False


def copy_values(config_new, config_old):
    for key in config_new.__dir__():
        if key[:2] != '__':
            setattr(config_old, key, getattr(config_new, key))


def load_trained_ckpt(ckpt_path, config_new):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['state_dict']
    config = ckpt['config']
    if getattr(config, 'knowledge_distill', False):
        for key in list(state_dict.keys()):
            if 'teacher' in key:
                del state_dict[key]

    # merge config
    copy_values(config_new, config)

    return state_dict, config


def load_adapted_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['state_dict']

    return state_dict


def load_finetuned_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['state_dict']
    config = ckpt['hyper_parameters']['config']

    return state_dict, config


def resized_load_state_dict(model, state_dict, config, verbose=True):
    # l2b preprocessing
    if config.learning_to_bias:
        # delete ema and bias_selector parameters
        if config.l2b_freeze_bias:
            for key in list(state_dict.keys()):
                if key.split('.')[1][:3] == 'ema' or key.split('.')[1] == 'bias_selector':
                    del state_dict[key]
            if config.stage == 1:
                state_dict['model.t_idx'] = model.state_dict()['model.t_idx']
            
            bias_parameters = [f'model.{name}' for name in model.model.bias_parameter_names()]
            for key in bias_parameters:
                state_dict[key] = state_dict[key][:config.n_bias_sets]
        # copy ema parameters
        else:
            for key, value in model.state_dict().items():
                if key.split('.')[1][:3] == 'ema':
                    state_dict[key] = value

    # resize relative position bias table and pos embed
    for key in list(state_dict.keys()):
        if "relative_position_index" in key:
            state_dict[key] = model.state_dict()[key]

        if "relative_position_bias_table" in key:
            state_dict[key] = resize_rel_pos_bias(state_dict[key], model.state_dict()[key].size(0),
                                                    verbose=verbose, verbose_tag=key)

        if 'pos_embed' in key:
            state_dict[key] = resize_pos_embed(state_dict[key], model.state_dict()[key].size(1),
                                                verbose=verbose, verbose_tag=key)

    print(model.load_state_dict(state_dict))


def select_task_specific_parameters(config, model, state_dict):
    if getattr(config, 'bitfit', True):
        if config.channel_idx < 0:
            if config.task == 'pose_6d':
                n_tasks = 9
            elif config.task == 'flow':
                n_tasks = 2
            elif config.task == 'derain':
                n_tasks = 3
            elif config.task == 'semseg':
                n_tasks = 7
            elif config.task == 'animalkp':
                n_tasks = 17
            else:
                n_tasks = 1
        else:
            n_tasks = 1

        # for fine-tuning
        bias_parameters = [f'model.{name}' for name in model.model.bias_parameter_names()]
        for key in state_dict.keys():
            if key in bias_parameters:
                if config.learning_to_bias:
                    if config.l2b_freeze_bias:
                        state_dict[key] = state_dict[key][:config.n_bias_sets]
                    else:
                        pass
                else:
                    if getattr(config, 'average_real_bias_only', False):
                        state_dict[key] = state_dict[key][:len(Unified.TASKS)].mean(0, keepdim=True).repeat(n_tasks, 1)
                    else:
                        state_dict[key] = state_dict[key].mean(0, keepdim=True).repeat(n_tasks, 1)

    # remove unnecessary parameters
    for key in list(state_dict.keys()):
        if key.split('.')[1] == 'teacher':
            del state_dict[key]


def expand_state_dict(state_dict, model):
    additional_bias_parameters = [f'model.{name}' for name in model.model.additional_bias_parameter_names()] + \
                                 [f'model.{name}' for name in model.model.ema_additional_bias_parameter_names()]
    for key, value in model.state_dict().items():
        if key in additional_bias_parameters:
            state_dict[key] = value
        if key.split('.')[0] == 'teachers':
            state_dict[key] = value


def load_model(config, verbose=True, reduced=False):
    load_path = None

    # create trainer for episodic training
    if config.stage == 0:
        update_legacy_config(config)
        model = LightningTrainWrapper(config, verbose=verbose)
        if config.continue_mode or config.resolution_finetune_mode:
            load_path = get_ckpt_path(config.load_dir, config.exp_name, config.load_step,
                                      save_postfix=config.save_postfix, reduced=reduced)

            # for resolution fine-tuning
            if config.resolution_finetune_mode:
                state_dict, config = load_trained_ckpt(load_path, config)
                resized_load_state_dict(model, state_dict, config, verbose=verbose)
                if verbose:
                    print(f'meta-trained checkpoint loaded from {load_path}')

                load_path = None

    # create trainer for fine-tuning or evaluation
    else:
        # load meta-trained checkpoint
        ckpt_path = get_ckpt_path(config.load_dir, config.exp_name, config.load_step, reduced=reduced)
        state_dict, config = load_trained_ckpt(ckpt_path, config)
        
        update_legacy_config(config)
        if config.stage != 3:
            config.knowledge_distill = False # no pseudo task for fine-tuning or testing
        # image size adjustment
        if config.img_size != 224:
            config.image_encoder = config.image_encoder.replace('224', str(config.img_size))
            config.label_encoder = config.label_encoder.replace('224', str(config.img_size))
        model = LightningTrainWrapper(config=config, verbose=verbose)

        if config.stage == 3:
            expand_state_dict(state_dict, model)
            print(model.load_state_dict(state_dict))
        else:
            # laod adapted checkpoint
            if config.additional_bias:
                da_ckpt_path = get_ckpt_path(config.load_dir_aux, config.exp_name, -1, config.exp_subname_aux, f'_{config.dataset}')
                da_state_dict = load_adapted_ckpt(da_ckpt_path)
                for key in da_state_dict:
                    state_dict[key] = da_state_dict[key]

            # select task-specific parameters for test task
            if config.stage == 1:
                select_task_specific_parameters(config, model, state_dict)

            # load fine-tuned checkpoint
            else:
                ft_ckpt_path = get_ckpt_path(config.save_dir, config.exp_name, 0, config.exp_subname, config.save_postfix)
                ft_state_dict, _ = load_finetuned_ckpt(ft_ckpt_path)
                for key in ft_state_dict:
                    state_dict[key] = ft_state_dict[key]
                        
            resized_load_state_dict(model, state_dict, config, verbose=verbose)

        if verbose:
            print(f'meta-trained checkpoint loaded from {ckpt_path}')
            if config.stage == 2:
                print(f'fine-tuned checkpoint loaded from {ft_ckpt_path}')

    return model, load_path

        
class CustomProgressBar(TQDMProgressBar):
    def __init__(self, rescale_validation_batches=1):
        super().__init__()
        self.rescale_validation_batches = rescale_validation_batches

    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        bar = tqdm.tqdm(
            desc="Training",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self):
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = tqdm.tqdm(
            desc="Validation",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar
    
    def init_test_tqdm(self):
        """Override this to customize the tqdm bar for testing."""
        bar = tqdm.tqdm(
            desc="Testing",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    
class CustomTBLogger(TensorBoardLogger):
    @pl.utilities.rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)

    
class CustomEarlyStopping(EarlyStopping):
    def _run_early_stopping_check(self, trainer):
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics
        if self.monitor not in logs:
            should_stop = False
            reason = None
        else:
            current = logs[self.monitor].squeeze()
            should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)


class CustomCheckpointIO(TorchCheckpointIO):
    def __init__(self, save_parameter_names):
        self.save_parameter_names = save_parameter_names
    
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        # store only task-specific parameters
        state_dict = checkpoint['state_dict']
        state_dict = {key: value for key, value in state_dict.items() if key in self.save_parameter_names}
        checkpoint['state_dict'] = state_dict
        
        super().save_checkpoint(checkpoint, path, storage_options)


def resize_rel_pos_bias(rel_pos_bias, dst_num_pos, verbose=True, verbose_tag=''):
    src_num_pos, num_attn_heads = rel_pos_bias.size()
    num_extra_tokens = 3

    src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
    dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
    if src_size != dst_size:
        if verbose:
            print("Position interpolate for %s from %dx%d to %dx%d" % (
            verbose_tag, src_size, src_size, dst_size, dst_size))
        extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
        rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

        def geometric_progression(a, r, n):
            return a * (1.0 - r ** n) / (1.0 - r)

        left, right = 1.01, 1.5
        while right - left > 1e-6:
            q = (left + right) / 2.0
            gp = geometric_progression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q

        dis = []
        cur = 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q ** (i + 1)

        r_ids = [-_ for _ in reversed(dis)]

        x = r_ids + [0] + dis
        y = r_ids + [0] + dis

        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)

        all_rel_pos_bias = []

        for i in range(num_attn_heads):
            z = rel_pos_bias[:, i].view(src_size, src_size).cpu().detach().float().numpy()
            f = interpolate.interp2d(x, y, z, kind='cubic')
            all_rel_pos_bias.append(
                torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

        rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

        new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
    else:
        new_rel_pos_bias = rel_pos_bias

    return new_rel_pos_bias


def resize_pos_embed(posemb, ntok_new, num_prefix_tokens=1, gs_new=(), verbose=True, verbose_tag=''):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    if posemb.size(1) == ntok_new:
        return posemb
    
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    if verbose:
        print('Position embedding %s grid-size from %s to %s' % (verbose_tag, [gs_old, gs_old], gs_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    cast = False
    if posemb_grid.dtype != torch.float32:
        dtype = posemb_grid.dtype
        cast = True
        posemb_grid = posemb_grid.float()
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    if cast:
        posemb_grid = posemb_grid.to(dtype)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb
