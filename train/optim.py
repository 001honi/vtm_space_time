import torch
import math
from deepspeed.ops.adam import DeepSpeedCPUAdam


optim_dict = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'cpuadam': DeepSpeedCPUAdam,
}


def get_optimizer(config, model):
    learnable_params = []
    
    # train all parameters for episodic training
    if config.stage == 0:
        if config.train_only_time_param:
            learnable_params.append(({'params': model.time_parameters(), 'lr': config.lr}))
        elif config.train_only_label_decoder:
            learnable_params.append(({'params': model.label_decoder_parameters(), 'lr': config.lr}))
        else:
            learnable_params.append({'params': model.pretrained_parameters(), 'lr': config.lr_pretrained})
            learnable_params.append({'params': model.scratch_parameters(), 'lr': config.lr})

    # train only task-specific parameters for fine-tuning
    elif config.stage == 1:
        if config.learning_to_bias and config.l2b_freeze_bias:
            learnable_params.append({'params': model.t_idx, 'lr': config.lr})
        else:
            learnable_params.append({'params': model.bias_parameters(), 'lr': config.lr})
        if getattr(config, 'head_tuning', False):
            learnable_params.append({'params': model.head_parameters(), 'lr': config.lr})

    # train only domain-specific parameters for domain adaptation
    elif config.stage == 3:
        learnable_params.append({'params': model.additional_bias_parameters(), 'lr': config.lr})
    
    kwargs = {}
    if config.optimizer == 'sgd':
        kwargs['momentum'] = 0.9
    optimizer = optim_dict[config.optimizer](learnable_params, weight_decay=config.weight_decay, **kwargs)
    if config.lr_warmup >= 0:
        lr_warmup = config.lr_warmup
    else:
        assert config.lr_warmup_scale >= 0. and config.lr_warmup_scale <= 1.
        lr_warmup = int(config.lr_warmup_scale * config.n_schedule_steps)
    lr_scheduler = CustomLRScheduler(optimizer, config.lr_schedule, config.lr, config.n_schedule_steps, lr_warmup,
                                     from_iter=config.schedule_from, decay_degree=config.lr_decay_degree)
    
    return optimizer, lr_scheduler
  

class CustomLRScheduler(object):
    '''
    Custom learning rate scheduler for pytorch optimizer.
    Assumes 1 <= self.iter <= 1 + num_iters.
    '''
    
    def __init__(self, optimizer, mode, base_lr, num_iters, warmup_iters=1000,
                 from_iter=0, decay_degree=0.9, decay_steps=5000):
        self.optimizer = optimizer
        self.mode = mode
        self.base_lr = base_lr
        self.lr = base_lr
        self.iter = from_iter
        self.N = num_iters + 1 
        self.warmup_iters = warmup_iters
        self.decay_degree = decay_degree
        self.decay_steps = decay_steps
        
        self.lr_coefs = []
        for param_group in optimizer.param_groups:
            self.lr_coefs.append(param_group['lr'] / base_lr)

        if self.iter > 0:
            self.step(self.iter)

    def step(self, step=-1):
        # updatae current step
        if step >= 0:
            self.iter = step
        else:
            self.iter += 1

        # schedule lr
        if self.mode == 'cos':
            self.lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * self.iter / self.N * math.pi))
        elif self.mode == 'poly':
            if self.iter < self.N:
                self.lr = self.base_lr * pow((1 - 1.0 * self.iter / self.N), self.decay_degree)
        elif self.mode == 'step':
            self.lr = self.base_lr * (0.1**(self.decay_steps // self.iter))
        elif self.mode == 'constant':
            self.lr = self.base_lr
        elif self.mode == 'sqroot':
            self.lr = self.base_lr * self.warmup_iters**0.5 * min(self.iter * self.warmup_iters**-1.5, self.iter**-0.5)
        else:
            raise NotImplementedError

        # warm up lr schedule
        if self.warmup_iters > 0 and self.iter < self.warmup_iters and self.mode != 'sqroot':
            self.lr = self.base_lr * 1.0 * self.iter / self.warmup_iters
        assert self.lr >= 0

        # adjust lr
        self._adjust_learning_rate(self.optimizer, self.lr)

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * self.lr_coefs[i]

    def reset(self):
        self.lr = self.base_lr
        self.iter = 0
        self._adjust_learning_rate(self.optimizer, self.lr)
