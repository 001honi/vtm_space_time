import torch
from torch.utils.data import Dataset
from .taskonomy import *
from .coco import *
from .midair import *
from .openimages import *
from .unlabeled import *


train_dataset_dict = {
    'taskonomy': {
        'base': TaskonomyBaseTrainDataset,
        'continuous': TaskonomyContinuousTrainDataset,
        'categorical': TaskonomyCategoricalTrainDataset,
        'unsupervised': TaskonomyUnsupervisedTrainDataset,
    },
    'coco': {
        'base': COCOBaseTrainDataset,
        'continuous': COCOContinuousTrainDataset,
        'categorical': COCOCategoricalTrainDataset,
        'unsupervised': COCOUnsupervisedTrainDataset,
    },
    'midair': {
        'base': MidAirBaseTrainDataset,
        'continuous': MidAirContinuousTrainDataset,
        'categorical': MidAirCategoricalTrainDataset,
        'unsupervised': MidAirUnsupervisedTrainDataset,
    },
    'openimages': {
        'base': OpenImagesBaseTrainDataset,
        'unsupervised': OpenImagesUnsupervisedTrainDataset,
    },
    'unlabeled': {
        'base': UnlabeledBaseTrainDataset,
        'unsupervised': UnlabeledUnsupervisedTrainDataset,
    },
}


test_dataset_dict = {
    'taskonomy': {
        'base': TaskonomyBaseTestDataset,
        'continuous': TaskonomyBaseTestDataset,
        'categorical': TaskonomyCategoricalTestDataset,
        'unsupervised': TaskonomyUnsupervisedTestDataset,
    },
    'coco': {
        'base': COCOBaseTestDataset,
        'continuous': COCOBaseTestDataset,
        'categorical': COCOCategoricalTestDataset,
        'unsupervised': COCOUnsupervisedTestDataset,
    },
    'midair': {
        'base': MidAirBaseTestDataset,
        'continuous': MidAirBaseTestDataset,
        'categorical': MidAirCategoricalTestDataset,
        'unsupervised': MidAirUnsupervisedTestDataset,
    },
    'openimages': {
        'base': OpenImagesBaseTestDataset,
        'unsupervised': OpenImagesUnsupervisedTestDataset,
    },
    'unlabeled': {
        'base': UnlabeledBaseTestDataset,
        'unsupervised': UnlabeledUnsupervisedTestDataset,
    },
}


class Unified(Dataset):
    NAME = 'unified'

    # Tasks
    TASKS_BASE = []
    TASKS_CONTINUOUS = []
    TASKS_CATEGORICAL = []

    # Task Groups
    TASK_GROUP_NAMES_BASE = []
    TASK_GROUP_NAMES_CONTINUOUS = []
    TASK_GROUP_NAMES_CATEGORICAL = []

    TASK_GROUP_TYPE = {}

    for dataset_name, dataset_dict in train_dataset_dict.items():
        for task_group_name in dataset_dict['base'].TASK_GROUP_NAMES_BASE:
            TASK_GROUP_NAMES_BASE.append((dataset_name, task_group_name))
            TASK_GROUP_TYPE[(dataset_name, task_group_name)] = 'base'
            for task in dataset_dict['base'].TASK_GROUP_DICT[task_group_name]:
                TASKS_BASE.append((dataset_name, task))

        for task_group_name in dataset_dict['base'].TASK_GROUP_NAMES_CONTINUOUS:
            TASK_GROUP_NAMES_CONTINUOUS.append((dataset_name, task_group_name))
            TASK_GROUP_TYPE[(dataset_name, task_group_name)] = 'continuous'
            for task in dataset_dict['base'].TASK_GROUP_DICT[task_group_name]:
                TASKS_CONTINUOUS.append((dataset_name, task))

        for task_group_name in dataset_dict['base'].TASK_GROUP_NAMES_CATEGORICAL:
            TASK_GROUP_NAMES_CATEGORICAL.append((dataset_name, task_group_name))
            TASK_GROUP_TYPE[(dataset_name, task_group_name)] = 'categorical'
            for task in dataset_dict['base'].TASK_GROUP_DICT[task_group_name]:
                TASKS_CATEGORICAL.append((dataset_name, task))

    TASKS = TASKS_BASE + TASKS_CONTINUOUS + TASKS_CATEGORICAL
    TASK_GROUP_NAMES = TASK_GROUP_NAMES_BASE + TASK_GROUP_NAMES_CONTINUOUS + TASK_GROUP_NAMES_CATEGORICAL


class UnifiedTrainDataset(Unified):
    def __init__(self, config, shot, dset_size, verbose=True, **kwargs):
        self.shot = shot
        self.dset_size = dset_size

        self._register_datasets(config, verbose, **kwargs)
        self._register_samplers(config.uniform_task_sampling, config.uniform_dataset_sampling, config.task_sampling_weight)
        
    def _register_datasets(self, config, verbose, unary_augmentation, binary_augmentation, *args, **kwargs):
        self.base_datasets = []
        self.continuous_datasets = []
        self.categorical_datasets = []
        self.unsupervised_datasets = []
        self.tasks = []
        self.base_size = 0
        self.continuous_size = 0
        self.categorical_size = 0
        self.unsupervised_size = 0
        if verbose:
            print('Registering datasets...')
        for dataset_name, dataset_dict in train_dataset_dict.items():
            if getattr(config, dataset_name, False):
                if config.base_task and 'base' in dataset_dict:
                    self.base_datasets.append(
                        dataset_dict['base'](unary_augmentation, binary_augmentation, **kwargs)
                    )
                    for task in dataset_dict['base'].TASKS_BASE:
                        self.tasks.append((dataset_name, task))
                    self.base_size += len(self.base_datasets[-1])
                if config.cont_task and 'continuous' in dataset_dict:
                    if not ((dataset_name == 'coco' and not getattr(config, 'coco_real', True)) or
                            (dataset_name == 'midair' and not getattr(config, 'midair_real', True))):
                        self.continuous_datasets.append(
                            dataset_dict['continuous'](unary_augmentation, binary_augmentation, **kwargs)
                        )
                        for task in dataset_dict['base'].TASKS_CONTINUOUS:
                            self.tasks.append((dataset_name, task))
                        self.continuous_size += len(self.continuous_datasets[-1])
                if config.cat_task and 'categorical' in dataset_dict:
                    if not ((dataset_name == 'coco' and not getattr(config, 'coco_real', True)) or
                            (dataset_name == 'midair' and not getattr(config, 'midair_real', True))):
                        self.categorical_datasets.append(
                            dataset_dict['categorical'](unary_augmentation, **kwargs)
                        )
                        for task in dataset_dict['base'].TASKS_CATEGORICAL:
                            self.tasks.append((dataset_name, task))
                        self.categorical_size += len(self.categorical_datasets[-1])
                # if config.knowledge_distill and 'unsupervised' in dataset_dict:
                #     self.unsupervised_datasets.append(
                #         dataset_dict['unsupervised'](**kwargs)
                #     )
                #     for task in dataset_dict['base'].TASKS_BASE:
                #         self.tasks.append((dataset_name, task))
                #     self.unsupervised_size += len(self.unsupervised_datasets[-1])
                if verbose:
                    print(f'{dataset_name} dataset registered')

        if verbose:
            print(f'Total {self.base_size} base images, {self.continuous_size} continuous images, '
                f'{self.categorical_size} categorical images, {self.unsupervised_size} unsupervised images are registered.')

    def _register_samplers(self, uniform_task_sampling=False, uniform_dataset_sampling=False, task_sampling_weight=None):
        # Base, Continuous, Categorical, Unsupervised
        if uniform_task_sampling:
            p_task_type = torch.tensor([1. if len(self.base_datasets) > 0 else 0.,
                                        1. if len(self.continuous_datasets) > 0 else 0.,
                                        1. if len(self.categorical_datasets) > 0 else 0.,
                                        1. if len(self.unsupervised_datasets) > 0 else 0.])
        else:
            assert task_sampling_weight is not None and len(task_sampling_weight) == 4, task_sampling_weight
            p_task_type = torch.tensor([task_sampling_weight[0] if len(self.base_datasets) > 0 else 0.,
                                        task_sampling_weight[1] if len(self.continuous_datasets) > 0 else 0.,
                                        task_sampling_weight[2] if len(self.categorical_datasets) > 0 else 0.,
                                        task_sampling_weight[3] if len(self.unsupervised_datasets) > 0 else 0.])
        assert p_task_type.sum() > 0
        p_task_type = p_task_type / p_task_type.sum()
        self.task_type_sampler = torch.distributions.Categorical(p_task_type)

        # Base Datasets
        if len(self.base_datasets) > 0:
            if uniform_dataset_sampling:
                p_datasets_base = torch.tensor([1. if len(dset)>0 else 0. for dset in self.base_datasets])
            else:
                p_datasets_base = torch.tensor([len(dset) for dset in self.base_datasets])
            p_datasets_base = p_datasets_base / p_datasets_base.sum()
            self.base_dataset_sampler = torch.distributions.Categorical(p_datasets_base)

        # Continuous Datasets
        if len(self.continuous_datasets) > 0:
            if uniform_dataset_sampling:
                p_datasets_cont = torch.tensor([1. if len(dset.TASKS_CONTINUOUS)>0 else 0. for dset in self.continuous_datasets])
            else:
                p_datasets_cont = torch.tensor([len(dset.TASKS_CONTINUOUS) for dset in self.continuous_datasets])
            p_datasets_cont = p_datasets_cont / p_datasets_cont.sum()
            self.continuous_dataset_sampler = torch.distributions.Categorical(p_datasets_cont)

        # Categorical Datasets
        if len(self.categorical_datasets) > 0:
            if uniform_dataset_sampling:
                p_datasets_cat = torch.tensor([1. if len(dset.TASKS_CATEGORICAL) else 0. for dset in self.categorical_datasets])
            else:
                p_datasets_cat = torch.tensor([len(dset.TASKS_CATEGORICAL) for dset in self.categorical_datasets])
            p_datasets_cat = p_datasets_cat / p_datasets_cat.sum()
            self.categorical_dataset_sampler = torch.distributions.Categorical(p_datasets_cat)

        # Unsupervised Datasets
        if len(self.unsupervised_datasets) > 0:
            if uniform_dataset_sampling:
                p_datasets_unsup = torch.tensor([1. if len(dset)>0 else 0. for dset in self.unsupervised_datasets])
            else:
                p_datasets_unsup = torch.tensor([len(dset) for dset in self.unsupervised_datasets])
            p_datasets_unsup = p_datasets_unsup / p_datasets_unsup.sum()
            self.unsupervised_dataset_sampler = torch.distributions.Categorical(p_datasets_unsup)
        
    def __len__(self):
        return self.dset_size

    def __getitem__(self, idx):
        # sample task type
        task_type = self.task_type_sampler.sample().item()

        # sample dataset
        if task_type == 0:
            dataset_idx = self.base_dataset_sampler.sample().item()
            dataset = self.base_datasets[dataset_idx]
        elif task_type == 1:
            dataset_idx = self.continuous_dataset_sampler.sample().item()
            dataset = self.continuous_datasets[dataset_idx]
        elif task_type == 2:
            dataset_idx = self.categorical_dataset_sampler.sample().item()
            dataset = self.categorical_datasets[dataset_idx]
        elif task_type == 3:
            dataset_idx = self.unsupervised_dataset_sampler.sample().item()
            dataset = self.unsupervised_datasets[dataset_idx]

        # sample episode
        X, Y, M, t_idx = dataset.sample_episode(2*self.shot, 2)

        # reindex task
        if t_idx.item() >= 0:
            task = dataset.TASKS[t_idx.item()]
            t_idx = torch.tensor(self.TASKS.index((dataset.NAME, task)))

        return X[None], Y[None], M[None], t_idx[None] # add channel dimension
