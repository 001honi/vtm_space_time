import os
import random
import numpy as np
from PIL import Image
import skimage
import scipy
import scipy.ndimage

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .augmentation import get_filtering_augmentation, Mixup
from .utils import HeatmapGenerator, SobelEdgeDetector, crop_arrays
from .coco_api_wrapper import SingletonCOCOFactory
from copy import deepcopy


class COCO(Dataset):
    NAME = 'coco'
    NEW_COCO = False # Change this flag to use the new version of COCO dataset.
    # print(f'WARNING: You are using {NAME} dataset with NEW_COCO={NEW_COCO}.')
    # Some changes after the original version used in unified_v2:
    # 1. Multiclasss keypoint task is not used.
    # 2. Images with more than one instance are not used for keypoint task.
    # 3. Generating keypoint labels are done in a different way (parameter changed and use maximum).
    # 4. Spatial softmax loss is used.

    # Class Splits for Semantic Segmentation
    CLASS_IDXS = list(i for i in range(1, 92) if i not in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91])
    CLASS_IDXS += [169, 172, 157, 96, 124] # Five classes from COCO Stuff
    # 183 == other
    # 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91 are removed from COCO.
    CLASS_IDXS_VAL = [169, 172, 157, 96, 124] + [1, 62, 3, 47, 44]
    # Manually set this because there are too many classes in COCO. These are top-10 by img count.
    # tree, wall-concrete, sky-other, building-other, grass | person, chair, car, cup, bottle
    CLASS_RANGE = range(184)
    CLASS_NAMES = \
        ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', # Things
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',  'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] + \
        ['tree', 'wall-concrete', 'sky-other', 'building-other', 'grass'] # Stuff
    KP_CLASSES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    CLASS_IDX_TO_NAME = {c: name for c, name in zip(CLASS_IDXS, CLASS_NAMES)}

    KP_IDXS = list(range(1, 18)) # 1-17
    KP_IDX_TO_NAME = {c: name for c, name in zip(KP_IDXS, KP_CLASSES)}

    # Task Groups
    TASK_GROUPS_BASE = []
    TASK_GROUPS_CONTINUOUS = []
    TASK_GROUPS_CATEGORICAL = []

    # Base Tasks
    TASKS_AUTOENCODING = [f'autoencoding_{c}' for c in range(3)]
    TASK_GROUPS_BASE.append(TASKS_AUTOENCODING)

    TASKS_DENOISING = [f'denoising_{c}' for c in range(3)]
    TASK_GROUPS_BASE.append(TASKS_DENOISING)

    TASKS_EDGE2D = [f'edge_texture_{c}' for c in range(3)]
    TASK_GROUPS_BASE.append(TASKS_EDGE2D)

    # Continuous Tasks
    TASKS_KP_HUMAN = ['keypoints_human_0'] # Original single channel, Gaussian label.
    TASKS_KP_MULTICLASS = ['keypoints_multiclass_0'] if not NEW_COCO else [] # Single channel with binning.
    TASK_GROUPS_CONTINUOUS.append(TASKS_KP_HUMAN)
    if not NEW_COCO:
        TASK_GROUPS_CONTINUOUS.append(TASKS_KP_MULTICLASS)

    # Categorical Tasks
    TASKS_SEMSEG = [f'segment_semantic_{c}' for c in CLASS_IDXS]
    TASKS_KP_SEMANTIC = [f'keypoints_semantic_{c}' for c in KP_IDXS]
    TASK_GROUPS_CATEGORICAL.append(TASKS_SEMSEG)
    TASK_GROUPS_CATEGORICAL.append(TASKS_KP_SEMANTIC)

    # Task Group Names
    TASK_GROUP_NAMES_BASE = ['_'.join(task_group[0].split('_')[:-1])
                             for task_group in TASK_GROUPS_BASE]
    TASK_GROUP_NAMES_CONTINUOUS = ['_'.join(task_group[0].split('_')[:-1])
                                   for task_group in TASK_GROUPS_CONTINUOUS]
    TASK_GROUP_NAMES_CATEGORICAL = ['_'.join(task_group[0].split('_')[:-1])
                                    for task_group in TASK_GROUPS_CATEGORICAL]
    TASK_GROUP_NAMES = TASK_GROUP_NAMES_BASE + TASK_GROUP_NAMES_CONTINUOUS + TASK_GROUP_NAMES_CATEGORICAL

    # All Task Groups, Task Group Dict, and Tasks
    TASK_GROUPS = TASK_GROUPS_BASE + TASK_GROUPS_CONTINUOUS + TASK_GROUPS_CATEGORICAL
    TASK_GROUP_DICT = {name: group for name, group in zip(TASK_GROUP_NAMES, TASK_GROUPS)}

    TASKS_BASE = [task for task_group in TASK_GROUPS_BASE for task in task_group]
    TASKS_CONTINUOUS = [task for task_group in TASK_GROUPS_CONTINUOUS for task in task_group]
    TASKS_CATEGORICAL = [task for task_group in TASK_GROUPS_CATEGORICAL for task in task_group]
    TASKS = TASKS_BASE + TASKS_CONTINUOUS + TASKS_CATEGORICAL

    def __init__(self, path_dict, split, base_size=(256, 256), crop_size=(224, 224), 
                 seed=None, precision='fp32', meta_dir='meta_info'):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        if split == 'valid':
            split = 'val'
        assert split in ['train', 'val'], split  # TODO: add train+val
        self.img_root = os.path.join(path_dict[self.NAME], 'images', f'{split}2017_{base_size[0]}')
        self.ann_root = os.path.join(path_dict[self.NAME], 'annotations')
        self.split = split
        
        self.base_size = base_size
        self.crop_size = crop_size
        self.precision = precision
        self.meta_info_path = os.path.join('dataset', meta_dir, self.NAME)
        # For using unlabeled data
        self.imgIds_unlabeled = set()
        self.unlabeled_root = os.path.join(path_dict[self.NAME], 'images', f'unlabeled2017_{base_size[0]}')
            
        # register euclidean depth and occlusion edge statistics, sobel edge detectors, and class dictionary
        self.edge_params = torch.load(os.path.join(self.meta_info_path, 'edge_params.pth'))
        self.sobel_detectors = [SobelEdgeDetector(kernel_size=k, sigma=s) for k, s in self.edge_params['params']]
        self.class_dict = torch.load(os.path.join(self.meta_info_path, f'coco_{split}_class_dict.pth'))

        # transforms
        self.toten = ToTensor()

    def _resize(self, target, size=None, mode='bilinear'):
        '''
        Resize image or label to given size(default to self.base_size with bilinear).
        Dimension is assumed to be 3D (C, H, W)
        '''
        if size is None:
            size = self.base_size
        assert mode in ['bilinear', 'nearest']
        target = F.interpolate(target.unsqueeze(0), size=size, mode=mode).squeeze(0)
        return target

    def _load_image(self, imgId):
        '''
        get image idx and return tensor
        '''
        # get path
        file_name = self.coco.imgs[imgId]['file_name']
        if imgId in self.imgIds_unlabeled:
            img_path = os.path.join(self.unlabeled_root, file_name)
        else:
            img_path = os.path.join(self.img_root, file_name)
        
        # open image file
        img = Image.open(img_path)
        img = self.toten(img)
        if len(img) == 1:
            img = img.repeat(3, 1, 1)
        
        return img

    def _load_label(self, task_group, imgId, class_id=None):
        '''
        return torch Tensor
        '''
        if 'keypoints' in task_group: # keypoints_human, keypoints_multiclass (if not NEWCOCO), keypoints_semantic
            # create a single-channel keypoint label from image id
            kp_anns = self.coco.getAnnIds(imgIds=[imgId])
            height, width = self.coco.imgs[imgId]['height'], self.coco.imgs[imgId]['width']
            keypoints = np.zeros((len(kp_anns), 17, 3))
            for i in range(len(kp_anns)):
                keypoints[i] = np.array(self.coco.anns[kp_anns[i]]['keypoints']).reshape(17, 3)

            # move w, h coordinates, because of resizing
            keypoints[..., 0] = keypoints[..., 0] * self.base_size[1] / width
            keypoints[..., 1] = keypoints[..., 1] * self.base_size[0] / height
            keypoints = keypoints.astype(int)
            kernel= np.zeros(self.base_size, dtype=np.float32)
            for inst in keypoints:
                for i, (w, h, v) in enumerate(inst):
                    if v == 2:
                        kernel[h, w] = (i + 1) / len(self.KP_CLASSES) # encode class information, zero for background.

            label = torch.from_numpy(kernel).unsqueeze(0)
            ret_label = torch.zeros_like(label)

            # generate appropriate label for each task
            if task_group == 'keypoints_human':
                target_pos = label.nonzero()
                # pad ret_label to properly handle boundary
                ret_label = F.pad(ret_label, (self.kp_radius, self.kp_radius, self.kp_radius, self.kp_radius), mode='constant', value=0)
                for (c, h, w) in target_pos.tolist():
                    h_start = h
                    h_end = h + 2*self.kp_radius + 1
                    w_start = w
                    w_end = w + 2*self.kp_radius + 1
                    if self.NEW_COCO:
                        ret_label[c, h_start:h_end, w_start:w_end] = torch.maximum(ret_label[c, h_start:h_end, w_start:w_end] , self.gaussian)
                    else:
                        ret_label[c, h_start:h_end, w_start:w_end] += self.gaussian
                ret_label = torch.clip(ret_label, 0, 1) # For safety
                # remove padding
                ret_label = ret_label[:, self.kp_radius:-self.kp_radius, self.kp_radius:-self.kp_radius]
            elif task_group == 'keypoints_multiclass':
                assert not self.NEW_COCO
                target_pos = label.nonzero()
                for (c, h, w) in target_pos.tolist():
                    # enlarge the single pixel label to 3x3 box.
                    h_start = max(0, h-1)
                    h_end = min(label.shape[-2], h+2)
                    w_start = max(0, w-1)
                    w_end = min(label.shape[-1], w+2)
                    ret_label[c, h_start:h_end, w_start:w_end] = label[c, h, w] # this may lead to overwriting other labels
            elif task_group == 'keypoints_semantic':
                assert class_id is not None
                target_pos = label.nonzero()
                # pad ret_label to properly handle boundary
                ret_label = F.pad(ret_label, (self.kp_radius, self.kp_radius, self.kp_radius, self.kp_radius), mode='constant', value=0)
                for (c, h, w) in target_pos.tolist():
                    if label[c, h, w] == (class_id / len(self.KP_CLASSES)):
                        h_start = h
                        h_end = h + 2*self.kp_radius + 1
                        w_start = w
                        w_end = w + 2*self.kp_radius + 1
                        if self.NEW_COCO:
                            ret_label[c, h_start:h_end, w_start:w_end] = torch.maximum(ret_label[c, h_start:h_end, w_start:w_end] , self.gaussian)
                        else:
                            ret_label[c, h_start:h_end, w_start:w_end] += self.gaussian
                ret_label = torch.clip(ret_label, 0, 1) # For safety
                # remove padding
                ret_label = ret_label[:, self.kp_radius:-self.kp_radius, self.kp_radius:-self.kp_radius]
            else:
                raise ValueError(task_group)
            label = ret_label
        
        elif task_group == 'segment_semantic':
            label = np.zeros((self.coco.imgs[imgId]['height'],
                        self.coco.imgs[imgId]['width']), dtype='uint8')

            # all
            if class_id is None:
                raise NotImplementedError
                annIds = self.coco.getAnnIds(imgIds=[imgId])
                anns = self.coco.loadAnns(annIds)
                annIds = self.coco_stuff.getAnnIds(imgIds=[imgId])
                anns += self.coco_stuff.loadAnns(annIds)

            # single class
            else:
                # things
                if class_id < 92:
                    annIds = self.coco_things.getAnnIds(imgIds=[imgId], catIds=[class_id])
                    anns = self.coco_things.loadAnns(annIds)
                # stuff
                else:
                    annIds = self.coco_stuff.getAnnIds(imgIds=[imgId], catIds=[class_id])
                    anns = self.coco_stuff.loadAnns(annIds)

                # load annotations
                for ann in anns:
                    label += self.coco.annToMask(ann) * class_id

            label = self.toten(label) * 255
            label = self._resize(label, mode='nearest')

        else:
            raise NotImplementedError
        
        return label
            
    def _load_and_preprocess_label(self, task_group, img_file, channels):
        # load label
        if task_group in self.TASK_GROUP_NAMES_BASE:
            label = mask = None
        else:
            if task_group in self.TASK_GROUP_NAMES_CONTINUOUS:
                # load label
                label = self._load_label(task_group, img_file)

                # bound label values
                label = torch.clip(label, 0, 1)
                mask = None
        
            # preprocess label
            elif task_group in self.TASK_GROUP_NAMES_CATEGORICAL:
                # load label
                label = self._load_label(task_group, img_file, class_id=channels[0])

                mask = None

            else:
                raise ValueError(f'Invalid task group: {task_group}')
            
        return label, mask
    
    def _postprocess_keypoints_human(self, labels, channels):
        '''
        labels: N C(=1) H W
        '''
        return labels, torch.ones_like(labels)

    def _postprocess_autoencoding(self, imgs, channels):
        labels = imgs.clone()
        labels = labels[:, channels]
        masks = torch.ones_like(labels)

        return labels, masks

    def _postprocess_denoising(self, imgs, channels):
        labels = imgs.clone()
        imgs = torch.from_numpy(skimage.util.random_noise(imgs, var=0.01))
            
        labels = labels[:, channels]
        masks = torch.ones_like(labels)

        return imgs, labels, masks
    
    def _postprocess_edge_texture(self, imgs, channels):
        labels = []
        # detect sobel edge with different set of pre-defined parameters
        for c in channels:
            labels_ = self.sobel_detectors[c].detect(imgs)
            labels.append(labels_)
        labels = torch.cat(labels, 1)

        # thresholding and re-normalizing
        labels = torch.clip(labels, 0, self.edge_params['threshold'])
        labels = labels / self.edge_params['threshold']

        masks = torch.ones_like(labels)
        
        return labels, masks
      
    def _postprocess_segment_semantic(self, labels, channels):
        if channels is not None:
            assert len(channels) == 1
            labels = (labels == channels[0]).float()
        masks = torch.ones_like(labels)

        return labels, masks

    def _postprocess_default(self, labels, masks, channels):
        labels = labels[:, channels]

        if masks is not None:
            masks = masks.expand_as(labels)
        else:
            masks = torch.ones_like(labels)
            
        return labels, masks
    
    def _postprocess(self, imgs, labels, masks, channels, task):
        # process all channels if not given
        if channels is None:
            assert task != 'segment_semantic'
            channels = range(len(self.TASK_GROUP_DICT[task]))
            
        # task-specific preprocessing
        if task == 'autoencoding':
            labels, masks = self._postprocess_autoencoding(imgs, channels)
        elif task == 'denoising':
            imgs, labels, masks = self._postprocess_denoising(imgs, channels)
        elif task == 'edge_texture':
            labels, masks = self._postprocess_edge_texture(imgs, channels)
        elif task == 'segment_semantic':
            labels, masks = self._postprocess_segment_semantic(labels, channels)
        # keypoints_human, keypoints_multiclass, keypoints_semantic
        elif task in ['keypoints_human', 'keypoints_multiclass', 'keypoints_semantic']:
            labels, masks = self._postprocess_keypoints_human(labels, channels)
        else:
            labels, masks = self._postprocess_default(labels, masks, channels)

        # ensure label values to be in [0, 1]
        labels = labels.clip(0, 1)

        return imgs, labels, masks

    def load_images(self, imgIds):
        # load images
        imgs = []
        for imgId in imgIds:
            # load image
            img = self._load_image(imgId)
            imgs.append(img)
        imgs = torch.stack(imgs)

        return imgs
    
    def load_labels(self, imgs, task_group, channels, imgIds):
        # load labels
        labels = []
        masks = []
        for imgId in imgIds:
            # load label
            label, mask = self._load_and_preprocess_label(task_group, imgId, channels)
            labels.append(label)
            masks.append(mask)
        labels = torch.stack(labels) if labels[0] is not None else None
        masks = torch.stack(masks) if masks[0] is not None else None

        # postprocess labels
        imgs, labels, masks = self._postprocess(imgs, labels, masks, channels, task_group)
        
        # precision conversion
        if self.precision == 'fp16':
            imgs = imgs.half()
            labels = labels.half()
            masks = masks.half()
        elif self.precision == 'bf16':
            imgs = imgs.bfloat16()
            labels = labels.bfloat16()
            masks = masks.bfloat16()

        return imgs, labels, masks

    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError


class COCOBaseDataset(COCO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.coco = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, f'person_keypoints_{self.split}2017.json'))

        # Get valid imgIds with at least on visible keypoint.
        keypoint_ids_path = os.path.join(self.ann_root, f'person_keypoints_{self.split}2017_keypoint_ids.pth')
        if os.path.exists(keypoint_ids_path):
            self.imgIds = torch.load(keypoint_ids_path)
        else:
            self.imgIds = []
            for imgid in sorted(self.coco.getImgIds(catIds=1)): # person
                if len(self.coco.getAnnIds(imgIds=[imgid])) > 0:
                    y = []
                    for i in range(len(self.coco.getAnnIds(imgIds=[imgid]))):
                        y += self.coco.anns[self.coco.getAnnIds(imgIds=[imgid])[i]]['keypoints']
                    y = np.array(y).reshape(-1, 3)
                    if (y[:, 2] == 2).astype(np.float64).sum() > 0:
                        self.imgIds.append(imgid)
            torch.save(self.imgIds, keypoint_ids_path)

        if self.NEW_COCO:
            # Filter out images with many instances.
            skip_imgIds = set(imgId for imgId in self.coco.getImgIds() if len(self.coco.getAnnIds(imgIds=imgId)) != 1)
            self.imgIds = [imgId for imgId in self.imgIds if imgId not in skip_imgIds]

        # Some attributes related to construct continuous keypoint labels
        self.kp_sigma = 2.0 if self.NEW_COCO else 3.0
        self.kp_max = 1.0 
        self.kp_truncate = 3.0 if self.NEW_COCO else 4.0
        self.kp_radius = round(self.kp_truncate * self.kp_sigma)
        base_array = np.zeros((2*self.kp_radius+1, 2*self.kp_radius+1))
        base_array[self.kp_radius, self.kp_radius] = 1
        self.gaussian = scipy.ndimage.gaussian_filter(base_array, sigma=self.kp_sigma, mode='constant', truncate=self.kp_truncate)
        self.gaussian = torch.tensor(self.gaussian / self.gaussian.max() * self.kp_max)

    def __len__(self):
        return len(self.imgIds)


class COCOBaseTrainDataset(COCOBaseDataset):
    def __init__(self, unary_augmentation, binary_augmentation, *args, use_unlabeled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.unary_augmentation = get_filtering_augmentation() if unary_augmentation else None
        self.binary_augmentation = Mixup() if binary_augmentation else None
        self.tasks = self.TASKS_BASE
        if self.binary_augmentation:
            self.coco_things = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, f'instances_{self.split}2017.json'))
            self.coco_stuff = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, f'stuff_{self.split}2017.json'))
        if use_unlabeled:
            self.coco = deepcopy(self.coco) # to avoid modifying the original singleton coco
            self.coco_unlabeled = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, 'image_info_unlabeled2017.json'))
            self.coco.imgs.update(self.coco_unlabeled.imgs)
            self.imgIds_unlabeled = set(self.coco_unlabeled.imgs.keys())
            self.imgIds = sorted(self.coco.getImgIds())
        
    def sample_task(self, from_all=False):
        if from_all:
            task = np.random.choice(self.TASKS)
        else:
            task = np.random.choice(self.tasks)
        task_group, channel = '_'.join(task.split('_')[:-1]), int(task.split('_')[-1])
        t_idx = torch.tensor(self.TASKS.index(task))

        return task_group, channel, t_idx
    
    def sample_files(self, n_files):
        # sample imageIds 
        imgIds = np.random.choice(self.imgIds, n_files, replace=False)
        return imgIds
 
    def sample_episode(self, n_files, n_domains=None):
        # sample tasks
        task_group, channel, t_idx = self.sample_task()
        if self.binary_augmentation is not None:
            task_group_aux, channel_aux, _ = self.sample_task(from_all=True)

        # sample image paths
        file_idxs = self.sample_files(n_files)

        # load images
        imgs = self.load_images(file_idxs)

        # load labels
        X, Y, M = self.load_labels(imgs, task_group, [channel], file_idxs)
        if self.binary_augmentation:
            _, Y_aux, M_aux = self.load_labels(imgs, task_group_aux, [channel_aux], file_idxs)

        # apply unary task augmentation
        if self.unary_augmentation:
            Y, M = self.unary_augmentation(Y, M)
            if self.binary_augmentation:
                Y_aux, M_aux = self.unary_augmentation(Y_aux, M_aux)

        # apply binary taks augmentation
        if self.binary_augmentation:
            Y, M = self.binary_augmentation(Y, Y_aux, M, M_aux)
        
        # random-crop arrays
        X, Y, M = crop_arrays(X, Y, M,
                              base_size=self.base_size,
                              crop_size=self.crop_size,
                              random=True)
        
        return X, Y, M, t_idx
    

class COCOBaseTestDataset(COCOBaseDataset):
    def __init__(self, task_group, dset_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_group = task_group
        self.dset_size = dset_size

    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.imgIds)
        
    def __getitem__(self, idx):
        imgIds = [self.imgIds[idx]]
        X = self.load_images(imgIds)
        X, Y, M = self.load_labels(X, self.task_group, None, imgIds)

        return X[0], Y[0], M[0]


class COCOContinuousTrainDataset(COCOBaseTrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_unlabeled=False, **kwargs)
        self.tasks = self.TASKS_CONTINUOUS


class COCOCategoricalDataset(COCOBaseDataset):
    '''
    In principle this should inherit from COCO, but it is really similar to COCOBaseDataset so we inherit from it.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kp_imgIds = self.imgIds # in super init, imgIds will be set to the ones with visible keypoints
        # make COCO object
        self.coco_things = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, f'instances_{self.split}2017.json'))
        self.coco_stuff = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, f'stuff_{self.split}2017.json'))
        # assert self.coco_stuff.getImgIds() == self.coco_things.getImgIds() <- this is already checked with jupyter notebook

        self.imgIds = sorted(self.coco.getImgIds())

    def __len__(self):
        return len(self.imgIds)
    

class COCOCategoricalTrainDataset(COCOCategoricalDataset):
    def __init__(self, unary_augmentation, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unary_augmentation = get_filtering_augmentation() if unary_augmentation else None
        self.tasks = self.TASKS_CATEGORICAL
        
    def sample_task(self):
        task = np.random.choice(self.tasks)
        task_group, channel = '_'.join(task.split('_')[:-1]), int(task.split('_')[-1])
        t_idx = torch.tensor(self.TASKS.index(task))

        return task_group, channel, t_idx
    
    def sample_files(self, c, n_files, in_kp=False):
        '''
        Samples image file indices for each domain.
        If in_kp is True, samples only from the keypoints domain.
        '''
        if in_kp:
            file_idxs = np.random.choice(self.kp_imgIds, n_files, replace=False)
        else:
            file_idxs =  np.random.choice(self.class_dict[c], n_files, replace=False)
        return file_idxs
        
    def sample_episode(self, n_files, n_domains=None):
        # sample tasks
        task_group, channel, t_idx = self.sample_task()

        # sample image paths
        in_kp = ('keypoints' in task_group)
        file_idxs = self.sample_files(channel, n_files, in_kp=in_kp)

        # load images and labels
        X = self.load_images(file_idxs)
        X, Y, M = self.load_labels(X, task_group, [channel], file_idxs)

        # apply unary task augmentation
        if self.unary_augmentation:
            Y, M = self.unary_augmentation(Y, M)

        # random-crop arrays
        X, Y, M = crop_arrays(X, Y, M,
                              base_size=self.base_size,
                              crop_size=self.crop_size,
                              random=True)
        
        return X, Y, M, t_idx


class COCOCategoricalTestDataset(COCOCategoricalDataset):
    def __init__(self, task_group, class_id, dset_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_id = class_id
        self.dset_size = dset_size
        self.task_group = task_group

    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        elif self.task_group == 'keypoints_semantic':
            return len(self.kp_imgIds)
        else:
            return len(self.class_dict[self.class_id])
    
    def __getitem__(self, idx):
        if self.task_group == 'keypoints_semantic':
            imgId = self.kp_imgIds[idx]
        else:
            imgId = self.class_dict[self.class_id][idx]
        X = self.load_images([imgId])
        X, Y, M = self.load_labels(X, self.task_group, [self.class_id], [imgId])

        return X[0], Y[0], M[0]
    

class COCOUnsupervisedTrainDataset(COCO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.coco = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, f'instances_{self.split}2017.json'), force_get_new=True)
        self.coco_unlabeled = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, f'image_info_unlabeled2017.json'))
        self.coco.imgs.update(self.coco_unlabeled.imgs)
        self.imgIds_unlabeled = set(self.coco_unlabeled.imgs.keys())
        self.imgIds = sorted(self.coco.getImgIds())

    def sample_files(self, n_files):
        # sample image file indices
        file_idxs = np.random.choice(self.imgIds, n_files, replace=False)
        return file_idxs
        
    def sample_episode(self, n_files, n_domains=None):
        # sample image paths
        file_idxs = self.sample_files(n_files)

        # load images
        X = self.load_images(file_idxs)

        # crop arrays
        X = crop_arrays(X,
                        base_size=self.base_size,
                        crop_size=self.crop_size,
                        random=True)[0]
        
        # create empty labels and masks
        Y = torch.empty(X.shape[0], 1, X.shape[2], X.shape[3])
        M = torch.empty(X.shape[0], 1, X.shape[2], X.shape[3])
        t_idx = torch.tensor(-1)
        
        # precision conversion
        if self.precision == 'fp16':
            X = X.half()
            Y = Y.half()
            M = M.half()
        elif self.precision == 'bf16':
            X = X.bfloat16()
            Y = Y.bfloat16()
            M = M.bfloat16()

        return X, Y, M, t_idx

    def __len__(self):
        return len(self.imgIds)


class COCOUnsupervisedTestDataset(COCO):
    def __init__(self, dset_size=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dset_size = dset_size
        self.coco = SingletonCOCOFactory.get_coco(os.path.join(self.ann_root, f'instances_{self.split}2017.json'))
        self.imgIds = sorted(self.coco.getImgIds())

    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            return len(self.imgIds)
        
    def __getitem__(self, idx):
        # load image
        X = self._load_image(self.imgIds[idx])
        
        # create empty labels and masks
        Y = torch.empty(1, X.shape[1], X.shape[2])
        M = torch.empty(1, X.shape[1], X.shape[2])
        
        # precision conversion
        if self.precision == 'fp16':
            X = X.half()
            Y = Y.half()
            M = M.half()
        elif self.precision == 'bf16':
            X = X.bfloat16()
            Y = Y.bfloat16()
            M = M.bfloat16()

        return X