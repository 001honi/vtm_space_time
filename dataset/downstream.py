import os
from PIL import Image
import numpy as np
import scipy
from itertools import zip_longest
import random
import skimage

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets import KittiFlow
import torch.nn.functional as F

from .utils import crop_arrays, simple_overlapped_crop, SobelEdgeDetector
from .augmentation import get_filtering_augmentation, Mixup
from .coco_api_wrapper import SilentCOCO


class DownstreamBaseDataset(Dataset):
    def __init__(self, dataset, data_root, base_size, img_ext='.jpg', lbl_ext='.png',
                 class_name='', shuffle_idxs=False, precision='fp32'):
        super().__init__()
        self.dataset = dataset
        self.data_root = data_root
        if class_name == 'none':
            class_name = ''
        self.image_dir = os.path.join(data_root, f'resized_{base_size[1]}', class_name, 'images')
        self.label_dir = os.path.join(data_root, f'resized_{base_size[0]}', class_name, 'labels')
        self.img_ext = img_ext
        self.lbl_ext = lbl_ext
        self.class_name = class_name
        self.shuffle_idxs = shuffle_idxs
        self.precision = precision
        self.base_size = base_size
        
        self.data_idxs = np.array(sorted(os.listdir(self.image_dir)))
        if shuffle_idxs:
            perm_path = f'dataset/meta_info/idxs_perm_{dataset}.pth'
            if os.path.exists(perm_path):
                self.idxs_perm = torch.load(perm_path)
            else:
                self.idxs_perm = torch.randperm(len(self.data_idxs))
                torch.save(self.idxs_perm, perm_path)
            self.data_idxs = self.data_idxs[self.idxs_perm]
        
        self.toten = ToTensor()
        
    def __len__(self):
        return len(self.data_idxs)
    
    def load_data(self, img_path, resize=None):
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        if resize is not None:
            image = image.resize(resize)
        
        lbl_path = img_path.replace(self.img_ext, self.lbl_ext)
        if self.lbl_ext == '.npy':
            label = np.load(os.path.join(self.label_dir, lbl_path))
            if label.ndim == 2:
                label = label[:, :, None]
        else:
            label = Image.open(os.path.join(self.label_dir, lbl_path))
            if self.dataset == 'duts':
                label = label.convert('L') # temporary fix for ill formatted labels in duts
            if resize is not None:
                label = label.resize(resize)
        
        return image, label
    
    def postprocess_data(self, image, label):
        X = self.toten(image)
        if isinstance(label, np.ndarray):
            Y = torch.from_numpy(label).permute(2, 0, 1)
        else:
            Y = self.toten(label)
            
        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
        
        return X, Y

    def __getitem__(self, idx):
        img_path = self.data_idxs[idx % len(self.data_idxs)]
        image, label = self.load_data(img_path)
        
        return self.postprocess_data(image, label)
    
    
class DownstreamFinetuneDataset(DownstreamBaseDataset):
    def __init__(self, split, shot, support_idx, channel_idx, dset_size,
                 base_size, img_size, fix_seed=False, train_files=False, sequential=False, **kwargs):
        super().__init__(base_size=base_size, **kwargs)
        assert split in ['train', 'valid', 'test']
        self.split = split
        self.shot = shot
        self.support_idx = support_idx
        self.channel_idx = channel_idx
        self.dset_size = dset_size
        self.base_size = base_size
        self.img_size = img_size
        self.fix_seed = fix_seed

        if train_files:
            file_path = os.path.join(self.data_root, 'meta', self.class_name, 'train.txt')
            with open(file_path, 'r') as f:
                train_files = [line.strip().split('/')[-1] for line in f.readlines()]
            if split == 'train':
                self.data_idxs = [data_idx for data_idx in self.data_idxs
                                if data_idx in train_files]
            else:
                self.data_idxs = [data_idx for data_idx in self.data_idxs
                                if data_idx not in train_files]
            self.data_idxs = np.array(self.data_idxs)
        if self.dset_size < 0:
            self.dset_size = len(self.data_idxs)

        train_idxs, valid_idxs, test_idxs = self.split_idxs(not self.shuffle_idxs,
                                                            test_all=(train_files is not None and
                                                                      (split == 'test')))
        if sequential and split == 'valid' and len(valid_idxs) > 10:
            n_vis = 10
            valid_idxs_reordered = []
            vis_idxs = torch.linspace(min(valid_idxs), max(valid_idxs), n_vis).round().long().tolist()
            vis_idxs = [min(valid_idxs, key=lambda x:abs(x-vis_idx)) for vis_idx in vis_idxs]
            for i in vis_idxs:
                valid_idxs_reordered.append(i)
            for i in valid_idxs:
                if i not in vis_idxs:
                    valid_idxs_reordered.append(i)
            valid_idxs = valid_idxs_reordered

        if split == 'train':
            self.data_idxs = self.data_idxs[train_idxs]
        elif split == 'valid':
            self.data_idxs = self.data_idxs[valid_idxs]
        else:
            self.data_idxs = self.data_idxs[test_idxs]
        
    def __len__(self):
        return self.dset_size

    def split_idxs(self, uniform_split=True, test_all=False):
        N = len(self.data_idxs)
        shot = self.shot
        
        if test_all:
            return None, None, np.arange(N)
        
        if uniform_split:
            train_idxs = [0]
            for i in range(1, shot - 1):
                train_idxs.append((N // (shot - 1)) * i)
            if shot > 1:
                train_idxs.append(N - 1)
            assert len(train_idxs) == shot
            
            valid_idxs = [1]
            for i in range(1, self.dset_size - 1):
                valid_idxs.append((N // (self.dset_size - 1)) * i + 1)
            if self.dset_size > 1:
                valid_idxs.append(N - 2)
            assert len(valid_idxs) == self.dset_size, f'{len(valid_idxs)} != {self.dset_size}'

            test_idxs = []
            for i in range(N):
                if (i not in train_idxs):
                    test_idxs.append(i)
        else:
            offset = self.support_idx*self.shot
            train_idxs = np.array(range(offset, offset + shot))
            valid_idxs = np.array(list(range(offset)) + list(range(offset + shot, N)))[:self.dset_size]
            test_idxs = np.array(list(range(offset)) + list(range(offset + shot, N)))

        return train_idxs, valid_idxs, test_idxs
    
    def postprocess_data(self, image, label):
        X, Y = super().postprocess_data(image, label)
        if self.channel_idx >= 0:
            Y = Y[self.channel_idx:self.channel_idx+1]

        valid_indices = Y[0].nonzero()
        upper_left = valid_indices.min(dim=0).values.tolist()
        lower_right = valid_indices.max(dim=0).values.tolist()
        offset_cuts = upper_left + lower_right

        X, Y = crop_arrays(X, Y,
                           base_size=X.size()[-2:],
                           crop_size=self.img_size,
                           random=(not self.fix_seed),
                           offset_cuts=offset_cuts)
        M = torch.ones_like(Y)
            
        return X, Y, M


class DAVIS2017FinetuneDataset(DownstreamFinetuneDataset):
    def __init__(self, *args, permute_classes=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = self.get_n_classes()
        self.permute_classes = permute_classes

    def get_n_classes(self):
        img_path = self.data_idxs[0] # select first frame
        lbl_path = img_path.replace(self.img_ext, self.lbl_ext)
        label = Image.open(os.path.join(self.label_dir, lbl_path))
        n_classes = len(self.toten(label).unique()) - 1 # number of objects (not including background)
        return n_classes

    def postprocess_data(self, image, label):
        X, Y = super(DownstreamFinetuneDataset, self).postprocess_data(image, label)
        Y = (Y*255).long()

        # permute_classes class indices
        if self.permute_classes:
            if self.n_classes == 5:
                perm = [3, 1, 4, 0, 5, 2]
            elif self.n_classes == 4:
                perm = [3, 1, 4, 0, 2]
            elif self.n_classes == 3:
                perm = [2, 1, 3, 0]
            elif self.n_classes == 2:
                perm = [1, 2, 0]
            elif self.n_classes == 1:
                perm = [1, 0]

            for c in range(self.n_classes + 1):
                Y[Y == c]  = self.n_classes + 1 + c
            for c in range(self.n_classes + 1):
                Y[Y == self.n_classes + 1 + perm[c]] = c

        # normalize class indices
        Y = Y / self.n_classes

        X, Y = crop_arrays(X, Y,
                           base_size=X.size()[-2:],
                           crop_size=self.img_size,
                           random=(not self.fix_seed))
        M = torch.ones_like(Y)
            
        return X, Y, M


class ISIC2018FinetuneDataset(Dataset):
    def __init__(self, dataset, data_root, base_size, img_size, split, shot, support_idx, channel_idx=-1, dset_size=-1,
                 fix_seed=False, train_files=False, sequential=False, img_ext='.jpg', lbl_ext='.png', class_name='', shuffle_idxs=False, precision='fp32'):
        
        super().__init__()
        self.dataset = dataset
        self.data_root = data_root
        if class_name == 'none':
            class_name = ''
        if split == 'test':
            self.image_dir = os.path.join(data_root, 'original', class_name, 'images')
            self.label_dir = os.path.join(data_root, 'original', class_name, 'labels')
        else:
            self.image_dir = os.path.join(data_root, f'resized_{base_size[0]}', class_name, 'images')
            self.label_dir = os.path.join(data_root, f'resized_{base_size[0]}', class_name, 'labels')
        self.img_ext = img_ext
        self.lbl_ext = lbl_ext
        self.class_name = class_name
        self.shuffle_idxs = shuffle_idxs
        self.precision = precision

        assert split in ['train', 'valid', 'test']
        self.split = split
        self.shot = shot
        self.support_idx = support_idx
        self.channel_idx = channel_idx
        self.dset_size = dset_size
        self.base_size = base_size
        self.img_size = img_size
        self.fix_seed = fix_seed
        self.toten = ToTensor()

        if split == 'test':
            file_path = os.path.join(self.data_root, 'meta', 'test_files.pth')
        else:
            file_path = os.path.join(self.data_root, 'meta', 'train_files.pth')
        file_dict = torch.load(file_path)

        self.file_names = []
        if split == 'test':
            for class_name in file_dict.keys():
                self.file_names += file_dict[class_name]
            if self.dset_size < 0:
                self.dset_size = len(self.file_names)
        else:
            # get class names sorted by number of files
            class_names = list(file_dict.keys())
            class_names.sort(key=lambda x: len(file_dict[x]))
            class_names = list(reversed(class_names))

            # choose number of files per class
            shot_per_class = [shot // len(class_names) for _ in range(len(class_names))]
            for i in range(shot % len(class_names)):
                shot_per_class[i] += 1
            
            # choose files for training
            if split == 'train':
                for class_name, n_files in zip(class_names, shot_per_class):
                    self.file_names += file_dict[class_name][support_idx*n_files:(support_idx+1)*n_files]
                    if len(file_dict[class_name]) - support_idx*n_files < n_files:
                        self.file_names += file_dict[class_name][:(n_files - len(file_dict[class_name]) + support_idx*n_files)]
            
            # choose files for validation
            else:
                files_per_class = [dset_size // len(class_names) for _ in range(len(class_names))]
                for i in range(dset_size % len(class_names)):
                    files_per_class[i] += 1
        
                for class_name, n_files_train, n_files_val in zip(class_names, shot_per_class, files_per_class):
                    valid_files = file_dict[class_name][:support_idx*n_files_train] + file_dict[class_name][(support_idx+1)*n_files_train:]
                    self.file_names += valid_files[:n_files_val]
            
    def __len__(self):
        return self.dset_size
    
    def __getitem__(self, idx):
        img_path = self.file_names[idx % len(self.file_names)] + self.img_ext
        image, label = self.load_data(img_path, resize=(512, 512))
        
        return self.postprocess_data(image, label)
    
    def load_data(self, img_path, resize=None):
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        if resize is not None:
            image = image.resize(resize)

        
        lbl_path = img_path.replace(self.img_ext, self.lbl_ext)
        if self.lbl_ext == '.npy':
            label = np.load(os.path.join(self.label_dir, lbl_path))
            if label.ndim == 2:
                label = label[:, :, None]
        else:
            label = Image.open(os.path.join(self.label_dir, lbl_path))
            if resize is not None:
                label = label.resize(resize)
        
        return image, label
    
    def postprocess_data(self, image, label):
        X = self.toten(image)
        if isinstance(label, np.ndarray):
            Y = torch.from_numpy(label).permute(2, 0, 1)
        else:
            Y = self.toten(label)
            
        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)

        if self.channel_idx >= 0:
            Y = Y[self.channel_idx:self.channel_idx+1]

        if self.split != 'test':
            X, Y = crop_arrays(X, Y,
                               base_size=X.size()[-2:],
                               crop_size=self.img_size,
                               random=(not self.fix_seed))
        M = torch.ones_like(Y)
            
        return X, Y, M


class DUTSTestDataset(DownstreamFinetuneDataset):
    '''
    This is just a DownstreamFinetuneDataset, but it also returns the label path in the original resolution
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        test_label_dict_path = os.path.join(self.data_root, 'meta', 'test_label_dict.pth')
        self.test_label_dict = torch.load(test_label_dict_path)

    def __getitem__(self, idx):
        img_path = self.data_idxs[idx % len(self.data_idxs)]
        X, Y, M = super().__getitem__(idx)
        Y_ori_path = self.test_label_dict[img_path]

        return X, Y, M, Y_ori_path

    def load_data(self, img_path):
        image, label = super().load_data(img_path)
        label = label.convert('L') # convert label to grayscale
        
        return image, label
    

class LoveDAFinetuneDataset(DownstreamFinetuneDataset):
    PREPROCESS_DICT = {
        (224, 224): {
            'interp_size': (512, 512),
            'crop_stride': 128,
        },
        (256, 256): {
            'interp_size': (512, 512),
            'crop_stride': 128,
        }
    }
    def __init__(self, multichannel=False, crop_not_resize=True, patches_per_img=4, **kwargs):
        self.multichannel = multichannel
        self.compatible_init(kwargs)
        self.semseg_classes = ['IGNORE', 'background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural']
        if not multichannel:
            self.class_id = self.semseg_classes.index(self.class_name)

        self.crop_not_resize = crop_not_resize
        self.patches_per_img = patches_per_img if self.crop_not_resize else 1
        self.interp_size = self.PREPROCESS_DICT[self.img_size]['interp_size']
        self.crop_stride = self.PREPROCESS_DICT[self.img_size]['crop_stride']

        if self.split == 'test':
            if self.crop_not_resize:
                self.data_idxs = os.listdir(os.path.join(self.data_root, f'{self.split}_merged', 'images_png'))
            else:
                self.data_idxs = torch.load(os.path.join(self.data_root, 'meta', f'test_data_idxs.pth'))
        else:
            self.class_dict = torch.load(os.path.join(self.data_root, 'meta', f'{self.split}_class_dict.pth'))
            if self.multichannel:
                # only use images that have all class in it.
                self.data_idxs = set(self.class_dict[1])
                for k, v in self.class_dict.items():
                    self.data_idxs.intersection_update(v)
                self.data_idxs = list(self.data_idxs)
            else:
                self.data_idxs = self.class_dict[self.class_id] # actually filenames not idxs.

        if self.split == 'train':
            self.data_idxs = self.data_idxs[self.support_idx*self.shot : (self.support_idx+1) * self.shot]
        elif self.split == 'valid':
            assert self.dset_size > 0
            self.data_idxs = self.data_idxs[:self.dset_size]


        if self.crop_not_resize:
            # To get images from full resolution data.
            self.image_dir = os.path.join(self.data_root, f'{self.split}_merged', 'images_png')
            self.label_dir = os.path.join(self.data_root, f'{self.split}_merged', 'masks_png')

        if self.dset_size < 0:
            self.dset_size = len(self.data_idxs)


    def compatible_init(self, kwargs):
        '''
        Do some minor hacks to inherit from DownstreamFinetuneDataset
        '''
        # temporarily disable shuffling and initialize dataset
        assert 'shuffle_idxs' not in kwargs or kwargs['shuffle_idxs'] is False, 'Shuffling indices is not yet implemented precisely'
        # also temporarily make class name = none
        self._class_name = kwargs['class_name']
        self._dset_size = kwargs['dset_size']
        kwargs['class_name'] = 'none'
        temp_dir = os.path.join(kwargs['data_root'], f'resized_{kwargs["base_size"][0]}', 'images')
        os.makedirs(temp_dir, exist_ok=True)        
        # disable split_idxs
        self.split_idxs = lambda _, uniform_split=True, test_all=False: (None, None, None)
        super().__init__(**kwargs)
        # remove temp dir if it is empty
        try:
            os.removedirs(temp_dir)
        except OSError:
            pass
        self.class_name = self._class_name
        self.dset_size = self._dset_size
        del self._class_name


    def __getitem__(self, idx):
        if self.split in ['train', 'valid']:
            img_path = self.data_idxs[idx % len(self.data_idxs)]
            image, label = self.load_data(img_path)
            Xs, Ys, Ms = self.postprocess_data(image, label)   

            if self.split == 'train':
                return Xs, Ys, Ms # CAUTION: P C H W
            elif self.split == 'valid':
                label = self.toten(label)            
                if self.precision == 'bf16':
                    label = label.to(torch.bfloat16)
                label = self.binary_label(label)

                return Xs, label, Ms # We don't need patch for label
        else: # no label for test set
            img_path = self.data_idxs[idx % len(self.data_idxs)]
            image, _ = self.load_data(img_path, skip_label=True)
            # convert to tensor
            image = image.resize(self.interp_size, Image.BILINEAR)
            # convert to tensor
            X = self.toten(image)
            if self.precision == 'bf16':
                X = X.to(torch.bfloat16)
            Xs = simple_overlapped_crop(X.unsqueeze(0), crop_size=self.base_size, stride=self.crop_stride) # L C h w
            if self.multichannel:
                return Xs, img_path, 0
            save_path = img_path.replace('.png', f'_{self.class_id}.pth')
            return Xs, save_path, 0
        

    def load_data(self, img_path, skip_label=False):
        target_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(target_path):
            target_path = target_path.replace('val_', 'valid_')
        image = Image.open(target_path).convert('RGB')        
        lbl_path = img_path.replace('images_png', 'masks_png')
        label = None
        if not skip_label:
            label_path = os.path.join(self.label_dir, lbl_path)
            if not os.path.exists(label_path):
                label_path = label_path.replace('val_', 'valid_')
            label = Image.open(label_path)

        return image, label

    def binary_label(self, Y):
        Y = (Y*255).long()
        if self.multichannel:
            Y = Y.squeeze(0) # H W
            Y = F.one_hot(Y, num_classes=8).permute(2,0,1) # (C+1) H W since first channel is ignore
            Y = Y[1:] # C H W
        else:
            Y[Y!=self.class_id] = 0
            Y[Y==self.class_id] = 1
        return Y.float()

    def postprocess_data(self, image, label):
        if self.crop_not_resize:
            # resize first to interp_size
            image = image.resize(self.interp_size, Image.BILINEAR)
            label = label.resize(self.interp_size, Image.NEAREST)
        
        # convert to tensor
        X = self.toten(image)
        Y = self.toten(label)            
        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)

        Y = self.binary_label(Y)
        if self.crop_not_resize:
            # generate patches
            X_patches = simple_overlapped_crop(X.unsqueeze(0), crop_size=self.base_size, stride=self.crop_stride) # L C h w
            Y_patches = simple_overlapped_crop(Y.unsqueeze(0), crop_size=self.base_size, stride=self.crop_stride) # L C h w

            if self.split == 'train':
                is_available = Y_patches.sum(dim=(1,2,3)) > 0 # L
                X_patches = X_patches[is_available] # P C h w; P = patches_per_img
                Y_patches = Y_patches[is_available] # P C h w
                selected_idxs = torch.randint(0, X_patches.size(0), (self.patches_per_img,))
            else:
                selected_idxs = torch.arange(X_patches.size(0))

            X_patches = X_patches[selected_idxs] # P C h w; P = patches_per_img
            Y_patches = Y_patches[selected_idxs] # P C h w
        else:
            X_patches = X.unsqueeze(0)
            Y_patches = Y.unsqueeze(0)

        # apply random crop, where at least one pixel of bounding box should remain.
        foo = []
        bar = []
        for i in range(len(Y_patches)):
            offset_cuts = None
            if self.split == 'train' and not self.multichannel:
                valid_indices = torch.where(Y_patches[i,0])
                upper_left = [valid_indices[0].min().item(), valid_indices[1].min().item()]
                lower_right = [valid_indices[0].max().item(), valid_indices[1].max().item()]
                offset_cuts = upper_left + lower_right

            first, second = crop_arrays(X_patches[i], Y_patches[i],
                                                    base_size=self.base_size,
                                                    crop_size=self.img_size,
                                                    random=(not self.fix_seed),
                                                    offset_cuts=offset_cuts)
            foo.append(first)
            bar.append(second)
        X = torch.stack(foo) # P C h w
        Y = torch.stack(bar)
        M = torch.ones_like(Y) 

        return X, Y, M


class KittiFlowFinetuneDataset(KittiFlow):
    def __init__(self, split, shot, support_idx, channel_idx, root_dir, base_size, img_size,
                 dset_size=-1, shuffle_idxs=False, fix_seed=False, train_files=False,
                 sequential=False, load_dense=False, precision='fp32'):
        super().__init__(root_dir)
        self.root_dir = root_dir
        self.base_size = base_size
        self.img_size = img_size
        self.fix_seed = fix_seed
        self.precision = precision
        self.channel_idx = channel_idx
        self.split = split
        self.shot = shot
        self.support_idx = support_idx
        self.train_files = train_files
        self.shuffle_idxs = shuffle_idxs
        self.load_dense = load_dense
        self.dense_dir = os.path.join(root_dir, 'KittiFlow', f'{split}_resized_{base_size[0]}', 'labels_interpolated')

        self.toten = ToTensor()
        if dset_size < 0:
            dset_size = len(self._image_list)
        self.dset_size = dset_size

        self.data_idxs = np.arange(len(self._image_list))
        train_idxs, valid_idxs, test_idxs = self.split_idxs(not self.shuffle_idxs,
                                                            test_all=(train_files is not None and
                                                                      (split == 'test')))
        if split == 'train':
            self.data_idxs = self.data_idxs[train_idxs]
        elif split == 'valid':
            self.data_idxs = self.data_idxs[valid_idxs]
            self.data_idxs = self.data_idxs[:-1]
        else:
            self.data_idxs = self.data_idxs[test_idxs]

        self.getitem_counter = 0
        self.return_first_im = True
        
    def __len__(self):
        return self.dset_size

    def split_idxs(self, uniform_split=True, test_all=False):
        N = len(self.data_idxs)
        shot = self.shot
        
        if test_all:
            return None, None, np.arange(N)
        
        if uniform_split:
            train_idxs = [0]
            for i in range(1, shot - 1):
                train_idxs.append((N // (shot - 1)) * i)
            if shot > 1:
                train_idxs.append(N - 1)
            assert len(train_idxs) == shot
            
            valid_idxs = [1]
            for i in range(1, self.dset_size - 1):
                valid_idxs.append((N // (self.dset_size - 1)) * i + 1)
            if self.dset_size > 1:
                valid_idxs.append(N - 2)
            assert len(valid_idxs) == self.dset_size, f'{len(valid_idxs)} != {self.dset_size}'

            test_idxs = []
            for i in range(N):
                if (i not in train_idxs):
                    test_idxs.append(i)
        else:
            offset = self.support_idx*self.shot
            train_idxs = np.array(range(offset, offset + shot))
            valid_idxs = np.array(list(range(offset)) + list(range(offset + shot, N)))[:self.dset_size]
            test_idxs = np.array(list(range(offset)) + list(range(offset + shot, N)))

        return train_idxs, valid_idxs, test_idxs

    def __getitem__(self, idx):
        self.getitem_counter = 0 if self.getitem_counter >= len(self.data_idxs) else self.getitem_counter
        idx = self.data_idxs[self.getitem_counter]
        idx = idx % len(self._image_list)
        X1, X2, Y, M = super().__getitem__(idx)
        X1 = X1.resize(self.base_size, Image.BILINEAR)
        X2 = X2.resize(self.base_size, Image.BILINEAR)
        X1 = self.toten(X1)
        X2 = self.toten(X2)
        if self.load_dense:
            U = Image.open(os.path.join(self.dense_dir, f'{idx:05d}_u.png'))
            V = Image.open(os.path.join(self.dense_dir, f'{idx:05d}_v.png'))
            U = np.asarray(U).astype(np.float32) / 255. / 256.
            V = np.asarray(V).astype(np.float32) / 255. / 256.
            Y = np.stack([U, V], axis=0)
        else:
            Y = Y / np.array(Y.shape[-2:])[:, None, None]
            Y = (Y + 1) / 2

        if self.channel_idx >= 0:
            Y = Y[self.channel_idx:self.channel_idx+1]
        M = M[None]
        Y = F.interpolate(torch.from_numpy(Y)[None], self.base_size, mode='nearest')[0]
        M = F.interpolate(torch.from_numpy(M)[None].float(), self.base_size, mode='nearest')[0]
        M = M.repeat(len(Y), 1, 1)

        X1, X2, Y, M = crop_arrays(X1, X2, Y, M,
                                   base_size=self.base_size,
                                   crop_size=self.img_size,
                                   random=(not self.fix_seed))
        # precision conversion
        if self.precision == 'fp16':
            X1 = X1.half()
            X2 = X2.half()
            Y = Y.half()
            M = M.half()

        if self.precision == 'bf16':
            X1 = X1.to(torch.bfloat16)
            X2 = X2.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
            M = M.to(torch.bfloat16)
        
        if self.return_first_im:
            X = X1; Y = Y[0].unsqueeze(0); M = M[0].unsqueeze(0)
            self.return_first_im = False
        else:
            X = X2; Y = Y[0].unsqueeze(0); M = M[1].unsqueeze(0) # (!) return same label channel
            self.return_first_im = True
            self.getitem_counter += 1
            
        return X, Y, M


class AP10KFinetuneDataset(Dataset):
    def __init__(self, split, dset_size, base_size, img_size, dataset, data_root, fix_seed=False, 
                 shot=-1, support_idx=-1, channel_idx=-1, train_files=False, sequential=False, img_ext='.jpg', lbl_ext='.png', 
                 class_name='', shuffle_idxs=False, precision='fp32', skip_crowd=False):
        if split == 'valid':
            split = 'val'
        assert split in ['train', 'val', 'test']
        self.dataset = dataset
        self.data_root = data_root
        if class_name == 'none':
            class_name = ''
        self.image_dir = os.path.join(data_root, f'images_{base_size[0]}_merged')
        self.ann_dir = os.path.join(data_root, 'annotations', f'ap10k-{split}-split1.json')
        self.class_name = class_name
        self.shuffle_idxs = shuffle_idxs
        self.precision = precision
        self.coco = SilentCOCO(self.ann_dir)
        # ['left_eye', 'right_eye', 'nose', 'neck', 'root_of_tail', 'left_shoulder', 
        # 'left_elbow', 'left_front_paw', 'right_shoulder', 'right_elbow', 'right_front_paw', 
        # 'left_hip', 'left_knee', 'left_back_paw', 'right_hip', 'right_knee', 'right_back_paw']
        self.kp_classes = list(self.coco.cats.values())[0]['keypoints']
        self.species_ids = list(i for i in range(1, 55) if i not in [7, 37, 49, 54]) # 54 species, 4 species removed (no images)
        self.class_id = self.kp_classes.index(self.class_name)
        self.toten = ToTensor()

        self.split = split
        self.channel_idx = channel_idx
        self.dset_size = dset_size
        self.base_size = base_size
        self.img_size = img_size
        self.fix_seed = fix_seed
        self.skip_crowd = skip_crowd
        self.support_idx = support_idx
        self.shot = shot

        # Some attributes related to construct continuous keypoint labels
        self.kp_sigma = 3
        self.kp_max = 1.0
        self.kp_truncate = 4.0
        self.kp_radius = round(self.kp_truncate * self.kp_sigma)
        base_array = np.zeros((2*self.kp_radius+1, 2*self.kp_radius+1))
        base_array[self.kp_radius, self.kp_radius] = 1
        self.gaussian = scipy.ndimage.gaussian_filter(base_array, sigma=self.kp_sigma, mode='constant', truncate=self.kp_truncate)
        self.gaussian = torch.tensor(self.gaussian / self.gaussian.max() * self.kp_max)

        # Get only images with at least one visible keypoint with target class.
        class_dict_path = os.path.join(self.data_root, 'meta', f'{self.split}_class_dict.pth')
        if self.skip_crowd:
            class_dict_path = os.path.join(self.data_root, 'meta', f'{self.split}_class_dict_skip_crowd.pth')
        if os.path.exists(class_dict_path):
            self.class_dict = torch.load(class_dict_path)
        elif split in ['train', 'val']:
            # If not exist, make one.
            os.makedirs(os.path.dirname(class_dict_path), exist_ok=True)
            self.class_dict = {i:{j:set([]) for j in self.species_ids} for i in range(len(self.kp_classes))} # 54 species
            # To skip crowded images
            skip_imgIds = set()
            if self.skip_crowd:
                skip_imgIds = set(imgId for imgId in self.coco.getImgIds() if len(self.coco.getAnnIds(imgIds=imgId)) != 1)

            for j in self.species_ids: # 54 species
                annIds = self.coco.getAnnIds(catIds=j)
                anns = self.coco.loadAnns(annIds)
                for ann in anns:
                    for c in range(len(self.kp_classes)):
                        if ann['keypoints'][c*3+2] > 0:
                            imgId = ann['image_id']
                            if imgId not in skip_imgIds:
                                self.class_dict[c][j].add(imgId) # add to set so no worries about duplicates
            # to nparray
            for c in range(len(self.kp_classes)):
                for j in self.species_ids:
                    self.class_dict[c][j] = np.array(list(self.class_dict[c][j]))
            # Shuffle
            for c in range(len(self.kp_classes)):
                np.random.shuffle(self.class_dict[c][j])
            torch.save(self.class_dict, class_dict_path)
        else: # test
            idx_all = sorted(self.coco.getImgIds())
            if self.skip_crowd:
                idx_all = [idx for idx in idx_all if len(self.coco.getAnnIds(imgIds=idx)) == 1]
            self.class_dict = {i:{None:idx_all} for i in range(len(self.kp_classes))}
            self.class_dict = {i:{None: self.class_dict[i][None][:int(len(self.class_dict[i][None])*0.1)]} for i in self.class_dict}


        self.data_idxs = []
        for bundle in zip_longest(*self.class_dict[self.class_id].values(), fillvalue=-1):
            self.data_idxs += list(bundle) # this will cut-off the images, based on the species with least images.
        self.data_idxs = np.array([i for i in self.data_idxs if i != -1])
        if self.split == 'train':
            self.data_idxs = self.data_idxs[self.support_idx*self.shot:(self.support_idx + 1)*self.shot]

        if shuffle_idxs:
            perm_path = f'dataset/meta_info/idxs_perm_{dataset}.pth'
            if os.path.exists(perm_path):
                self.idxs_perm = torch.load(perm_path)
            else:
                self.idxs_perm = torch.randperm(len(self.data_idxs))
                torch.save(self.idxs_perm, perm_path)
            self.data_idxs = self.data_idxs[self.idxs_perm]
        

        if self.dset_size < 0:
            self.dset_size = len(self.data_idxs)


    def __len__(self):
        return self.dset_size
    

    def load_image(self, imgId):
        '''
        get image idx and return tensor
        '''
        # get path
        file_name = self.coco.imgs[imgId]['file_name']
        img_path = os.path.join(self.image_dir, file_name)
        
        # open image file
        img = Image.open(img_path)
        img = self.toten(img)
        if len(img) == 1:
            img = img.repeat(3, 1, 1)
        
        return img

    def load_label(self, imgId, class_id=None):
        '''
        return torch Tensor
        '''
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
                if i == self.class_id and v == 2:
                    kernel[h, w] = 1 # maybe in future I should modify here to make it multi-class

        label = torch.from_numpy(kernel).unsqueeze(0)
        ret_label = torch.zeros_like(label)
        # elif task_group == 'keypoints_multiclass':
            # target_pos = label.nonzero()
            # for (c, h, w) in target_pos.tolist():
            #     # enlarge the single pixel label to 3x3 box.
            #     h_start = max(0, h-1)
            #     h_end = min(label.shape[-2], h+2)
            #     w_start = max(0, w-1)
            #     w_end = min(label.shape[-1], w+2)
            #     ret_label[c, h_start:h_end, w_start:w_end] = label[c, h, w] # this may lead to overwriting other labels
        assert class_id is not None
        target_pos = label.nonzero()
        # pad ret_label to properly handle boundary
        ret_label = F.pad(ret_label, (self.kp_radius, self.kp_radius, self.kp_radius, self.kp_radius), mode='constant', value=0)
        for (c, h, w) in target_pos.tolist():
            # if label[c, h, w] == (self.class_id / len(self.kp_classes)):
            h_start = h
            h_end = h + 2*self.kp_radius + 1
            w_start = w
            w_end = w + 2*self.kp_radius + 1
            ret_label[c, h_start:h_end, w_start:w_end] += self.gaussian
        ret_label = torch.clip(ret_label, 0, 1) # add-and-clip
        # remove padding
        ret_label = ret_label[:, self.kp_radius:-self.kp_radius, self.kp_radius:-self.kp_radius]
        label = ret_label

        return label


    def load_data(self, imgId):
        image = self.load_image(imgId)
        label = self.load_label(imgId=imgId, class_id=self.class_id)
                
        return image, label


    def postprocess_data(self, image, label):
        X, Y = image, label
        
        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
        
        if self.channel_idx >= 0:
            Y = Y[self.channel_idx:self.channel_idx+1]

        X, Y = crop_arrays(X, Y,
                           base_size=X.size()[-2:],
                           crop_size=self.img_size,
                           random=(not self.fix_seed))
        M = torch.ones_like(Y)
            
        return X, Y, M

    def __getitem__(self, idx):
        imgId = self.data_idxs[idx % len(self.data_idxs)]
        image, label = self.load_data(imgId)
        X, Y, M = self.postprocess_data(image, label)
        if self.split == 'train':
            return X, Y, M
        else:
            return X, Y, M, imgId # also return imgId (used later for evaluation)


class RainFinetuneDataset(DownstreamFinetuneDataset):
    '''
    Since we only have train-test split, we use train.txt to split train and val
    '''
    def __init__(self, *args, **kwargs):
        # temporarily disable getting data idx by train files.
        train_files = kwargs['train_files']
        assert train_files and not kwargs.get('shuffle_idxs', False), 'Check input arguments again.'
        kwargs['train_files'] = False
        dset_size = kwargs['dset_size']
        super().__init__(*args, **kwargs)

        # Now assume train_files=True
        self.data_idxs = np.array(sorted(os.listdir(self.image_dir)))
        file_path = os.path.join(self.data_root, 'meta', self.class_name, 'train.txt')
        with open(file_path, 'r') as f:
            train_files = [line.strip().split('/')[-1] for line in f.readlines()]
        # modification here.
        if self.split in ['train', 'valid']:
            self.data_idxs = [data_idx for data_idx in self.data_idxs
                            if data_idx in train_files]
        else:
            self.data_idxs = [data_idx for data_idx in self.data_idxs
                            if data_idx not in train_files]
        self.data_idxs = np.array(self.data_idxs)

        train_idxs, valid_idxs, test_idxs = self.split_idxs(uniform_split=False,
                                                            test_all=(train_files is not None and
                                                                      (self.split == 'test')))
        if self.split == 'train':
            self.data_idxs = self.data_idxs[train_idxs]
        elif self.split == 'valid':
            self.data_idxs = self.data_idxs[valid_idxs]
        else:
            self.data_idxs = self.data_idxs[test_idxs]

        if dset_size < 0:
            self.dset_size = len(self.data_idxs)


    def load_data(self, img_path):
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        
        lbl_path = img_path.replace(self.img_ext, self.lbl_ext)
        if self.lbl_ext == '.npy':
            label = np.load(os.path.join(self.label_dir, lbl_path))
            if label.ndim == 2:
                label = label[:, :, None]
        else:
            label = Image.open(os.path.join(self.label_dir, lbl_path))
        
        return image, label
    

    def __getitem__(self, idx):
        img_path = self.data_idxs[idx % len(self.data_idxs)]
        image, label = self.load_data(img_path)
        
        return super().postprocess_data(image, label)


class EigenFinetuneDataset(Dataset):
    '''
    Downstream finetune dataset for KITTI Eigen split depth estimation.
    '''
    def __init__(self, split, dset_size, base_size, img_size, dataset, data_root, fix_seed=False, 
                 shot=-1, support_idx=-1, channel_idx=-1, train_files=False, sequential=False, img_ext='.jpg', lbl_ext='.png', 
                 class_name='', shuffle_idxs=False, precision='fp32'):
        assert split in ['train', 'valid', 'test']

        self.dataset = dataset
        self.data_root = data_root
        if class_name == 'none':
            class_name = ''

        if split == 'train' or split == 'valid':
            file_list = os.path.join(data_root, 'eigen_train_files_with_gt.txt')
        else:
            file_list = os.path.join(data_root, 'eigen_test_files_with_gt.txt')

        with open(file_list, 'r') as f:
            filenames = f.readlines()

        self.image_paths = []
        self.label_paths = []
        for line in filenames:
            rgb_file, depth_file, *_ = line.split()
            if depth_file == 'None':
                continue
            self.image_paths.append(rgb_file)
            self.label_paths.append(depth_file)


        if split == 'train' or split == 'valid':
            self.image_dir = os.path.join(data_root, f'train_resized_{base_size[0]}', 'images')
            self.ann_dir = os.path.join(data_root, f'train_resized_{base_size[0]}', 'labels')
            self.full_res_dir = os.path.join(data_root, 'train_kb_cropped')
            if split == 'train':
                self.ann_inter_dir = os.path.join(data_root, f'train_resized_{base_size[0]}', 'labels_interpolated')
            else:
                self.ann_inter_dir = None
        else:
            self.image_dir = os.path.join(data_root, f'test_resized_{base_size[0]}', 'images')
            self.ann_dir = os.path.join(data_root, f'test_resized_{base_size[0]}', 'labels')
            self.ann_inter_dir  = None
            self.full_res_dir = os.path.join(data_root, 'test_kb_cropped')

        self.class_name = class_name
        self.shuffle_idxs = shuffle_idxs
        self.precision = precision
        self.toten = ToTensor()

        self.split = split
        self.shot = shot
        self.support_idx = support_idx
        self.channel_idx = channel_idx
        self.dset_size = dset_size
        self.base_size = base_size
        self.img_size = img_size
        self.fix_seed = fix_seed

        self.data_idxs = np.arange(len(self.image_paths))
        if shuffle_idxs:
            perm_path = f'dataset/meta_info/idxs_perm_{dataset}.pth'
            if os.path.exists(perm_path):
                self.idxs_perm = torch.load(perm_path)
            else:
                self.idxs_perm = torch.randperm(len(self.data_idxs))
                torch.save(self.idxs_perm, perm_path)
            self.data_idxs = self.data_idxs[self.idxs_perm]

        # Make Split
        self.data_idxs = self.split_idxs(uniform_split=(not self.shuffle_idxs))
        
        # Kitti Specific Values
        self.max_depth = 80
        self.min_depth = 1e-3 # only for eval
        
        if self.dset_size < 0:
            self.dset_size = len(self.data_idxs)


    def __len__(self):
        return self.dset_size
    

    def load_image(self, idx, full_res=False):
        '''
        get image idx and return tensor
        '''
        # get path
        if full_res:
            image_path = os.path.join(self.full_res_dir, 'images', self.image_paths[idx])
        else:
            image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path)
        image = np.asarray(image, dtype=np.float32) / 255.0 # H W C
        image = self.toten(image) # C H W

        return image


    def load_label(self, idx, full_res=False):
        '''
        return torch Tensor
        '''
        # get path
        if full_res:
            label_path = os.path.join(self.full_res_dir, 'labels', self.label_paths[idx])
        else:
            label_path = os.path.join(self.ann_dir, self.label_paths[idx])
            if self.ann_inter_dir is not None:
                label_inter_path = os.path.join(self.ann_inter_dir, self.label_paths[idx].replace('/', '_'))

        label = Image.open(label_path)
        label = np.asarray(label, dtype=np.float32)
        label = np.expand_dims(label, axis=2)

        label = np.asarray(label, dtype=np.float32) / 256.0 # H W nC
        label = self.toten(label) # C H W 

        if (not full_res) and (self.ann_inter_dir is not None):
            label_dense = Image.open(label_inter_path)
            label_dense = np.asarray(label_dense, dtype=np.float32)
            label_dense = np.expand_dims(label_dense, axis=2)
            # label_dense = np.asarray(label_dense, dtype=np.float32) / 256.0 / 255.

            # convert to original scale
            label_dense = label_dense / 255. / 256.
            label_dense = label_dense*23007 + 506
            label_dense = label_dense.clip(1, 20480)

            # convert to log scale
            label_dense = (np.log(label_dense) - np.log(506)) / (np.log(23153) - np.log(506))
            label_dense = np.where(np.asarray(label_dense) == 0, np.zeros_like(label_dense), label_dense)
            label_dense = self.toten(label_dense)
        else:
            label_dense = None

        return label, label_dense


    def load_data(self, idx, full_res=False):
        image = self.load_image(idx, full_res=full_res)
        label, label_dense = self.load_label(idx, full_res=full_res)
                
        return image, label, label_dense
    

    def postprocess_data(self, image, label, Y_dense=None, eval_mode=False):
        X, Y = image, label
        if not eval_mode:
            if Y_dense is not None:
                X, Y, Y_dense = crop_arrays(X, Y, Y_dense,
                                base_size=X.size()[-2:],
                                crop_size=self.img_size,
                                random=(not self.fix_seed))
            else:
                X, Y = crop_arrays(X, Y,
                                base_size=X.size()[-2:],
                                crop_size=self.img_size,
                                random=(not self.fix_seed))

        # Normalize
        if eval_mode:
            M = torch.logical_and((Y>self.min_depth), (Y<self.max_depth)).float()
        else:
            M = torch.logical_and((Y>0.), (Y<self.max_depth)).float()
            if Y_dense is not None:
                Y = Y_dense
              
        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
            M = M.to(torch.bfloat16)

        assert self.channel_idx < 0, self.channel_idx        
        # if self.channel_idx >= 0:
        #     Y = Y[self.channel_idx:self.channel_idx+1]

        return X, Y, M


    def __getitem__(self, idx):
        idx = self.data_idxs[idx % len(self.data_idxs)]
        image, label, label_dense = self.load_data(idx)
        X, Y, M = self.postprocess_data(image, label, label_dense)
        if self.split == 'valid' or self.split == 'test':
            full_image, full_label, _ = self.load_data(idx, full_res=True)
            aux = self.postprocess_data(full_image, full_label, eval_mode=True)

        if self.split == 'train':
            return X, Y, M
        else:
            return X, Y, M, aux # also return full resolution gt for evaluation


    def split_idxs(self, uniform_split=True):
        N = len(self.data_idxs)
        shot = self.shot

        if self.split == 'test':
            return np.arange(N)

        if uniform_split:
            if self.split == 'train':
                train_idxs = [0]
                for i in range(1, shot - 1):
                    train_idxs.append((N // (shot - 1)) * i)
                if shot > 1:
                    train_idxs.append(N - 1)
                assert len(train_idxs) == shot

                train_idxs = [(i + self.support_idx*self.shot) % N for i in train_idxs]

                return train_idxs
            
            elif self.split == 'valid':
                valid_idxs = [1]
                for i in range(1, self.dset_size - 1):
                    valid_idxs.append((N // (self.dset_size - 1)) * i + 1)
                if self.dset_size > 1:
                    valid_idxs.append(N - 2)

                valid_idxs = [(i + self.support_idx*self.shot) % N for i in valid_idxs]
                assert len(valid_idxs) == self.dset_size, f'{len(valid_idxs)} != {self.dset_size}'
                return valid_idxs

        else:
            offset = self.support_idx * self.shot
            if self.split == 'train':
                return np.array(range(offset, offset + shot))
            else:
                return np.array(list(range(offset)) + list(range(offset + shot, N)))[:self.dset_size]



# class DomainBaseDataset(DownstreamBaseDataset):
#     # Task Groups
#     TASK_GROUPS_BASE = []

#     # Base Tasks
#     TASKS_AUTOENCODING = [f'autoencoding_{c}' for c in range(3)]
#     TASK_GROUPS_BASE.append(TASKS_AUTOENCODING)

#     TASKS_DENOISING = [f'denoising_{c}' for c in range(3)]
#     TASK_GROUPS_BASE.append(TASKS_DENOISING)

#     TASKS_EDGE2D = [f'edge_texture_{c}' for c in range(3)]
#     TASK_GROUPS_BASE.append(TASKS_EDGE2D)

#     # Task Group Names
#     TASK_GROUP_NAMES_BASE = ['_'.join(task_group[0].split('_')[:-1])
#                              for task_group in TASK_GROUPS_BASE]
#     TASK_GROUP_NAMES = TASK_GROUP_NAMES_BASE

#     # All Task Groups, Task Group Dict, and Tasks
#     TASK_GROUPS = TASK_GROUPS_BASE
#     TASK_GROUP_DICT = {name: group for name, group in zip(TASK_GROUP_NAMES, TASK_GROUPS)}

#     TASKS_BASE = [task for task_group in TASK_GROUPS_BASE for task in task_group]
#     TASKS = TASKS_BASE

#     def __init__(self, dset_size, shot, unary_augmentation, binary_augmentation,
#                  *args, crop_size=(224, 224), meta_dir='meta_info', seed=None, **kwargs):
        
#     def __init__(self, dataset, data_root, base_size, img_ext='.jpg', lbl_ext='.png',
#                  class_name='', shuffle_idxs=False, precision='fp32'):
#         super().__init__()
#         self.dataset = dataset
#         self.data_root = data_root
#         if class_name == 'none':
#             class_name = ''
#         self.image_dir = os.path.join(data_root, f'resized_{base_size[1]}', class_name, 'images')
#         self.label_dir = os.path.join(data_root, f'resized_{base_size[0]}', class_name, 'labels')
#         self.img_ext = img_ext
#         self.lbl_ext = lbl_ext
#         self.class_name = class_name
#         self.shuffle_idxs = shuffle_idxs
#         self.precision = precision
#         self.base_size = base_size
        
#         self.data_idxs = np.array(sorted(os.listdir(self.image_dir)))
#         if shuffle_idxs:
#             perm_path = f'dataset/meta_info/idxs_perm_{dataset}.pth'
#             if os.path.exists(perm_path):
#                 self.idxs_perm = torch.load(perm_path)
#             else:
#                 self.idxs_perm = torch.randperm(len(self.data_idxs))
#                 torch.save(self.idxs_perm, perm_path)
#             self.data_idxs = self.data_idxs[self.idxs_perm]
        
#         self.toten = ToTensor()
        
#     def __init__(self, unary_augmentation, binary_augmentation, path_dict, split,
#                  base_size=(256, 256), crop_size=(224, 224), seed=None, precision='fp32', meta_dir='meta_info'):
#         super().__init__(*args, **kwargs)
#         if seed is not None:
#             random.seed(seed)
#             np.random.seed(seed)
#             torch.manual_seed(seed)
#             torch.cuda.manual_seed(seed)

#         self.dset_size = dset_size
#         self.shot = shot
#         self.crop_size = crop_size

#         self.meta_info_path = os.path.join('dataset', meta_dir, 'taskonomy')
            
#         # register sobel edge detectors
#         self.edge_params = torch.load(os.path.join(self.meta_info_path, 'edge_params.pth'))
#         self.sobel_detectors = [SobelEdgeDetector(kernel_size=k, sigma=s) for k, s in self.edge_params['params']]

#         self.unary_augmentation = get_filtering_augmentation() if unary_augmentation else None
#         self.binary_augmentation = Mixup() if binary_augmentation else None
#         self.tasks = self.TASKS_BASE
        
#     def _postprocess_autoencoding(self, imgs, channels):
#         labels = imgs.clone()
#         masks = torch.ones_like(labels)
            
#         labels = labels[:, channels]
#         masks = masks[:, channels]

#         return labels, masks

#     def _postprocess_denoising(self, imgs, channels):
#         labels = imgs.clone()
#         masks = torch.ones_like(labels)
#         imgs = torch.from_numpy(skimage.util.random_noise(imgs, var=0.01))
            
#         labels = labels[:, channels]
#         masks = masks[:, channels]

#         return imgs, labels, masks
    
#     def _postprocess_edge_texture(self, imgs, channels):
#         labels = []
#         # detect sobel edge with different set of pre-defined parameters
#         for c in channels:
#             labels_ = self.sobel_detectors[c].detect(imgs)
#             labels.append(labels_)
#         labels = torch.cat(labels, 1)

#         # thresholding and re-normalizing
#         labels = torch.clip(labels, 0, self.edge_params['threshold'])
#         labels = labels / self.edge_params['threshold']

#         masks = torch.ones_like(labels)
        
#         return labels, masks

#     def _postprocess(self, imgs, channels, task):
#         # task-specific preprocessing
#         if task == 'autoencoding':
#             labels, masks = self._postprocess_autoencoding(imgs, channels)
#         elif task == 'denoising':
#             imgs, labels, masks = self._postprocess_denoising(imgs, channels)
#         elif task == 'edge_texture':
#             labels, masks = self._postprocess_edge_texture(imgs, channels)
#         else:
#             raise NotImplementedError

#         # ensure label values to be in [0, 1]
#         labels = labels.clip(0, 1)

#         return imgs, labels, masks
    
#     def _load_image(self, img_file):
#         image = Image.open(os.path.join(self.image_dir, img_file)).convert('RGB')
#         x = self.toten(image)
#         return x
  
#     def load_images(self, file_idxs):
#         # load images
#         imgs = []
#         for file_idx in file_idxs:
#             # index image path
#             img_file = self.data_idxs[file_idx]

#             # load image
#             img = self._load_image(img_file)
#             imgs.append(img)
#         imgs = torch.stack(imgs)

#         return imgs
    
#     def load_labels(self, imgs, task, channels):
#         imgs, labels, masks = self._postprocess(imgs, channels, task)
        
#         # precision conversion
#         if self.precision == 'fp16':
#             imgs = imgs.half()
#             labels = labels.half()
#             masks = masks.half()
#         elif self.precision == 'bf16':
#             imgs = imgs.bfloat16()
#             labels = labels.bfloat16()
#             masks = masks.bfloat16()

#         return imgs, labels, masks
        
#     def sample_task(self, from_all=False):
#         if from_all:
#             task = np.random.choice(self.TASKS)
#         else:
#             task = np.random.choice(self.tasks)
#         task_group, channel = '_'.join(task.split('_')[:-1]), int(task.split('_')[-1])
#         t_idx = torch.tensor(self.TASKS.index(task))

#         return task_group, channel, t_idx
    
#     def sample_files(self, n_files):
#         # sample image file indices
#         file_idxs = np.random.choice(self.file_idxs, n_files, replace=False)
#         return file_idxs
 
#     def sample_episode(self, n_files, n_domains=None):
#         # sample tasks
#         task_group, channel, t_idx = self.sample_task()
#         if self.binary_augmentation is not None:
#             task_group_aux, channel_aux, _ = self.sample_task(from_all=True)

#         # sample image paths
#         file_idxs = self.sample_files(n_files)

#         # load images
#         imgs = self.load_images(file_idxs)

#         # load labels
#         X, Y, M = self.load_labels(imgs, task_group, [channel], file_idxs)
#         if self.binary_augmentation:
#             _, Y_aux, M_aux = self.load_labels(imgs, task_group_aux, [channel_aux], file_idxs)

#         # apply unary task augmentation
#         if self.unary_augmentation:
#             Y, M = self.unary_augmentation(Y, M)
#             if self.binary_augmentation:
#                 Y_aux, M_aux = self.unary_augmentation(Y_aux, M_aux)

#         # apply binary taks augmentation
#         if self.binary_augmentation:
#             Y, M = self.binary_augmentation(Y, Y_aux, M, M_aux)
        
#         # random-crop arrays
#         X, Y, M = crop_arrays(X, Y, M,
#                               base_size=self.base_size,
#                               crop_size=self.crop_size,
#                               random=True)
        
#         return X, Y, M, t_idx
    

class DownstreamUnsupervisedDataset(DownstreamBaseDataset):
    def __init__(self, dset_size, shot, crop_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dset_size = dset_size
        self.shot = shot
        self.crop_size = crop_size

    def __len__(self):
        return self.dset_size
    
    def __getitem__(self, idx):
        idxs = np.random.choice(len(self.data_idxs), 2*self.shot, replace=False)

        # load images
        X = []
        for idx in idxs:
            img_path = self.data_idxs[idx % len(self.data_idxs)]
            image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
            x = self.toten(image)
            X.append(x)
        X = torch.stack(X, dim=0)

        # crop arrays
        X = crop_arrays(X,
                        base_size=self.base_size,
                        crop_size=self.crop_size,
                        random=True)[0]

        # create empty labels and masks
        Y = torch.empty(X.shape[0], 1, X.shape[2], X.shape[3])
        M = torch.empty(X.shape[0], 1, X.shape[2], X.shape[3])
        t_idx = torch.tensor(-1)

        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)
            M = M.to(torch.bfloat16)

        return X[None], Y[None], M[None], t_idx[None]
