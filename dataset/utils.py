from typing import List, Tuple
from einops import rearrange
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from scipy import ndimage, signal
# import pydensecrf.utils as putils
# import pydensecrf.densecrf as dcrf
import math


################# Crop Tools ####################
def crop_arrays(*arrays, base_size=(256, 256), crop_size=(224, 224), random=True, get_offsets=False, offset_cuts=None):
    '''
    Crop arrays from base_size to img_size.
    Apply center crop if not random.
    '''
    if isinstance(base_size, int):
        base_size = (base_size, base_size)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    if base_size[0] == crop_size[0] and base_size[1] == crop_size[1]:
        if get_offsets:
            return arrays, (0, 0)
        else:
            return arrays

    # TODO: check offset_cuts in LMFinetuneDataset and refactor here.
    if random and offset_cuts is not None and len(offset_cuts) == 2:
        off_H = np.random.randint(min(base_size[0] - crop_size[0],  offset_cuts[0]))
        off_W = np.random.randint(min(base_size[1] - crop_size[1],  offset_cuts[1]))

    elif random:
        min_H = 0
        min_W = 0
        max_H = base_size[0] - crop_size[0]
        max_W = base_size[1] - crop_size[1]
        if offset_cuts is not None:
            assert len(offset_cuts) == 4
            h_min, w_min, h_max, w_max = offset_cuts
            min_H = max(min_H, h_min - crop_size[0])
            min_W = max(min_W, w_min - crop_size[1])
            max_H = min(max_H, h_max)
            max_W = min(max_W, w_max)
            
        off_H = np.random.randint(max_H - min_H + 1) + min_H
        off_W = np.random.randint(max_W - min_W + 1) + min_W

    else:
        if offset_cuts is not None:
            off_H = min(
                max(0, offset_cuts[0] - (base_size[0] - crop_size[0]) // 2),
                (base_size[0] - crop_size[0]) // 2
            )
            off_W = min(
                max(0, offset_cuts[1] - (base_size[1] - crop_size[1]) // 2),
                (base_size[1] - crop_size[1]) // 2
            )
        else:
            off_H = (base_size[0] - crop_size[0]) // 2
            off_W = (base_size[1] - crop_size[1]) // 2

    slice_H = slice(off_H, off_H + crop_size[0])
    slice_W = slice(off_W, off_W + crop_size[1])

    arrays_cropped = []
    for array in arrays:
        if array is not None:
            assert array.ndim >= 2
            array_cropped = array[..., slice_H, slice_W]
            arrays_cropped.append(array_cropped)
        else:
            arrays_cropped.append(array)

    if get_offsets:
        return arrays_cropped, (off_H, off_W)
    else:
        return arrays_cropped

def simple_overlapped_crop(array, crop_size=(256, 256), stride=(128, 128)):
    '''
    Crop array with stride.
    array: (B, C, H, W)
    Returns (B*num_patches, C, crop_size[0], crop_size[1])
    '''
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    
    B, C, H, W = array.shape
    assert stride[0] <= crop_size[0] and stride[1] <= crop_size[1]
    cropped = torch.nn.Unfold(kernel_size=crop_size, stride=stride)(array) # (B, C*crop_size[0]*crop_size[1], num_patches)
    cropped = rearrange(cropped, 'b (c h w) n -> (b n) c h w', c=C, h=crop_size[0], w=crop_size[1])  

    return cropped


def simple_mix_overlapped_crop(array, crop_size=(256, 256), output_size=(512,512), stride=(128, 128)):
    '''
    Mix cropped array stride.
    For overlapping regions, mean is used.
    array: (B, C*H*W, L)
    Returns (B, C, output_size[0], output_size[1])
    '''
    folder = torch.nn.Fold(output_size, crop_size, stride=stride)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    denominator = folder(torch.ones_like(array))
    numerator = folder(array)

    return numerator / denominator



def overlapped_crop(array, crop_size=(224, 224), stride=(112, 112)):
    '''
    NOTE: Currently this is hard-coded for 1024x1024 input.
    Crop array with stride.
    array: (B, C, H, W)
    Returns (B, C, crop_size[0], crop_size[1], L)
    '''
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    
    B, C, H, W = array.shape
    L = 9*9 # HARDCODED
    h, w = crop_size
    dh, dw = stride
    assert H == W == 1024, 'I assume the input is 1024x1024, but got {}x{}'.format(H, W)
    assert h == w == 224, 'I assume the crop_size is 224x224, but got {}x{}'.format(h, w)
    assert dh == dw == 112, 'I assume the stride is 112x112, but got {}x{}'.format(dh, dw)
    cropped = torch.zeros(B, C, h, w, L, device=array.device)
    # Below code is really dirty, but nothing wierd is done here.
    # We just crop the full image with sliding window, and the boundary(rightmost and bottommost)
    # pathces are carefully handled by allowing more overlapped region (or in other word smaller stride).
    for h_start in range(0, 896, dh):
        h_end = h_start + h
        for w_start in range(0, 896, dw):
            w_end = w_start + w
            cropped[..., h_start//dh*9 + w_start//dw] = array[..., h_start:h_end, w_start:w_end]
        # Remaining region
        w_start = 800
        w_end = 1024
        cropped[..., h_start//dh*9 + 8] = array[..., h_start:h_end, w_start:w_end]
    h_start = 800
    h_end = h_start + h
    for w_start in range(0, 896, dw):
        w_end = w_start + w
        cropped[..., 8*9 + w_start//dw] = array[..., h_start:h_end, w_start:w_end]
    w_start = 800
    w_end = 1024
    cropped[..., 8*9 + 8] = array[..., h_start:h_end, w_start:w_end]

    return cropped


def mix_overlapped_crop(array, crop_size=(224, 224), output_size=(1024,1024), stride=(112, 112)):
    '''
    NOTE: Currently this is hard-coded for 1024x1024 input.
    Mix cropped array stride.
    For overlapping regions, mean is used.
    array: (B, C, h, w, L)
    Returns (B, C, H, W)
    '''
    # folder = torch.nn.Fold(output_size, crop_size, stride=stride)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    # denominator = folder(torch.ones_like(array))
    # numerator = folder(array)

    B, C, h, w, L = array.shape
    H, W = output_size
    assert (h, w) == crop_size
    dh, dw = stride
    mixed = torch.zeros(B, C, H, W, device=array.device)
    denom = torch.zeros(B, C, H, W, device=array.device)
    # Below code is really dirty, but this is just a reverse operation of overlapped crop.
    for h_start in range(0, 896, dh):
        h_end = h_start + h
        for w_start in range(0, 896, dw):
            w_end = w_start + w
            mixed[..., h_start:h_end, w_start:w_end] += array[..., h_start//dh*9 + w_start//dw]
            denom[..., h_start:h_end, w_start:w_end] += 1
        # Remaining region
        w_start = 800
        w_end = 1024
        mixed[..., h_start:h_end, w_start:w_end] += array[..., h_start//dh*9 + 8]
        denom[..., h_start:h_end, w_start:w_end] += 1
    h_start = 800
    h_end = h_start + h
    for w_start in range(0, 896, dw):
        w_end = w_start + w
        mixed[..., h_start:h_end, w_start:w_end] += array[..., 8*9 + w_start//dw]
        denom[..., h_start:h_end, w_start:w_end] += 1
    w_start = 800
    w_end = 1024
    mixed[..., h_start:h_end, w_start:w_end] += array[..., 8*9 + 8]
    denom[..., h_start:h_end, w_start:w_end] += 1

    return mixed / denom


def mix_fivecrop(x_crop, base_size=256, crop_size=224):
    margin = base_size - crop_size
    submargin = margin // 2
    
    ### Five-pad each crops
    pads = [
        T.Pad((0, 0, margin, margin)),
        T.Pad((margin, 0, 0, margin)),
        T.Pad((0, margin, margin, 0)),
        T.Pad((margin, margin, 0, 0)),
        T.Pad((submargin, submargin, submargin, submargin)),
    ]
    x_pad = []
    for x_, pad in zip(x_crop, pads):
        x_pad.append(pad(x_))
    x_pad = torch.stack(x_pad)

    x_avg = torch.zeros_like(x_pad[0])

    ### Mix padded crops
    # top-left corner
    x_avg[:, :, :submargin, :margin] = x_pad[0][:, :, :submargin, :margin]
    x_avg[:, :, submargin:margin, :submargin] = x_pad[0][:, :, submargin:margin, :submargin]
    x_avg[:, :, submargin:margin, submargin:margin] = (x_pad[0][:, :, submargin:margin, submargin:margin] + \
                                                       x_pad[4][:, :, submargin:margin, submargin:margin]) / 2

    # top-right corner
    x_avg[:, :, :submargin, -margin:] = x_pad[1][:, :, :submargin, -margin:]
    x_avg[:, :, submargin:margin, -submargin:] = x_pad[1][:, :, submargin:margin, -submargin:]
    x_avg[:, :, submargin:margin, -margin:-submargin] = (x_pad[1][:, :, submargin:margin, -margin:-submargin] + \
                                                         x_pad[4][:, :, submargin:margin, -margin:-submargin]) / 2

    # bottom-left corner
    x_avg[:, :, -submargin:, :margin] = x_pad[2][:, :, -submargin:, :margin]
    x_avg[:, :, -margin:-submargin:, :submargin] = x_pad[2][:, :, -margin:-submargin, :submargin]
    x_avg[:, :, -margin:-submargin, submargin:margin] = (x_pad[2][:, :, -margin:-submargin, submargin:margin] + \
                                                         x_pad[4][:, :, -margin:-submargin, submargin:margin]) / 2

    # bottom-left corner
    x_avg[:, :, -submargin:, -margin:] = x_pad[3][:, :, -submargin:, -margin:]
    x_avg[:, :, -margin:-submargin, -submargin:] = x_pad[3][:, :, -margin:-submargin, -submargin:]
    x_avg[:, :, -margin:-submargin, -margin:-submargin] = (x_pad[3][:, :, -margin:-submargin, -margin:-submargin] + \
                                                           x_pad[4][:, :, -margin:-submargin, -margin:-submargin]) / 2

    # top side
    x_avg[:, :, :submargin, margin:-margin] = (x_pad[0][:, :, :submargin, margin:-margin] + \
                                               x_pad[1][:, :, :submargin, margin:-margin]) / 2
    x_avg[:, :, submargin:margin, margin:-margin] = (x_pad[0][:, :, submargin:margin, margin:-margin] + \
                                                     x_pad[1][:, :, submargin:margin, margin:-margin] + \
                                                     x_pad[4][:, :, submargin:margin, margin:-margin]) / 3

    # right side
    x_avg[:, :, margin:-margin, -submargin:] = (x_pad[1][:, :, margin:-margin, -submargin:] + \
                                                x_pad[3][:, :, margin:-margin, -submargin:]) / 2
    x_avg[:, :, margin:-margin, -margin:-submargin] = (x_pad[1][:, :, margin:-margin, -margin:-submargin] + \
                                                       x_pad[3][:, :, margin:-margin, -margin:-submargin] + \
                                                       x_pad[4][:, :, margin:-margin, -margin:-submargin]) / 3

    # bottom side
    x_avg[:, :, -submargin:, margin:-margin] = (x_pad[2][:, :, -submargin:, margin:-margin] + \
                                                x_pad[3][:, :, -submargin:, margin:-margin]) / 2
    x_avg[:, :, -margin:-submargin:, margin:-margin] = (x_pad[2][:, :, -margin:-submargin, margin:-margin] + \
                                                        x_pad[3][:, :, -margin:-submargin, margin:-margin] + \
                                                        x_pad[4][:, :, -margin:-submargin, margin:-margin]) / 3

    # left side
    x_avg[:, :, margin:-margin, :submargin] = (x_pad[0][:, :, margin:-margin, :submargin] + \
                                               x_pad[2][:, :, margin:-margin, :submargin]) / 2
    x_avg[:, :, margin:-margin, submargin:margin] = (x_pad[0][:, :, margin:-margin, submargin:margin] + \
                                                     x_pad[2][:, :, margin:-margin, submargin:margin] + \
                                                     x_pad[4][:, :, margin:-margin, submargin:margin]) / 3

    # center
    x_avg[:, :, margin:-margin, margin:-margin] = (x_pad[0][:, :, margin:-margin, margin:-margin] + \
                                                   x_pad[1][:, :, margin:-margin, margin:-margin] + \
                                                   x_pad[2][:, :, margin:-margin, margin:-margin] + \
                                                   x_pad[3][:, :, margin:-margin, margin:-margin] + \
                                                   x_pad[4][:, :, margin:-margin, margin:-margin]) / 5
    
    return x_avg


def to_device(data, device=None, dtype=None):
    '''
    Load data with arbitrary structure on device.
    '''
    def to_device_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)
        elif isinstance(data, tuple):
            return tuple(map(to_device_wrapper, data))
        elif isinstance(data, list):
            return list(map(to_device_wrapper, data))
        elif isinstance(data, dict):
            return {key: to_device_wrapper(data[key]) for key in data}
        else:
            return data
            
    return to_device_wrapper(data)


################# Sobel Edge ####################
class SobelEdgeDetector:
    def __init__(self, kernel_size=5, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

        # compute gaussian kernel
        size = kernel_size // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

        self.gaussian_kernel = torch.from_numpy(g)[None, None, :, :].float()
        self.Kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)[None, None, :, :]
        self.Ky = -torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float)[None, None, :, :]

    def detect(self, img, normalize=True):
        squeeze = False
        if len(img.shape) == 3:
            img = img[None, ...]
            squeeze = True

        img = pad_by_reflect(img, padding=self.kernel_size//2)
        img = F.conv2d(img, self.gaussian_kernel.repeat(1, img.size(1), 1, 1))

        img = pad_by_reflect(img, padding=1)
        Gx = F.conv2d(img, self.Kx)
        Gy = F.conv2d(img, self.Ky)

        G = (Gx.pow(2) + Gy.pow(2)).pow(0.5)
        if normalize:
            G = G / G.max()
        if squeeze:
            G = G[0]

        return G


def pad_by_reflect(x, padding=1):
    x = torch.cat((x[..., :padding], x, x[..., -padding:]), dim=-1)
    x = torch.cat((x[..., :padding, :], x, x[..., -padding:, :]), dim=-2)
    return x
    

################# FLOW COLORMAP ####################
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


################# Keypoint Related Utils ####################
class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma,):
        self.output_res = output_res
        self.num_joints = num_joints
        self.target_joint = None
        self.sigma = sigma

    def get_heat_val(self, x, y, x0, y0):
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
        return g

    def __call__(self, joints, bg_weight=0.0):
        assert self.num_joints == joints.shape[1], \
            'the number of joints should be %d' % self.num_joints

        heatmaps = torch.zeros(self.num_joints, self.output_res, self.output_res)
        ignored_heatmaps = 2*torch.ones(self.num_joints, self.output_res, self.output_res)

        ret = [heatmaps, ignored_heatmaps]
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.floor(x - 3 * self.sigma - 1)
                             ), int(np.floor(y - 3 * self.sigma - 1))
                    br = int(np.ceil(x + 3 * self.sigma + 2)
                             ), int(np.ceil(y + 3 * self.sigma + 2))

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)

                    joint_rg = torch.zeros((bb-aa, dd-cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy-aa, sx-cc] = self.get_heat_val(self.sigma, sx, sy, x, y)

                    ret[0][idx, aa:bb, cc:dd] = torch.maximum(ret[0][idx, aa:bb, cc:dd], joint_rg)
                    ret[1][idx, aa:bb, cc:dd] = 1.

        ret[1][ret[1] == 2] = bg_weight

        return ret


def preprocess_kpmap(arrs:torch.Tensor) -> torch.Tensor:
    # arr is B x 1 x H x W matrix or 1 x H x W matrix.
    # threshold, normalize to (0,1) and threshold again.
    pre_threshold = 0.1
    post_threshold = 0.4
    if (arrs > pre_threshold).any():
        arrs = arrs.clip(pre_threshold, 1.0)
        minimums = arrs.min(dim=-2, keepdim=True)[0].min(dim=-1, keepdim=True)[0]
        arrs -= minimums
        maximums = arrs.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0]
        arrs = arrs / maximums
        arrs[arrs < post_threshold] = 0
    else:
        # if all values are below threshold, return all zeros except the max point (to ensure at leas one mode)
        arrs = torch.where(arrs==arrs.max(), max(1e-4, arrs.max()), 0) # 1e-4 for numerical stability
    return arrs


def get_modes(arrs:torch.Tensor, return_scores=True, top_one=False):
    '''
    arrs: 1 x H x W matrix
    returns a list of modes, each mode is a (x, y) tuple.
    '''
    assert len(arrs.shape) == 3
    arrs = preprocess_kpmap(arrs).cpu()
    arrs = arrs.squeeze(0).numpy() # H x W
    # get local maximum points for each batch.
    labeled, n_feat = ndimage.label(arrs)
    coms_hw:List[Tuple[float, float]] = ndimage.center_of_mass(arrs, labeled, range(1, n_feat+1)) # in [[h, w]] format
    coms_hw = np.round(coms_hw).astype(int).tolist()
    coms_wh = [(c[1], c[0]) for c in coms_hw] # in [[w, h]] format
    coms_wh.sort()
    scores:List[float] = []
    for w, h in coms_wh:
        scores.append(arrs[h, w])
    if top_one:
        # get only one point with the largest score
        coms_wh = [coms_wh[np.argmax(scores)]]
        scores = [1.0]

    if return_scores:
        return coms_wh, scores
    return coms_wh


def modes_to_array(modes:List[List[Tuple]], scores:List[List[float]], max_detection=20):
    assert len(modes) == 17 # 17 keypoints per instances => 17, num_inst(varying), (x, y)
    assert len(scores) == 17 # 17 x num_modes(varying)
    # convert to array
    arr = np.zeros((17, max_detection, 3), dtype=int)
    pred_score = np.zeros(max_detection)
    for k in range(17):
        # Let's do just the simplest matching here.
        # Maybe should do some sort of matching later.
        current = modes[k]
        assert len(current) > 0, f'No mode for keypoint {k} in {modes}'
        for d in range(max_detection):
            arr[k,d,0] = round(current[d % len(current)][0])
            arr[k,d,1] = round(current[d % len(current)][1])
            arr[k,d,2] = 1
            pred_score[d] += scores[k][d % len(current)]

    return arr, pred_score


def ret_to_coco(all_ret, gt_coco, img_size=256):
    '''
    all_ret : dict {imgId: (17 x max_det x 3 ndarray, max_det score)}
    '''
    pred_dict = {}
    for imgId, (detection, scores) in all_ret.items():
        img = gt_coco.imgs[imgId]
        w = img['width']
        h = img['height']
        cat = gt_coco.anns[gt_coco.getAnnIds(imgIds=imgId)[0]]['category_id'] # assume one category for one img
        # convert size
        for d in range(detection.shape[1]):
            one_det = detection[:, d, :]  # 17 x 3
            # resize
            one_det[:, 0] = np.round(one_det[:, 0] * w / img_size)
            one_det[:, 1] = np.round(one_det[:, 1] * h / img_size)
            one_det = one_det.astype(int)
            res = {
                'image_id': imgId,
                'category_id': cat,
                'keypoints': one_det.reshape(-1).tolist(),
                'score': scores[d]
            }
            if imgId not in pred_dict:
                pred_dict[imgId] = []
            pred_dict[imgId].append(res)

    oks_thr = 0.9
    sigmas = np.array([
        0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
        0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089
    ])
    valid_kpts = []
    for image_id in pred_dict.keys():
        img_kpts = pred_dict[image_id]
        for n_p in img_kpts:
            box_score = n_p['score']
            n_p['keypoints'] = np.array(n_p['keypoints']).reshape(-1, 3)
            kpt_score = 0
            valid_num = 0
            x_min = np.min(n_p['keypoints'][:, 0])
            x_max = np.max(n_p['keypoints'][:, 0])
            y_min = np.min(n_p['keypoints'][:, 1])
            y_max = np.max(n_p['keypoints'][:, 1])
            area = (x_max - x_min) * (y_max - y_min)
            n_p['area'] = int(area)
            valid_num = 17 # assume all visible
            kpt_score = kpt_score / valid_num
            # rescoring
            n_p['score'] = float(kpt_score * box_score)
        keep = oks_nms(list(img_kpts), thr=oks_thr, sigmas=sigmas)
        valid_kpts.append([img_kpts[_keep] for _keep in keep])
    ret = []
    for each in valid_kpts:
        for det in each:
            det['keypoints'] = det['keypoints'].reshape(-1).astype(int).tolist()
            ret.append(det)

    return ret

def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.
    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
    Returns:
        list: The oks ious.
    """
    if sigmas is None:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious

def oks_nms(kpts_db, thr=0.9, sigmas=None, vis_thr=None, score_per_joint=False):
    """OKS NMS implementations.
    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
        score_per_joint: the input scores (in kpts_db) are per joint scores
    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep



################# Dense CRF ####################
MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3


def dense_crf(x, y):
    
    assert x.ndim == 3
    if y.ndim == 3:
        y = y.squeeze(0)
    
    img = (255*x).byte().permute(1, 2, 0).cpu().numpy()
    output_probs = torch.stack([1-y, y]).cpu().numpy()
    
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    # U = putils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    # d = dcrf.DenseCRF2D(w, h, c)
    d = None
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    Q = torch.from_numpy(Q)
    y_map = torch.argmax(Q, dim=0).float()
    
    return y_map


### Deraining PSNR / SSIM ###
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
  """
  2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
  Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
  """
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h


def calc_ssim(X, Y, sigma=1.5, K1=0.01, K2=0.03, R=255):
  '''
  X : y channel (i.e., luminance) of transformed YCbCr space of X
  Y : y channel (i.e., luminance) of transformed YCbCr space of Y
  Please follow the setting of psnr_ssim.m in EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
  Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
  The authors of EDSR use MATLAB's ssim as the evaluation tool, 
  thus this function is the same as ssim.m in MATLAB with C(3) == C(2)/2. 
  '''
  gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

  X = X.astype(np.float64)
  Y = Y.astype(np.float64)

  window = gaussian_filter

  ux = signal.convolve2d(X, window, mode='same', boundary='symm')
  uy = signal.convolve2d(Y, window, mode='same', boundary='symm')

  uxx = signal.convolve2d(X*X, window, mode='same', boundary='symm')
  uyy = signal.convolve2d(Y*Y, window, mode='same', boundary='symm')
  uxy = signal.convolve2d(X*Y, window, mode='same', boundary='symm')

  vx = uxx - ux * ux
  vy = uyy - uy * uy
  vxy = uxy - ux * uy

  C1 = (K1 * R) ** 2
  C2 = (K2 * R) ** 2

  A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
  D = B1 * B2
  S = (A1 * A2) / D
  mssim = S.mean()

  return mssim


def calc_psnr(sr, hr, scale, rgb_range, cal_type='y'):
    diff = (sr - hr) / rgb_range
    
    if cal_type=='y':
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)
    
    if scale == 1:
        valid = diff
    else:
        valid = diff[..., scale:-scale, scale:-scale]
    mse = np.power(valid, 2).mean()

    return -10 * math.log10(mse)


def get_y_channel(img, rgb_range):
    img = quantize(img, 1.0)
    gray_coeffs = [65.738, 129.057, 25.064]
    convert = img.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
    img = img.mul(convert).sum(dim=1).squeeze(0).numpy()
    return img


### Kitti Eigen Depth Estimation ###
def get_depth_metric(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]
