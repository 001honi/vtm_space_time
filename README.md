# Visual Token Matching with Time Attention

Follow this [link](https://github.com/GitGyun/visual_token_matching) for the base Visual Token Matching (VTM) published in ICLR 2023!
Follow this [link](https://drive.google.com/drive/folders/1-74V1qhEGX6uUnhH52nFgUjSmgUO6VWy?usp=sharing) for the presentation of VTM with temporal attention. 

This repository extends the base VTM by incorporating temporal attention.

![image-VTM](https://github.com/001honi/vtm_space_time/blob/base/asset/vtm-time.png)

**Abstract** 

VTM is a general-purpose few-shot learner for arbitrary dense prediction tasks. However, VTM lacks the ability to process temporal information, which limits its performance in video domains. 

We incorporate time attention into VTM to enhance its generalizability. Empirical results demonstrate that the proposed method outperforms the baseline VTM when there are substantial translational or scaling discrepancies between the video frames and the limited support set provided, such as in 1-shot or 2-shot scenarios. The method uses time attention to simulate a coarse motion memory that guides its predictions. It also benefits from using more frames for time attention, even with significant interpolation applied after meta-training. 

We evaluate the models on the DAVIS16 video segmentation dataset using mean-IoU scores. To ensure a fair comparison, we use the same parameters of VTM (ICLR’23) to initialize both models and train them on the MidAir video tasks dataset. However, VTM-time also computes time attention for 8 frames during the training, unlike VTM-base.

| Model    | Shot-1         | Shot-2         | Shot-4         |
|----------|----------------|----------------|----------------|
| VTM-time | 0.427 (+8.89%) | 0.540 (+4.37%) | 0.626 (+0.44%) |
| VTM-base | 0.392          | 0.517          | 0.623          |
