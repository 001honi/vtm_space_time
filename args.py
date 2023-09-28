import argparse
import yaml
from easydict import EasyDict


def str2bool(v):
    if v == 'True' or v == 'true':
        return True
    elif v == 'False' or v == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

# argument parser
def parse_args(shell_script=None):
    parser = argparse.ArgumentParser()

    # necessary arguments
    parser.add_argument('--debug_mode', '-debug', default=False, action='store_true')
    parser.add_argument('--continue_mode', '-cont', default=False, action='store_true')
    parser.add_argument('--skip_mode', '-skip', default=False, action='store_true')
    parser.add_argument('--no_train', '-nt', default=False, action='store_true')
    parser.add_argument('--no_eval', '-ne', default=False, action='store_true')
    parser.add_argument('--no_save', '-ns', default=False, action='store_true')
    parser.add_argument('--reset_mode', '-reset', default=False, action='store_true')
    parser.add_argument('--profile_mode', '-prof', default=False, action='store_true')
    parser.add_argument('--sanity_check', '-sc', default=False, action='store_true')
    parser.add_argument('--resolution_finetune_mode', '-resft', default=False, action='store_true')
    parser.add_argument('--quick_mode', '-quick', default=False, action='store_true')
    parser.add_argument('--check_mode', '-check', default=False, action='store_true')
    parser.add_argument('--large_mode', '-large', default=False, action='store_true')
    parser.add_argument('--temporary_checkpointing', '-tc', default=False, action='store_true')
    parser.add_argument('--development_mode', '-dev', default=False, action='store_true')
    parser.add_argument('--benchmark_mode', '-bm', default=False, action='store_true')
    parser.add_argument('--neurips2023_mode', '-nm', default=False, action='store_true')

    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--task', type=str, default='', choices=['', 'all', 'vos', 'mvos', 'ds', 'sod', 'semseg', 'animalkp', 'depth', 'flow'])
    parser.add_argument('--task_fold', '-fold', type=int, default=None, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--rgb_tasks', '-rgb', type=str2bool, default=False)
    parser.add_argument('--pose_tasks', '-pose', type=str2bool, default=False)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--exp_subname', type=str, default='')
    parser.add_argument('--exp_subname_aux', '-aname', type=str, default='', help='Experiment subname for loading domain adaptation')
    parser.add_argument('--name_postfix', '-ptf', type=str, default='')
    parser.add_argument('--save_postfix', '-sptf', type=str, default='')
    parser.add_argument('--result_postfix', '-rptf', type=str, default='')

    # optional arguments
    parser.add_argument('--model', type=str, default='VTM', choices=['VTM', 'DPT'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--strategy', '-str', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None, choices=['unified', 'taskonomy', 'davis2016', 'davis2017', 'isic2018', 'duts', 'loveda', 'ap10k', 'eigen', 'kittiflow'])
    parser.add_argument('--taskonomy', type=str2bool, default=None)
    parser.add_argument('--coco', type=str2bool, default=None)
    parser.add_argument('--midair', type=str2bool, default=None)
    parser.add_argument('--openimages', type=str2bool, default=None)
    parser.add_argument('--unlabeled', type=str2bool, default=None)
    parser.add_argument('--coco_real', type=str2bool, default=None)
    parser.add_argument('--midair_real', type=str2bool, default=None)
    # parser.add_argument('--unlabeled_domains', type=str, nargs='+', default=None) # 'all' or some of ['ph2', 'animals10', 'potsdam']
    parser.add_argument('--uniform_task_sampling', '-uts', type=str2bool, default=None)
    parser.add_argument('--task_sampling_weight', '-tsw', type=float, nargs='+', default=None)
    parser.add_argument('--uniform_dataset_sampling', '-uds', type=str2bool, default=None)
    parser.add_argument('--base_task', type=str2bool, default=None)
    parser.add_argument('--cont_task', type=str2bool, default=None)
    parser.add_argument('--cat_task', type=str2bool, default=None)

    parser.add_argument('--num_workers', '-nw', type=int, default=None)
    parser.add_argument('--global_batch_size', '-gbs', type=int, default=None)
    parser.add_argument('--eval_batch_size', '-ebs', type=int, default=None)
    parser.add_argument('--n_eval_batches', '-neb', type=int, default=None)
    parser.add_argument('--shot', type=int, default=None)
    parser.add_argument('--max_channels', '-mc', type=int, default=None)
    parser.add_argument('--support_idx', '-sid', type=int, default=None)
    parser.add_argument('--channel_idx', '-cid', type=int, default=None)
    parser.add_argument('--n_buildings', '-nb', type=int, default=None)
    parser.add_argument('--use_valid', '-uv', type=str2bool, default=None)
    parser.add_argument('--test_split', '-split', type=str, default=None)
    parser.add_argument('--class_name', '-class', type=str, default=None)
    parser.add_argument('--semseg_threshold', '-sth', type=float, default=None)
    parser.add_argument('--dense_crf', '-dcrf', type=str2bool, default=None)

    parser.add_argument('--image_augmentation', '-ia', type=str2bool, default=None)
    parser.add_argument('--unary_augmentation', '-ua', type=str2bool, default=None)
    parser.add_argument('--binary_augmentation', '-ba', type=str2bool, default=None)
    parser.add_argument('--mixed_augmentation', '-ma', type=str2bool, default=None)
    parser.add_argument('--trivial_augmentation', '-ta', type=str2bool, default=None)
    parser.add_argument('--order_mixup', '-om', type=str2bool, default=None)
    parser.add_argument('--image_encoder', '-ie', type=str, default='beit_base_patch16_224_in22k')
    parser.add_argument('--label_encoder', '-le', type=str, default='vit_base_patch16_224')
    parser.add_argument('--decoder_features', '-df', type=int, default=96)
    parser.add_argument('--deconv_head', '-dh', type=str2bool, default='False')
    parser.add_argument('--image_encoder_drop_path_rate', '-iedpr', type=float, default=None)
    parser.add_argument('--label_encoder_drop_path_rate', '-ledpr', type=float, default=None)
    parser.add_argument('--n_attn_heads', '-nah', type=int, default=12)
    parser.add_argument('--bitfit', '-bf', type=str2bool, default='True')
    parser.add_argument('--qkv_bitfit', '-qkvbf', type=str2bool, default=None)
    parser.add_argument('--n_channel_interaction_blocks', '-ncib', type=int, default=None)
    parser.add_argument('--channel_interaction_type', '-cit', type=str, default=None, choices=['global', 'axial', 'none'])
    parser.add_argument('--post_channel_interaction', '-pci', type=str2bool, default=None)
    parser.add_argument('--interaction_drop', '-id', type=int, default=None)
    parser.add_argument('--head_tuning', '-ht', type=str2bool, default=None)
    parser.add_argument('--knowledge_distill', '-kd', type=str2bool, default='False')
    parser.add_argument('--distill_type', '-dt', type=str, default=None, choices=['attention_map', 'feature_map'])
    parser.add_argument('--distill_weight', '-dw', type=float, default=None)
    parser.add_argument('--n_pseudo_channels', '-npc', type=int, default=None)
    parser.add_argument('--teacher_encoder', '-te', type=str, default=None)
    parser.add_argument('--dpt_seg_bce', '-dsbce', type=str2bool, default=None)
    parser.add_argument('--bitfit_init', '-bi', type=str, default=None, choices=['default', 'avg'])
    parser.add_argument('--multimodal_softmax_loss', '-msl', type=str2bool, default=None)
    parser.add_argument('--multimodal_mse_loss', '-mml', type=str2bool, default=None)
    parser.add_argument('--temp_msl', '-tmsl', type=float, default=None)
    parser.add_argument('--distill_start_block', '-dsb', type=int, default=None)
    parser.add_argument('--n_levels', '-nl', type=int, default=4)
    parser.add_argument('--dynamic_support', '-ds', type=str2bool, default=None)
    parser.add_argument('--dynamic_support_interval', '-dsi', type=int, default=None)
    parser.add_argument('--normalized_bce', '-nbce', type=str2bool, default=None, help='In finetune, normalize BCE loss by number of pixels')
    parser.add_argument('--crop_not_resize', '-cnr', type=str2bool, default=None, help='Use crop from full resolution(LoveDA)')
    parser.add_argument('--patches_per_img', type=int, default=None, help='Number of patches to choose from single image(LoveDA)')
    parser.add_argument('--spatial_softmax_loss', '-ssl', type=str2bool, default=None, help='Use spatial softmax loss in finetune')
    parser.add_argument('--mse_loss', '-mse', type=str2bool, default=None, help='Use mse loss in finetune')  
    parser.add_argument('--skip_crowd', type=str2bool, default=None, help='Skip images with multiple instances in ap10k finetune/test.')
    parser.add_argument('--top_one', type=str2bool, default=None, help='Get the top one detection for ap10k; NOTE:THIS SHOULD BE MERGED TO SKIP_CROWD')
    parser.add_argument('--multichannel', '-mch', type=str2bool, default=None, help='Use multichannel input for ap10k and loveda')
    parser.add_argument('--channel_ce', '-cce', type=str2bool, default=None, help='Use channel-wise CE loss for ap10k and loveda')
    parser.add_argument('--learning_to_bias', '-l2b', type=str2bool, default='False')
    parser.add_argument('--n_bias_sets', '-nbs', type=int, default=None)
    parser.add_argument('--average_real_bias_only', '-arbo', type=str2bool, default=None, help='Average biases from real tasks only during initialization of task-specific bias parameters.')
    parser.add_argument('--l2b_pre_projection', '-l2bpp', type=str2bool, default='False', help='Projection before averaging ema features in L2B.')
    parser.add_argument('--l2b_freeze_bias', '-l2bfb', type=str2bool, default=None, help='Freeze learned bias parameters and tune only the coefficients.')
    parser.add_argument('--permute_classes', '-pcls', type=str2bool, default=None, help='Permute class indices for multi-class binning.')
    parser.add_argument('--additional_bias', '-ab', type=str2bool, default=None, help='Additional Bias Term')

    parser.add_argument('--n_steps', '-nst', type=int, default=None)
    parser.add_argument('--n_schedule_steps', '-nscst', type=int, default=None)
    parser.add_argument('--optimizer', '-opt', type=str, default=None, choices=['sgd', 'adam', 'adamw', 'cpuadam'])
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr_pretrained', '-lrp', type=float, default=None)
    parser.add_argument('--lr_schedule', '-lrs', type=str, default=None, choices=['constant', 'sqroot', 'cos', 'poly'])
    parser.add_argument('--schedule_from', '-scf', type=int, default=None)
    parser.add_argument('--early_stopping_patience', '-esp', type=int, default=None)

    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--val_iter', '-viter', type=int, default=None)
    parser.add_argument('--save_iter', '-siter', type=int, default=None)
    parser.add_argument('--load_step', '-ls', type=int, default=None)

    parser.add_argument('--img_size', '-is', type=int, default=None, choices=[224, 384, 416, 512])

    if shell_script is not None:
        args = parser.parse_args(args=shell_script.split(' '))
    else:
        args = parser.parse_args()

    # load config file
    if args.stage == 0:
        if args.resolution_finetune_mode:
            config_path = 'configs/resolution_finetune_config.yaml'
        else:
            config_path = 'configs/train_config.yaml'
    elif args.stage == 1:
        config_path = 'configs/finetune_config.yaml'
    elif args.stage == 2:
        config_path = 'configs/test_config.yaml'
    elif args.stage == 3:
        config_path = 'configs/domain_adaptation_config.yaml'
        args.no_eval = True

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = EasyDict(config)

    # copy parsed arguments
    for key in args.__dir__():
        if key[:2] != '__' and getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))

    # retrieve data root
    with open('data_paths.yaml', 'r') as f:
        path_dict = yaml.safe_load(f)
        config.path_dict = path_dict

    ### benchmark mode
    if args.neurips2023_mode:
        config.large_mode = True
        config.knowledge_distill = True
        config.learning_to_bias = True
        config.l2b_pre_projection = True
        config.name_postfix = '_neurips2023' + config.name_postfix

    ### develpment mode
    if args.development_mode:
        config.debug_mode = True
        config.coco = False
        config.midair = False
        config.openimages = False
        config.global_batch_size = 2
        config.n_eval_batches = 1
        config.cont_task = False
        config.cat_task = False

    # quick mode
    if config.quick_mode:
        config.global_batch_size = 4
        config.n_steps = 100000
        config.name_postfix = f'_QUICK{config.name_postfix}'
    if config.benchmark_mode:
        config.no_eval = True
        config.n_steps = 100000
        config.name_postfix = f'_BENCHMARK{config.name_postfix}'
    if config.large_mode:
        config.image_encoder = 'beitv2_large_patch16_224'
        config.label_encoder = 'vit_large_patch16_224'
        config.n_attn_heads = 16
        config.decoder_features = 256
        config.name_postfix = f'_LARGE{config.name_postfix}'

    # for debugging
    if config.debug_mode:
        if config.profile_mode:
            config.n_steps = 500
            config.log_iter = 10
            config.val_iter = 250
            config.save_iter = 250
        elif config.stage == 1:
            config.n_steps = 10
            config.log_iter = 1
            config.val_iter = 5
            config.save_iter = 5
        else:
            config.n_steps = 10
            config.log_iter = 1
            config.val_iter = 5
            config.save_iter = 5

        if config.stage == 2:
            config.n_eval_batches = 2
        config.log_dir += '_debugging'
        if config.stage == 0 and not config.resolution_finetune_mode:
            config.load_dir += '_debugging'
        if config.stage != 2:
            config.save_dir += '_debugging'


    if config.exp_name == '':
        if config.dataset == 'unified' or config.stage != 0: # MINOR UPDATE for finetuning 
            config.exp_name = f'{config.model}_unified{config.name_postfix}'
        else:
            if config.task == '':
                config.exp_name = f'{config.model}_fold:{config.task_fold}{config.name_postfix}'
            else:
                config.exp_name = f'{config.model}_task:{config.task}{config.name_postfix}'

    if config.stage == 0:
        if config.n_schedule_steps < 0:
            config.n_schedule_steps = config.n_steps

    elif config.stage == 1:
        tag = 'mtest_valid' if config.use_valid else 'mtest_support'
        if config.dataset == 'taskonomy':
            if config.task == 'segment_semantic':
                config.monitor = f'{tag}/segment_semantic_{config.channel_idx}_pred'
            else:
                config.monitor = f'{tag}/{config.task}_pred'
        elif config.dataset in ['davis2016', 'davis2017', 'loveda', 'ap10k']:
            config.monitor = f'{tag}/{config.dataset}_{config.task}_{config.class_name}_pred'
        elif config.dataset in ['isic2018', 'duts', 'kittiflow', 'rain100l', 'rain100h', 'eigen']:
            config.monitor = f'{tag}/{config.dataset}_{config.task}_pred'

    elif config.stage == 3:
        config.monitor = ''

    # Time attention
    if config.time_attn < 2:
        config.time_attn = 0 

    return config
