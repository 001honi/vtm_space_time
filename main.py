import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pytorch_lightning as pl
import torch
import warnings

from args import parse_args
from train.train_utils import configure_experiment, load_model

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pl.utilities.warnings.PossibleUserWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    IS_RANK_ZERO = int(os.environ.get('LOCAL_RANK', 0)) == 0
    
    # parse args
    config = parse_args()

    if not config.check_mode:
        # load model
        # reduced = False #(config.stage > 0) if reduced is True load from .pth else .ckpt
        model, ckpt_path = load_model(config, verbose=IS_RANK_ZERO, reduced=(config.stage > 0))

        # environmental settings
        logger, log_dir, save_dir, callbacks, profiler, precision, strategy, plugins = configure_experiment(config, model)
        if config.stage == 1:
            model.config.ckpt_dir = save_dir
        if config.stage == 2:
            model.config.result_dir = log_dir

    if IS_RANK_ZERO:
        print(f'''\
        Running Stage {config.stage} with {config.strategy} Strategy:
        > Exp Name: {config.exp_name}
        > Model: {config.model}
            - Image Encoder: {config.image_encoder}
            - Label Encoder: {config.label_encoder}
            - Decoder Features: {config.decoder_features}
            - Deconv Head: {config.deconv_head}
            - Num Attention Heads: {config.n_attn_heads}
            - Num Levels: {config.n_levels}
            - Hyper Matching: {config.learning_to_bias}
            - Joint Embedding w/ Post Projection: {config.l2b_pre_projection}
        > Image Size: {config.img_size}
        > Global Batch Size: {config.global_batch_size}
        > Eval Batch Size: {config.eval_batch_size}
        > Max Channels: {config.max_channels}
        > Num Workers: {config.num_workers}
        > Num Steps: {config.n_steps}
        > Learning Rate: {config.lr}
        > Val Iters: {config.val_iter}
        ''') 

        # create pytorch lightning trainer.
        trainer = pl.Trainer(
            logger=logger,
            default_root_dir=save_dir,
            accelerator='gpu',
            max_epochs=((config.n_steps // config.val_iter) if (not config.no_eval) and config.stage <= 1 else 1),
            log_every_n_steps=-1,
            num_sanity_val_steps=(2 if config.sanity_check else 0),
            callbacks=callbacks,
            benchmark=True,
            devices=-1,
            strategy=strategy,
            precision=precision,
            profiler=profiler,
            plugins=plugins
        )

        # validation at start
        if config.stage == 1:
            trainer.validate(model, verbose=False)
            
        # start training or fine-tuning or domain adaptation
        if config.stage != 2:
            trainer.fit(model, ckpt_path=ckpt_path)
        # start evaluation
        else:
            trainer.test(model)
