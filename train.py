import argparse
from argparse import Namespace
from pathlib import Path
import warnings

import torch
import pytorch_lightning as pl
import yaml
import numpy as np
import os
from lightning_modules import LigandPocketDDPM
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# ... existing code ...
print(torch.__version__)
print(torch.cuda.is_available())
# 打印 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")
# 打印 CUDA 版本，如果可用
if torch.version.cuda is not None:
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA is not available.")
# ... existing code ...

def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser() # 解析命令行参数
    p.add_argument('--config', type=str, required=True) # 指定配置文件的路径。
    p.add_argument('--resume', type=str, default=None) # 恢复训练时的检查点文件路径。
    args = p.parse_args() # 修改 configs\crossdock_fullatom_cond.yml文件

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume) # 恢复训练时的检查点文件路径
    if args.resume is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)
    # print(args)
    out_dir = Path(args.logdir, args.run_name) # 输出目录的路径


    # 确保日志目录存在且可写
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    
    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()
    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=args.virtual_nodes
    )
    '''
    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir, # 日志文件的保存目录
        project='ligand-pocket-ddpm', # 项目名称
        group=args.wandb_params.group,
        name=args.run_name,
        id=args.run_name,
        resume='must' if args.resume is not None else False, # 如果 args.resume 不为 None，恢复训练时的检查点文件路径
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )
    '''
    # 根据配置决定是否初始化 wandb
    if args.wandb_params.mode != 'disabled':
        logger = pl.loggers.WandbLogger(
            save_dir=args.logdir,
            project='ligand-pocket-ddpm',
            group=getattr(args.wandb_params, 'group', None),
            name=args.run_name,
            id=args.run_name,
            resume='must' if args.resume is not None else False,
            entity=args.wandb_params.entity,
            mode=args.wandb_params.mode,
        )
    else:
        logger = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint( #回调函数，用于在训练过程中保存模型的检查点。
        dirpath=Path(out_dir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val", # 指定要监控的指标，这里是验证集的损失 loss/val
        save_top_k=1, #指定保存性能最好的前 k 个模型，这里设置为 1。
        save_last=True,
        mode="min",
    )
    ############################管理模型的训练过程########################
    trainer = pl.Trainer(
        max_epochs=args.n_epochs, # 训练的最大轮数
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=args.enable_progress_bar, # 是否启用进度条
        num_sanity_val_steps=args.num_sanity_val_steps,
        accelerator='gpu', devices=args.gpus,
        strategy=('ddp' if args.gpus > 1 else None),
        limit_train_batches=1.0,  # 限制训练批次的比例
        limit_val_batches=1.0,  # 限制验证批次的比例
        limit_test_batches=1.0,  # 限制测试批次的比例
        #precision=16,  # 启用混合精度训练
        accumulate_grad_batches=4  # 梯度累积，减少内存使用
    )

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)
