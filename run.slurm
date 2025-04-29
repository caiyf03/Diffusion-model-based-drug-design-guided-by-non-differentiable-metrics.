#!/bin/bash
#SBATCH -J test_job              # 作业名称
#SBATCH -p normal                # 提交到normal队列
#SBATCH -N 1                     # 请求1个节点
#SBATCH --cpus-per-task=4        # 每个任务请求4个CPU核心
#SBATCH --gres=gpu:1             # 请求1个GPU
#SBATCH --time=00:30:00          # 作业最大运行时间为30分钟
#SBATCH --output=%j.out          # 标准输出文件
#SBATCH --error=%j.err           # 标准错误文件
#SBATCH --mail-type=all          # 发送所有类型的邮件通知
#SBATCH --mail-user=YOU@MAIL.COM # 接收通知的邮箱

# 加载必要的模块（如果需要）
module load cuda/10.2
conda activate sbdd-env8
python -u DiffSBDD/train.py --config DiffSBDD/configs/crossdock_fullatom_cond.yml --resume logdir/SE3-cond-full/checkpoints/last.ckpt
