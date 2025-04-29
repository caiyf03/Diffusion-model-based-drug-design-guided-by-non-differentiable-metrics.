#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
该脚本用于批量调用 inpaint.py，对输入文件夹中的 PDB 文件进行配体 inpainting。
示例：
    python run_inpaint.py \
        --input_folder DiffSBDD/example \
        --output_folder DiffSBDD/my_example_inpaint \
        --checkpoint DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt \
        --ref_suffix _C_8V2.sdf \
        --fix_atoms DiffSBDD/example/fragments.sdf \
        --center ligand \
        --add_n_nodes 10 \
        --svdd 1
"""
import os
import argparse
import gc
import torch

def main():
    parser = argparse.ArgumentParser(
        description="批量运行 inpaint.py，进行配体 inpainting")
    parser.add_argument("--input_folder", type=str, default="DiffSBDD/my_new_data/processed_crossdock_noH_full_temp/test",
                        help="输入文件夹路径，包含pdb、sdf、txt文件")
    parser.add_argument("--output_folder", type=str, required=True,
        help="输出文件夹，inpaint 生成的 .sdf 会写到这里")
    parser.add_argument(
        "--fix_atoms", type=str, default="DiffSBDD/fix_ligand",
        help="fragment 文件 (.sdf)，inpaint 时固定的原子子结构")
    parser.add_argument(
        "--center", type=str, default="ligand", choices=["ligand", "pocket"],
        help="inpaint 时的 center 参数")
    parser.add_argument(
        "--add_n_nodes", type=int, default=10,
        help="每个样本额外添加的节点数")
    parser.add_argument(
        "--svdd", type=int, default=0, choices=[0,1],
        help="是否启用 SVDD (0/1)")
    parser.add_argument(
        "--timesteps", type=int, default=60,
        help="denoising 步数，不传则使用训练时的 diffusion_steps")
    parser.add_argument(
        "--resamplings", type=int, default=10,
        help="denoising 步数，不传则使用训练时的 diffusion_steps")
    parser.add_argument(
        "--n_samples", type=int, default=20,
        help="每次调用 inpaint.py 时的样本数量")
    args = parser.parse_args()

    # 定义基础命令
    base_command = "python DiffSBDD/inpaint.py DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
    # 创建输出文件夹
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # 获取输入文件夹中的所有文件
    all_files = os.listdir(args.input_folder)
    the_fixed = os.listdir(args.fix_atoms)
    # 分别筛选出 pdb, sdf 和 txt 文件
    pdb_files = sorted([f for f in all_files if f.endswith('.pdb')])
    sdf_files = sorted([f for f in all_files if f.endswith('.sdf')])
    txt_files = sorted([f for f in all_files if f.endswith('.txt')])
    fixed_files = sorted([f for f in the_fixed if f.endswith('.sdf')])
    # 选择前100个pdb文件
    selected_pdb_files = pdb_files[0:50]
    selected_sdf_files = sdf_files[0:50]
    selected_txt_files = txt_files[0:50]
    selected_fixed_files = fixed_files[0:50]

    for i, pdb_name in enumerate(selected_pdb_files):
        print(f"[INFO] my_test: handling {pdb_name}, sdf file {selected_fixed_files[i]} ...")
        print(f"[INFO] my_test: is the {i+1}th file")
        pdb_path = os.path.join(args.input_folder, pdb_name)

        # 构建输出文件路径（将sdf文件名的扩展名替换为 .sdf）
        sdf_file = selected_sdf_files[i]
        outfile = os.path.join(args.output_folder, os.path.splitext(sdf_file)[0] + ".sdf")
        # 获取对应的txt文件路径
        sdf_file = os.path.join(args.input_folder, selected_sdf_files[i])
        # fixed
        fixed_file = os.path.join(args.fix_atoms, selected_fixed_files[i])

        # 构建完整命令
        cmd = (
            f"{base_command} "
            f"--pdbfile {pdb_path} "
            f"--outfile {outfile} "
            f"--ref_ligand {sdf_file} "
            f"--fix_atoms {fixed_file} "
            f"--center {args.center} "
            f"--add_n_nodes {args.add_n_nodes} "
            f"--timesteps {args.timesteps} "
            f"--resamplings {args.resamplings} "
            f"--svdd {args.svdd} "
            f"--n_samples {args.n_samples} "
        )
        os.system(cmd)

        # 清理 GPU 缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("所有 inpainting 任务已完成。")

if __name__ == "__main__":
    main()
