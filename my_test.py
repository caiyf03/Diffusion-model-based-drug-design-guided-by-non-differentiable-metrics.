#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
该脚本用于调用 generate_ligands.py，并对输入文件夹中的文件进行处理。
可以通过命令行传入输出文件夹、ATP和SPSA参数。
示例：
    python run_generate.py --output_folder DiffSBDD/my_example_SVDD_0/test_4 --ATP 1 --SPSA 0
"""

import os
import argparse
import gc
import torch

def main():
    parser = argparse.ArgumentParser(description="Run ligand generation with custom parameters")
    parser.add_argument("--operate_file", type=str, default="DiffSBDD/generate_ligands.py",
                        help="checkpoint")
    parser.add_argument("--checkpoint", type=str, default="DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt",
                        help="checkpoint")
    parser.add_argument("--input_folder", type=str, default="DiffSBDD/my_new_data/processed_crossdock_noH_full_temp/test",
                        help="输入文件夹路径，包含pdb、sdf、txt文件")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="输出文件夹路径，将生成的 sdf 文件写入此文件夹")
    parser.add_argument("--ATP", type=int, default=1, help="SVDD参数")
    parser.add_argument("--SPSA", type=int, default=0, help="SPSA参数")
    parser.add_argument("--optimize", type=int, default=0, help="optimize参数")
    parser.add_argument("--path", type=str, default="DiffSBDD/RL_check_point/calculate1.pth", help="path参数")
    parser.add_argument("--path_save", type=str, default="DiffSBDD/RL_check_point/final.pth", help="path_save参数")
    args = parser.parse_args()

    # 定义基础命令
    base_command = "python "
    #base_command = "python DiffSBDD/generate_ligands.py DiffSBDD/my_logdir/SE3-cond-full/checkpoints/best-model-epoch=epoch=17.ckpt"

    input_folder = args.input_folder
    output_folder = args.output_folder
    optimize = args.optimize
    path = args.path
    path_save = args.path_save
    svdd = args.ATP
    SPSA = args.SPSA

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有文件
    all_files = os.listdir(input_folder)

    # 分别筛选出 pdb, sdf 和 txt 文件
    pdb_files = sorted([f for f in all_files if f.endswith('.pdb')])
    sdf_files = sorted([f for f in all_files if f.endswith('.sdf')])
    txt_files = sorted([f for f in all_files if f.endswith('.txt')])

    # 选择前100个pdb文件
    selected_pdb_files = pdb_files[0:100]
    # 检查是否所有文件都是pdb格式
    for file in selected_pdb_files:
        if not file.endswith('.pdb'):
            raise ValueError(f"文件 {file} 不是PDB格式, 请检查文件夹内容。")

    # 选择前100个sdf和txt文件
    selected_sdf_files = sdf_files[0:100]
    selected_txt_files = txt_files[0:100]

    # 循环运行命令
    for i, pdb_file in enumerate(selected_pdb_files):
        print(f"[INFO] my_test: handling {pdb_file}, txt file {selected_txt_files[i]}, sdf file {selected_sdf_files[i]} ...")
        print(f"[INFO] my_test: is the {i+1}th file")
        # 构建输入文件路径
        pdbfile = os.path.join(input_folder, pdb_file)

        # 构建输出文件路径（将sdf文件名的扩展名替换为 .sdf）
        sdf_file = selected_sdf_files[i]
        outfile = os.path.join(output_folder, os.path.splitext(sdf_file)[0] + ".sdf")

        # 获取对应的txt文件路径
        sdf_file = os.path.join(input_folder, selected_sdf_files[i])

        # 构建完整命令
        full_command = (
            f"{base_command} "
            f"{args.operate_file} {args.checkpoint} "
            f"--pdbfile {pdbfile} --outfile {outfile} "
            f"--ref_ligand {sdf_file} --n_samples 20 --optimize {optimize} "
            f"--path {path} --SVDD {svdd} --SPSA {SPSA} --timesteps 600"
        )
        # 运行命令
        os.system(full_command)
        # 手动清除当前进程中可能残留的GPU缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
