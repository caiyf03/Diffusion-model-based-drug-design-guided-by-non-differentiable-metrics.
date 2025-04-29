#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python extract_fixed.py /path/to/input_folder /path/to/output_folder
import os
import argparse
from rdkit import Chem
from rdkit.Chem import SanitizeFlags

def extract_fixed_substructures(input_dir, output_dir):
    """
    从 input_dir 中提取所有 .sdf 文件，
    对每个 SDF 文件，读取第一个分子，取前 1/4 原子构成子分子，
    并将结果写入 output_dir，文件名与原文件相同。
    写出时禁用 Kekulize，并移除所有立体化学标记。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.sdf'):
            continue
        input_path = os.path.join(input_dir, fname)

        # 1. 读取原始 SDF，只取第一个分子
        suppl = Chem.SDMolSupplier(input_path, removeHs=False)
        mol = None
        for m in suppl:
            if m is not None:
                mol = m
                break
        if mol is None:
            print(f"警告：无法从 {fname} 中读取到分子，已跳过")
            continue

        # 2. 计算要固定的原子索引（前 1/4）
        num_atoms = mol.GetNumAtoms()
        n_fix = num_atoms // 4
        if n_fix < 1:
            print(f"{fname}: 原子数太少（{num_atoms}），无法提取子结构，已跳过")
            continue
        fix_idxs = list(range(n_fix))

        # 3. 提取子分子（保留选中原子及它们之间的键）
        submol = Chem.PathToSubmol(mol, fix_idxs)

        # 4. Sanitize 但跳过 Kekulize
        try:
            Chem.SanitizeMol(
                submol,
                sanitizeOps=SanitizeFlags.SANITIZE_ALL
                          ^ SanitizeFlags.SANITIZE_KEKULIZE
            )
        except Exception as e:
            print(f"警告：{fname} 在 Sanitize 跳过 Kekulize 时出错：{e}")

        # 5. 移除所有立体化学标记，避免“no eligible neighbors for chiral center”
        Chem.RemoveStereochemistry(submol)

        # 6. 写出新的 SDF，禁止 Kekulize
        output_path = os.path.join(output_dir, fname)
        mol_block = Chem.MolToMolBlock(submol, kekulize=False)
        with open(output_path, 'w') as f:
            f.write(mol_block)
            f.write('$$$$\n')

        print(f"{fname}: 原子总数={num_atoms}, 已提取前 {n_fix} 个原子到 {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="提取 SDF 中前 1/4 原子构成的子结构（子分子），保存到新的文件夹，写出时禁用 Kekulize 并移除立体化学"
    )
    parser.add_argument(
        "input_dir",
        help="包含原始 .sdf 文件的输入文件夹路径"
    )
    parser.add_argument(
        "output_dir",
        help="提取后的子分子 SDF 文件保存到此文件夹"
    )
    args = parser.parse_args()

    extract_fixed_substructures(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
