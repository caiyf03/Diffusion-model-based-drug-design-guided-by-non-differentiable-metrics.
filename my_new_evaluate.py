#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
这是最基础版的evaluate函数
传入参数: 目标sdf所在文件夹, 写入的result文件的文件名
输出: 目标文件夹下生成txt文档。
"""
import os
import argparse
import numpy as np
from rdkit import Chem
from analysis.metrics import MoleculeProperties

def check_sdf_file(file_path):
    # 读取 SDF 文件
    suppl = Chem.SDMolSupplier(file_path)
    molecules = [mol for mol in suppl if mol is not None]

    # 这里简单假设每个分子属于一个独立的口袋
    pocket_rdmols = [[mol] for mol in molecules]

    # 检查每个分子是否有效
    for pocket in pocket_rdmols:
        for mol in pocket:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                print(f"无效分子: {Chem.MolToSmiles(mol)}")
                return False

    print("文件中的分子均有效，符合基本要求。")
    return pocket_rdmols

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="评估指定文件夹下所有 SDF 文件中的分子属性")
    parser.add_argument("folder_path", type=str, help="包含 SDF 文件的文件夹路径")
    parser.add_argument("output_filename", type=str, help="写入的txt文件名字（例如: output.txt）")
    args = parser.parse_args()
    folder_path = args.folder_path
    output_filename = args.output_filename

    # 获取文件夹下的所有 SDF 文件
    folder_files = [os.path.join(folder_path, file) 
                    for file in os.listdir(folder_path) if file.endswith('.sdf')]

    output_file = os.path.join(folder_path, output_filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("QED: 计算给定 RDKit 分子的类药性指数(Quantitative Estimate of Drug-likeness,QED)值越接近 1 表示分子越具有类药性.\n")
        f.write("SA: 一个分子在合成方面的难易程度,分数越高, 合成越容易.\n")
        f.write("logp: 分子在辛醇(油脂)和水中的分配系数,通常LogP值应在-0.4和5.6之间的分子,可能成为良好的候选药物。其反映了分子在油水两相中的分配情况,LogP值越大,说明该物质越亲油脂;反之则越亲水,水溶性越好.\n")
        f.write("Lipinski: 是用于评估化合物是否具备良好口服生物利用度的经验法则，涵盖了分子量、氢键供体数量、氢键受体数量、脂水分配系数以及可旋转键数量这五个方面。越大越好\n")

        final_QED = []
        final_SA = []
        final_logp = []
        final_lipinski = []
        final_pocket_len = []

        for file_path in folder_files:
            result = check_sdf_file(file_path)
            if result:
                mol_metrics = MoleculeProperties()
                all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = mol_metrics.evaluate(result)
                file_name = os.path.basename(file_path)
                qed_flattened = [x for px in all_qed for x in px]
                sa_flattened = [x for px in all_sa for x in px]
                logp_flattened = [x for px in all_logp for x in px]
                lipinski_flattened = [x for px in all_lipinski for x in px]
                f.write(
                    f"file_name: {file_name} QED: {np.mean(qed_flattened):.3f} +/- {np.std(qed_flattened):.2f}, "
                    f"SA: {np.mean(sa_flattened):.3f} +/- {np.std(sa_flattened):.2f}, "
                    f"LogP: {np.mean(logp_flattened):.3f} +/- {np.std(logp_flattened):.2f}, "
                    f"Lipinski: {np.mean(lipinski_flattened):.3f} +/- {np.std(lipinski_flattened):.2f} "
                    f"pocket_len: {len(result)}\n"
                )
                print(f"对于 {file_path} 的评估完成！")
                final_QED.append(np.mean(qed_flattened))
                final_SA.append(np.mean(sa_flattened))
                final_logp.append(np.mean(logp_flattened))
                final_lipinski.append(np.mean(lipinski_flattened))
                final_pocket_len.append(len(result))

        f.write(
            f"final QED: {np.mean(final_QED):.3f} +/- {np.std(final_QED):.2f}, "
            f"SA: {np.mean(final_SA):.3f} +/- {np.std(final_SA):.2f}, "
            f"LogP: {np.mean(final_logp):.3f} +/- {np.std(final_logp):.2f}, "
            f"Lipinski: {np.mean(final_lipinski):.3f} +/- {np.std(final_lipinski):.2f} "
            f"pocket_len: {np.mean(final_pocket_len):.3f} +/- {np.std(final_pocket_len):.2f}, {(len(final_pocket_len)*20)},{np.sum(final_pocket_len)},{(100*np.sum(final_pocket_len)/(len(final_pocket_len)*20)):.3f}% \n"
        )
        print(f"对于 {folder_path} 的评估完成！")

if __name__ == "__main__":
    main()
