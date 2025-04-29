from analysis.metrics import MoleculeProperties
from rdkit import Chem
import os 
import numpy as np
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

# 替换为实际的文件路径
file_path1 = 'DiffSBDD/example/3rfm_mol.sdf'

# 自定义排序函数，提取文件名中 _ 与 . 之间的数字
def get_number_from_filename(filename):
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    if len(parts) > 1:
        number_part = parts[-1].split('.')[0]
        try:
            return int(number_part)
        except ValueError:
            return 0
    return 0

# 打开一个txt文件用于写入数据
with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
    # 写入表头
    f.write("QED: 计算给定 RDKit 分子的类药性指数(Quantitative Estimate of Drug-likeness,QED)值越接近 1 表示分子越具有类药性.\n")
    f.write("SA: 一个分子在合成方面的难易程度,分数越高, 合成越容易.\n")
    f.write("logp: 分子在辛醇(油脂)和水中的分配系数,通常LogP值应在-0.4和5.6之间的分子,可能成为良好的候选药物。其反应了分子在油水两相中的分配情况,LogP值越大,说明该物质越亲油脂;反之则越亲水,水溶性越好.\n")
    f.write("Lipinski: 是用于评估化合物是否具备良好口服生物利用度的经验法则，涵盖了分子量、氢键供体数量、氢键受体数量、脂水分配系数以及可旋转键数量这五个方面。越大越好\n")
    # 可以在这里调用 evaluate 方法进行评估
    from analysis.metrics import MoleculeProperties
    mol_metrics = MoleculeProperties()
    result = check_sdf_file(file_path1)
    all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = mol_metrics.evaluate(result)
        # 提取文件名中的编号
    file_name = os.path.basename(file_path1)
    qed_flattened = [x for px in all_qed for x in px]
    sa_flattened = [x for px in all_sa for x in px]
    logp_flattened = [x for px in all_logp for x in px]
    lipinski_flattened = [x for px in all_lipinski for x in px]
    # 将结果写入文件
    f.write(f"name {file_name}: QED: {np.mean(qed_flattened):.3f} +/- {np.std(qed_flattened):.2f}, SA: {np.mean(sa_flattened):.3f} +/- {np.std(sa_flattened):.2f},LogP: {np.mean(logp_flattened):.3f} +/- {np.std(logp_flattened):.2f},\
Lipinski: {np.mean(lipinski_flattened):.3f} +/- {np.std(lipinski_flattened):.2f} pocket_len: {len(result)}\n")
    print(f"对于{file_path1}的评估完成！")
    

