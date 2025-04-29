from    import Chem
from rdkit.Chem import Draw

# 加载 SDF 文件
supplier = Chem.SDMolSupplier(r"D:\school\bishe\DiffSBDD\example\3rfm_mol.sdf", sanitize=False)

# 显示分子
for mol in supplier:
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)  # 手动校验分子结构
            img = Draw.MolToImage(mol)
            img.show()
        except Exception as e:
            print(f"分子校验失败: {e}")
