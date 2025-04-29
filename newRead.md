1. 生成新配体分子
目标是为给定的蛋白质生成可能与其结合的新配体分子。该过程使用预训练的模型，并指定蛋白质的结合口袋。

命令：
bash
复制
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/3rfm.pdb --outfile example/3rfm_mol.sdf --ref_ligand A:330 --n_samples 20
解释：
generate_ligands.py：用于生成新配体分子的脚本。

checkpoints/crossdocked_fullatom_cond.ckpt：训练好的模型权重文件。

--pdbfile example/3rfm.pdb：蛋白质的PDB文件（这里使用的是PDB ID为3RFM的蛋白质）。

--outfile example/3rfm_mol.sdf：生成的配体分子将保存到该输出文件中（格式为SDF）。

--ref_ligand A:330：指定蛋白质中的参考配体。这里参考配体位于蛋白质的A链，残基编号为330。

--n_samples 20：生成20个新的配体样本。

替代方法：使用SDF文件指定参考配体
如果参考配体是以SDF文件形式提供的，可以使用以下命令：

bash
复制
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/3rfm.pdb --outfile example/3rfm_mol.sdf --ref_ligand example/3rfm_B_CFF.sdf --n_samples 20
--ref_ligand example/3rfm_B_CFF.sdf：参考配体通过SDF文件提供，而不是通过链和残基编号指定。

2. 子结构修复（Substructure Inpainting）
该技术用于围绕固定的子结构（如骨架扩展或片段连接）设计分子。使用inpaint.py脚本来实现这一功能。

命令：
bash
复制
python inpaint.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile example/5ndu.pdb --outfile example/5ndu_linked_mols.sdf --ref_ligand example/5ndu_C_8V2.sdf --fix_atoms example/fragments.sdf --center ligand --add_n_nodes 10
解释：
inpaint.py：用于子结构修复的脚本。

--pdbfile example/5ndu.pdb：蛋白质的PDB文件（这里使用的是PDB ID为5NDU的蛋白质）。

--outfile example/5ndu_linked_mols.sdf：生成的分子将保存到该输出文件中（格式为SDF）。

--ref_ligand example/5ndu_C_8V2.sdf：参考配体的SDF文件。

--fix_atoms example/fragments.sdf：指定固定的子结构（例如片段或骨架），这些子结构在生成新分子时保持不变。

--center ligand：新原子将围绕固定子结构的质心生成。

--add_n_nodes 10：指定生成的新分子中要添加的原子数量（这里为10个）。

其他选项：
--center pocket：新原子将围绕结合口袋的中心生成，而不是围绕固定子结构的质心。

--add_n_nodes：如果不指定该参数，脚本将随机选择添加的原子数量。

3. 注意事项
固定子结构的中心选择：--center ligand选项会将新原子围绕固定子结构的质心生成，但如果子结构大小差异较大（例如两个片段大小不同），可能会导致生成结果不理想。此时可以尝试--center pocket，或者根据具体问题调整生成策略。

原子数量：--add_n_nodes参数控制生成的新原子数量。如果不指定，脚本会随机选择数量。

总结
生成新配体：使用generate_ligands.py脚本，通过指定蛋白质和参考配体，生成可能与其结合的新分子。

子结构修复：使用inpaint.py脚本，围绕固定的子结构（如片段或骨架）设计新分子，适用于骨架扩展或片段连接等任务。

这些工具可以大大加速药物设计中的分子生成和优化过程。


口袋残基（Pocket Residues）
定义：

口袋残基是指蛋白质结合口袋（Binding Pocket）中与配体（Ligand）相互作用的氨基酸残基。

这些残基通常位于蛋白质的表面或内部空腔中，能够通过氢键、疏水作用、静电作用等方式与配体分子结合。

作用：

口袋残基定义了结合口袋的空间范围和化学环境，是配体生成任务的重要输入。

在配体生成过程中，模型会根据口袋残基的几何形状和化学特性，生成与之匹配的配体分子。

表示方式：

口袋残基通常通过**链标识（Chain ID）和残基编号（Residue Number）**来指定。例如：

A:123 表示蛋白质的 A 链中第 123 个残基。

B:45 表示蛋白质的 B 链中第 45 个残基。

在配体生成任务中的应用：

在生成配体时，可以通过指定口袋残基列表来定义结合口袋。例如：

bash
复制
python generate_ligands.py <checkpoint>.ckpt --pdbfile 1abc.pdb --outfile results/1abc_mols.sdf --resi_list A:123 A:124 A:125
这里 A:123 A:124 A:125 指定了结合口袋的残基。

2. 配体节点数量（Number of Ligand Nodes）
定义：

配体节点数量是指配体分子中的原子数量（即配体的大小）。

在分子生成任务中，节点通常对应于配体分子中的原子。

作用：

配体节点数量决定了生成配体的大小和复杂度。

通过控制节点数量，可以生成不同大小的配体分子，例如小分子片段或较大的药物分子。

在配体生成任务中的应用：

在生成配体时，可以通过参数控制配体的节点数量。例如：

--num_nodes_lig：直接指定生成配体的节点数量。

--fix_n_nodes：生成与参考配体相同节点数量的分子。

--n_nodes_bias：在随机采样节点数量时添加偏差。

--n_nodes_min：设置生成配体的最小节点数量。

示例：

bash
复制
python generate_ligands.py <checkpoint>.ckpt --pdbfile 1abc.pdb --outfile results/1abc_mols.sdf --resi_list A:123 A:124 A:125 --num_nodes_lig 20
这里 --num_nodes_lig 20 指定生成配体的节点数量为 20。

