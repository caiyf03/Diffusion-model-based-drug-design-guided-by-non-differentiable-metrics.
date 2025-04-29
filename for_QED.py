import subprocess

# 定义要运行的命令
command = [
    "python", "DiffSBDD/generate_ligands.py",
    "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt",
    "--pdbfile", "DiffSBDD/example/3rfm.pdb",
    "--outfile", "DiffSBDD/my_example_for_origion/3rfm_mol.pdb",
    "--ref_ligand", "A:330",
    "--n_samples", "20",
    "--optimize", "1",
    "--path", "DiffSBDD/RL_check_point/try77.pth",
    "--path_save", "try77.pth"
]

# 循环30次
for i in range(1, 31):
    print(f"Running iteration {i}")
    subprocess.run(command)
    print(f"Iteration {i} completed")

print("All 30 iterations completed")