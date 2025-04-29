import argparse
from pathlib import Path

import torch
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages
import os
import utils
from lightning_modules import LigandPocketDDPM
import torch.optim as optim

def load_adjust_checkpoint(model, optimizer, filename="adjust_checkpoint.pth"):
    try:
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))
        model.adjust_net.load_state_dict(checkpoint['adjust_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"AdjustNet checkpoint loaded from {filename}")
    except FileNotFoundError:
        print("No adjust checkpoint found. Starting from scratch.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path) # 预训练模型的检查点文件
    parser.add_argument('--pdbfile', type=str) # 蛋白质的PDB文件路径
    parser.add_argument('--resi_list', type=str, nargs='+', default=None) #：默认为 None，用于指定氨基酸残基列表。
    parser.add_argument('--ref_ligand', type=str, default=None) #指定参考配体的信息，可能用于指导配体的生成过程。
    parser.add_argument('--outfile', type=Path)
    parser.add_argument('--n_samples', type=int, default=20) #指定要生成的配体样本数量，默认为20。
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_nodes_lig', type=int, default=None) #默认为 None，指定配体的节点数量。
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    #####################################################################################
    parser.add_argument('--optimize', type=int, default=0) # 是否开启噪声优化
    parser.add_argument('--path', type=str, default=None) # 噪声优化模型的路径
    parser.add_argument('--path_save', type=str, default=None) # 新的参数保存路径

    parser.add_argument('--SVDD', type=int, default=0) # 是否启用SVDD

    parser.add_argument('--SPSA', type=int, default=0) # 是否启用SPSA
    #########################################################################################
    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem # 例如 3rfm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device, strict=False)
    model = model.to(device)
    print("we are in generate_ligands.py")
    ##############################################################################################
    if args.optimize == 1:
        # 查看加载完毕后 adjust_net 的部分参数
        print("=== [DEBUG] Before loading adjust checkpoint ===")
        if hasattr(model.ddpm, "adjust_net"):
            fc1_weights_before = model.ddpm.adjust_net.fc1.weight.data.clone()
            print("fc1.weight mean before =", fc1_weights_before.mean().item())
        else:
            print("No adjust_net found in model.ddpm")
        # 加载 adjust_net 的 checkpoint
        adjust_ckpt_path = args.path
        if os.path.exists(adjust_ckpt_path):
            print(f"loading from {adjust_ckpt_path}")
            # 注意：这里我们加载到 model.ddpm.adjust_net，因为 model.ddpm 是 ConditionalDDPM 的实例
            model.ddpm.load_checkpoint(model.ddpm.adjust_optimizer, adjust_ckpt_path)
            # print(f"=== [DEBUG] Loading adjust_net from {adjust_ckpt_path} ===")
            # 4. 再次打印对比
            fc1_weights_after = model.ddpm.adjust_net.fc1.weight.data
            print("=== [DEBUG] After loading adjust checkpoint ===")
            # print("fc1.weight[:5] after  =", fc1_weights_after.view(-1)[:5])
            print("fc1.weight mean after =", fc1_weights_after.mean().item())

            # 你也可以比对一下和之前的差异
            diff = (fc1_weights_after - fc1_weights_before).abs().sum().item()
        else:
            print(f"No adjust checkpoint found at {adjust_ckpt_path}. Starting from scratch.")
    else:
        print("No noise optimization.")
    #################################################################################################
    if args.num_nodes_lig is not None:
        num_nodes_lig = torch.ones(args.n_samples, dtype=int) * \
                        args.num_nodes_lig
    else:
        num_nodes_lig = None

    molecules = []
    #print(f"timestep is {args.timesteps}")
    for i in range(args.n_samples // args.batch_size):
        molecules_batch = model.generate_ligands(
            args.pdbfile, args.batch_size, args.resi_list, args.ref_ligand,
            num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
            relax_iter=(200 if args.relax else 0),
            resamplings=args.resamplings, jump_length=args.jump_length,
            timesteps=args.timesteps, pdb_id=pdb_id,optimize=args.optimize,path=args.path,path_save=args.path_save,svdd = args.SVDD, spsa = args.SPSA) # 额外传入pdb_id参数
        molecules.extend(molecules_batch)

    # Make SDF files
    utils.write_sdf_file(args.outfile, molecules)
