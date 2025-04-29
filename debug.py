import torch
checkpoint_path = "DiffSBDD/RL_check_point/try2.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(key)

if "adjust_net_state_dict" in checkpoint:
    print("\nKeys in adjust_net_state_dict:")
    for key in checkpoint["adjust_net_state_dict"].keys():
        print(key)
else:
    print("Key 'adjust_net_state_dict' not found in checkpoint!")
