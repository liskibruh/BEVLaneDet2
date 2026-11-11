import torch

checkpoint_path = r"C:\Users\oo_wa\Downloads\python_scripts\BEVLaneDet2\checkpoints\bevlanedet_r34_vox_exp6.pth"

# Setting weights_only to False lets PyTorch unpickle custom MMEngine metadata structures seamlessly
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

for key, tensor in state_dict.items():
    if hasattr(tensor, "shape"):
        print(f"{key}: {tensor.shape}")
