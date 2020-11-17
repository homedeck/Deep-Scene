# 
#   Deep Scene
#   Copyright (c) 2020 Homedeck, LLC.
#

from argparse import ArgumentParser
from PIL import Image
from torch import cat, device as get_device, set_grad_enabled
from torch.cuda import is_available as cuda_available
from torch.jit import load
from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor

# Parse arguments
parser = ArgumentParser(description="Deep Fusion: Test")
parser.add_argument("--model", type=str, default="deep_scene.pt", help="Path to trained model")
parser.add_argument("--input", type=str, nargs="+", help="Path to exposures")
args = parser.parse_args()

# Load model
device = get_device("cuda:0") if cuda_available() else get_device("cpu")
model = load(args.model, map_location=device).to(device)
set_grad_enabled(False)

# Load exposures
to_tensor = Compose([
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
input = Image.open(args.input)
input = to_tensor(input).unsqueeze(dim=0).to(device)

# Run forward
logits = model(input)
index = logits.argmax().item()
print(f"Image class is index {index}")