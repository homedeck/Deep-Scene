# 
#   Deep Scene
#   Copyright (c) 2020 Homedeck, LLC.
#

from argparse import ArgumentParser
from colorama import Fore, Style
from suya import set_suya_access_key
from suya.torch import LabeledDataset
from torch import device as get_device
from torch.cuda import is_available as cuda_available
from torch.jit import save, script
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import CenterCrop, ColorJitter, Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor
from torchsummary import summary
import tableprint

from model import DeepScene

# Parse arguments
parser = ArgumentParser(description="Deep Scene: Train")
parser.add_argument("--suya-key", type=str, required=False, default=None, help="Suya access key")
parser.add_argument("--tags", type=str, nargs="+", help="Image tags")
parser.add_argument("--learning-rate", type=float, default=2e-5, help="Nominal learning rate")
parser.add_argument("--epochs", type=int, default=40, help="Epochs")
parser.add_argument("--batch-size", type=int, default=16, help="Minibatch size")
args = parser.parse_args()

# Create dataset
set_suya_access_key(args.suya_key)
transform = Compose([
    Resize(512),
    CenterCrop(512),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = LabeledDataset(*args.tags, size=2000, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)

# Create model
device = get_device("cuda:0") if cuda_available() else get_device("cpu")
model = DeepScene(classes=len(args.tags))
model = model.to(device)

# Create optimizer and loss
cross_entropy_loss = CrossEntropyLoss().to(device)
optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

# Print
print("Preparing for training:")
summary(model, (3, 512, 512), batch_size=args.batch_size)

# Create summary writer
with SummaryWriter() as summary_writer:

    # Print table and graph
    HEADERS = ["Iteration", "Epoch", "Content"]
    print(tableprint.header(HEADERS))

    # Setup for training
    model.train(mode=True)
    iteration_index = 0
    last_loss = 1e+10

    # Iterate over epochs
    for epoch in range(args.epochs):

        # Iterate over all minibatches
        for image, label in dataloader:
            
            # Run forward pass
            image, label = image.to(device), label.to(device)
            predicted_label = model(image)

            # Backpropagate
            optimizer.zero_grad()
            loss_total = cross_entropy_loss(predicted_label, label.squeeze())
            loss_total.backward()
            optimizer.step()

            # Log
            summary_writer.add_scalar("CE Loss", loss_total, iteration_index)
            LOG_DATA = [
                f"{iteration_index}",
                f"{epoch}",
                f"{Style.BRIGHT}{Fore.GREEN if loss_total < last_loss else Fore.RED}{loss_total:.4f}{Style.RESET_ALL}"
            ]
            print(tableprint.row(LOG_DATA))
            last_loss = loss_total
            iteration_index += 1

        # Save model
        model = model.cpu()
        scripted_model = script(model)
        save(scripted_model, "deep_scene.pt")
        model = model.to(device)
    
    # Print
    print(tableprint.bottom(len(HEADERS)))