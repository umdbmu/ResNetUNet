import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time, copy

from resnetunet import ResNetUNet
from trainer import train_model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_class = 6
    model = ResNetUNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    model = train_model(model, optimizer_ft, exp_lr_scheduler, device=device, num_epochs=60)