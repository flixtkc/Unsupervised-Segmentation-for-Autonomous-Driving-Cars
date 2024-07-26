import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from datasets.carla_dataset import CARLADataset

def main(config):
    # Dataset and DataLoader
    train_dataset = CARLADataset(root=config['data']['train_images'], split='train')
    val_dataset = CARLADataset(root=config['data']['val_images'], split='val')
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=config['train']['num_workers'])

    # Model
    model = models.segmentation.deeplabv3_resnet50(pretrained=config['model']['pretrained'])
    model.classifier[4] = nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1)) 

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(config['train']['epochs']):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation Step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}/{config["train"]["epochs"]}, Train Loss: {loss.item()}, Val Loss: {val_loss/len(val_loader)}')

    torch.save(model.state_dict(), 'carla_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on CARLA dataset')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    main(config)
