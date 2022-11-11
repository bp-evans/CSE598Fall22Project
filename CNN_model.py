import glob
import os
# Getting this conflicting error. Most probably b/w sklearn and pytorch. Couldn't resolve it without this hack.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = torch.cuda.is_available()
mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
if cuda:
    device = torch.device('gpu')
elif mps:
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"PyTorch using device {device}")


class ImageDataset(Dataset):
    def __init__(self, transforms_=None, image_folder=False, dataset=None):
        self.transform = transforms.Compose(transforms_)
        self.folder = image_folder
        self.image_files = []
        self.image_labels = []
        self.populate_data(dataset)

    def populate_data(self, dataset):
        for _, row in dataset.iterrows():
            image_ = os.path.join(".", self.folder, row[3])
            if os.path.exists(image_):
                self.image_files.append(image_)
                self.image_labels.append(int(row[4]))

    def to_rgb(self, image):
        rgb_image = Image.new("RGB", image.size)
        rgb_image.paste(image)
        return rgb_image

    def __getitem__(self, index):
        ret_image = Image.open(self.image_files[index % len(self.image_files)])

        # Convert grayscale/rbg images to rgb
        if ret_image.mode != "RGB":
            ret_image = self.to_rgb(ret_image)

        ret_image = self.transform(ret_image)
        return {"x": ret_image, "y": self.image_labels[index % len(self.image_files)]}

    def __len__(self):
        return len(self.image_files)


class CNNModel(nn.Module):
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()
        channels, height, width = input_shape

        def cnn_block(in_filters, out_filters, normalize=True):
            layers = list()
            layers.append(nn.Conv2d(in_filters, out_filters, 5, stride=2, padding=1))
            layers.append(nn.MaxPool2d(2, 2))
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *cnn_block(channels, 64, normalize=False),
            *cnn_block(64, 128),
            nn.Flatten(),
            nn.Linear(128*15*15, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5),
            nn.ReLU(inplace=True),
        )

    def forward(self, img):
        x = self.model(img)
        return x


def train_test(csv_path='imageLabels.csv', batch_size=64, lr=0.0001,
               epochs=200, save_path='cnn_model.pk', image_folder="image_demos"):
    save_dict = "trained_models"
    dataset = pd.read_csv(csv_path, skiprows=1, header=None)
    train_data, test_data = train_test_split(dataset, train_size=0.8)
    transforms_ = [
        transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    train_dataloader = DataLoader(
        ImageDataset(transforms_=transforms_,
                     dataset=train_data,
                     image_folder=image_folder),
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        ImageDataset(transforms_=transforms_,
                     dataset=test_data,
                     image_folder=image_folder),
        batch_size=batch_size,
        shuffle=True,
    )

    input_size = (3, 256, 256)  # channel, width and height
    model = CNNModel(input_size)
    if cuda or mps:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loss, test_loss = 0.0, 0.0
    for epoch in range(epochs):
        for i, batch in enumerate(train_dataloader):
            x = torch.tensor(batch['x'], dtype=torch.float, device=device)
            y = torch.tensor(batch['y'], dtype=torch.long, device=device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'[Epoch: {epoch + 1}] loss: {train_loss / i:.3f}')
        train_loss = 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            x = torch.tensor(batch['x'], dtype=torch.float, device=device, requires_grad=False)
            y = torch.tensor(batch['y'], dtype=torch.long, device=device, requires_grad=False)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y).sum().item()
            total += 1

    print(f'Accuracy: {correct/total}')
    # Save model
    save_state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(save_state, "./models/" + epochs + "_" + save_path)


if __name__ == "__main__":
    # concatenate csv files of demos
    train_test()
