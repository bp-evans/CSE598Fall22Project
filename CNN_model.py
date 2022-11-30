import argparse
import glob
import math
import os
import time
import pandas as pd
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
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
    def __init__(self, image_folder, annotations_file, transforms_=None, data_size=None):
        self.transform = transforms_
        self.folder = image_folder
        self.labels = pd.read_csv(annotations_file)
        if data_size is not None:
            self.labels = self.labels[:data_size]

    def __getitem__(self, index):
        # Grab the image associated with the label at index idx
        img_path = os.path.join(self.folder, self.labels["Image Name"][index])
        image = read_image(img_path).to(device=device)
        label = self.labels["Label"][index]

        if self.transform is not None:
            image = self.transform(image)

        return {"x":image, "y":label}

    def __len__(self):
        return len(self.labels)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        def new_shape_from_convpool(shape, conv: nn.Conv2d, pool: nn.MaxPool2d):
            # Figure out the new shape from the conv and maxpool layers: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
            shape = (
            math.floor((shape[0] + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) / conv.stride[0] + 1),
            math.floor((shape[1] + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) / conv.stride[1] + 1))
            shape = (
                math.floor((shape[0] + 2 * pool.padding - pool.dilation * (pool.kernel_size - 1) - 1) / pool.stride + 1),
                math.floor((shape[1] + 2 * pool.padding - pool.dilation * (pool.kernel_size - 1) - 1) / pool.stride + 1)
            )
            return shape

        resized_shape = (256,256)
        conv1 = nn.Conv2d(3, 64, 5, stride=2, padding=1)
        pool1 = nn.MaxPool2d(2, stride=2)
        shape = resized_shape

        shape = new_shape_from_convpool(shape, conv1, pool1)
        conv2 = nn.Conv2d(conv1.out_channels, 128, 5, stride=2, padding=1)
        pool2 = nn.MaxPool2d(2, stride=2)
        shape = new_shape_from_convpool(shape, conv2, pool2)

        linear1 = nn.Linear(conv2.out_channels * shape[0] * shape[1], 64)
        linear2 = nn.Linear(linear1.out_features, 4)

        self.convolutions = nn.Sequential(
            # Image transforms
            transforms.Resize(resized_shape),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

            conv1,
            nn.ReLU(inplace=True),
            pool1,
            conv2,
            nn.ReLU(inplace=True),
            pool2,

            nn.InstanceNorm2d(conv2.out_channels)
        )

        self.fc = nn.Sequential(
            linear1,
            nn.ReLU(inplace=True),
            linear2,
            nn.ReLU(inplace=True),
        )

    def forward(self, img):
        x = self.convolutions(img)
        # Flatten the cnn output. If there is a batch layer, then do not flatten that.
        x = torch.flatten(x, start_dim=1 if len(x.shape) > 3 else 0)
        x = self.fc(x)
        return x


def train_test(csv_path='ImageLabels.csv', batch_size=64, lr=0.0001,
               epochs=100, save_path='cnn_model.pk', image_folder="image_demos/", data_size=100000):
    identifier = input("Identifier for this training run: ")
    dataset = ImageDataset(image_folder=image_folder, annotations_file=csv_path, transforms_=None, data_size=data_size)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = CNNModel()
    if cuda or mps:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loss, test_loss = 0.0, 0.0
    for epoch in range(epochs):
        start = time.time()
        for i, batch in enumerate(train_dataloader):
            x = batch["x"].to(device=device)
            y = batch["y"].to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'[Epoch: {epoch + 1}] loss: {train_loss / i:.3f}, elapsed time: {time.time() - start:.2f} sec')
        train_loss = 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            x = batch["x"].to(device=device)
            y = batch["y"].to(device=device, dtype=torch.long)
            # x = torch.tensor(batch['x'], dtype=torch.float, device=device, requires_grad=False)
            # y = torch.tensor(batch['y'], dtype=torch.long, device=device, requires_grad=False)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y).sum().item()
            total += 1
    accuracy = correct / total
    print(f'Accuracy: {accuracy}')
    scripted_model = torch.jit.script(model)
    if not os.path.exists("./models/"):
        os.makedirs("./models/")
    save_path2 = f"./models/ts_{epochs}_{identifier}_{accuracy:.3f}.pt"
    scripted_model.save(save_path2)
    print("Saved JIT of model to " + save_path2)
    # Save model
    save_state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(save_state, "./models/" + str(epochs) + "_" + identifier + "_" + save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='CNN model',
        description='Train CNN model')

    parser.add_argument('-e', '--epochs', action='store', type=int,
                        default=100, help='number of epochs to train')
    parser.add_argument('-s', '--data_size', action='store', type=int,
                        default=100000, help='max number of training examples')
    parser.add_argument('-l', '--dataset_label', help="The label of the dataset to use", default="")
    args = parser.parse_args()
    train_test(csv_path=f"{args.dataset_label}ImageLabels.csv", image_folder=f"{args.dataset_label}{'_' if not args.dataset_label == '' else ''}image_demos", epochs=args.epochs, data_size=args.data_size)
