import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.io import read_image
import numpy as np
import os
import glob
import numpy as np
import DeveloperName
import os
from torch.utils.data import Subset, random_split


cuda = torch.cuda.is_available()
# For macos-arm64 support
mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
if cuda:
    device = torch.device('gpu')
elif mps:
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"PyTorch using device {device}")


class ImageDemonstrationDataset(Dataset):
    """
    A dataset for demonstrations given as images
    """
    def __init__(self, annotations_file="ImageLabels.csv", img_dir="image_demos/"):
        # Pull in the image demonstration dataset from disk
        # self.data = np.load("Images", mmap_mode='r')
        # ImageLabels should be a n*2 table where each entry is the filename and then the label
        self.labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Grab the image associated with the label at index idx
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx,1])
        image = read_image(img_path)
        label = self.labels.iloc[idx, 2]

        # TODO: Maybe support transform here?

        return image, label


class ImageClassifier(nn.Module):
    """
    A classifier for the whole pygame image space
    """

    def __init__(self, input_shape: np.shape = (3, 500, 800)):
        super(ImageClassifier, self).__init__()
        self.layers = nn.Sequential(

        )
        pass

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return x


def train(batch_size=64, lr=0.0001, epochs=1000, save_path='model.pk'):
    # Get the dataset
    dataset = ImageDemonstrationDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = ImageClassifier()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_losses = []
        test_losses = []
        model.train()
        for i, batch in enumerate(train_dataloader):
            x = batch[0]
            y = batch[1]
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                x = batch[0]
                y = batch[1]
                outputs = model(x)
                loss = loss_fn(outputs, y)
                test_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y).sum().item()
                total += 1

        print('epoch : {}, Mean train loss : {:.4f}, Mean test loss : {:.4f}, acc : {:.2f}%'
              .format(epoch + 1, np.mean(train_losses), np.mean(test_losses), correct / total))


if __name__=="__main__":
    train()