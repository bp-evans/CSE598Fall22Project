import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
import pandas as pd
import DeveloperName

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


class DemostrationDataset(Dataset):
    """
    Dataset object for configurations in demonstrations
    """

    def __init__(self, demo):
        """
        Args:
            config_demo (list[(Configuration, Action)]): List of tuple of configuration and action taken from demo.
        """
        self.x = []
        self.y = []
        self.populate_x_y(demo)

    def populate_x_y(self, demo):
        for _, row in demo.iterrows():
            self.x.append(np.fromstring(row[0][1:-1], dtype='float64', sep=' '))
            self.y.append(int(row[1]) - 1)

    def __getitem__(self, index):
        """
        Args:
            index (int): index of the item to be returned

        Returns:
            dict of {x: input vector, y: action category}
        """
        x = self.x[index]
        y = self.y[index]
        return {'x': x, 'y': y}

    def __len__(self):
        """
        Returns len of the dataset
        """
        return len(self.x)


class MLP(nn.Module):
    """
    MLP model for behavior cloning
    """

    def __init__(self, input_shape=4):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return x


def train_test(dataset_path='demo.csv', input_size=4, batch_size=64, lr=0.0001, epochs=1000, save_path='model.pk'):
    """
    Train and test the model and print the accuracies.
    """
    dataset = pd.read_csv(dataset_path, skiprows=1, header=None)
    train_data, test_data = train_test_split(dataset, train_size=0.8)
    train_dataloader = DataLoader(
        DemostrationDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        DemostrationDataset(test_data),
        batch_size=batch_size,
        shuffle=True,
    )
    model = MLP(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_losses = []
        test_losses = []
        model.train()
        for i, batch in enumerate(train_dataloader):
            x = torch.tensor(batch['x'], dtype=torch.float, device=device)
            y = torch.tensor(batch['y'], dtype=torch.long, device=device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                x = torch.tensor(batch['x'], dtype=torch.float, device=device, requires_grad=False)
                y = torch.tensor(batch['y'], dtype=torch.long, device=device, requires_grad=False)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                test_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y).sum().item()
                total += 1

        print('epoch : {}, Mean train loss : {:.4f}, Mean test loss : {:.4f}, acc : {:.2f}%'
              .format(epoch + 1, np.mean(train_losses), np.mean(test_losses), correct / total))

    # Save model
    save_state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(save_state, DeveloperName.my_name + save_path)


if __name__ == "__main__":
    # concatenate csv files of demos
    extension = 'csv'
    csv_files = [i for i in glob.glob('*Demos.{}'.format(extension))]
    demos_concat = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    # export to csv
    demos_concat.to_csv("demos_concat.csv", index=False, encoding='utf-8-sig')
    # train BC on concatenated demos
    train_test('demos_concat.csv')
