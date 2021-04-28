import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import FFNET


class ToyDataSet(Dataset):
    """
    class that defines what a data-sample looks like
    In the __init__ you could for example load in the data from file
    and then return specific items in __getitem__
    and return the length in __len__
    """

    def __init__(self, length: int):
        """ loads all stuff relevant for dataset """

        # save the length, usually depends on data-file but here data is generated instead
        self.length = length

        # generate random binary labels
        self.classes = [random.choice([0, 1]) for _ in range(length)]

        # generate data from those labels
        self.data = [np.random.normal(self.classes[i], 0.15, 128) for i in range(length)]

    def __getitem__(self, item_index):
        """ defines how to get one sample """

        class_ = torch.tensor(self.classes[item_index])  # python scalar to torch tensor
        tensor = torch.from_numpy(self.data[item_index])  # numpy array/tensor to torch array/tensor
        return tensor, class_

    def __len__(self):
        """ defines how many samples in an epoch, independently of batch size"""

        return self.length


def get_accuracy(output, y):
    """ calculates accuracy """
    predictions = output.argmax(dim=-1, keepdim=True).view_as(y)
    correct = y.eq(predictions).sum().item()
    return correct / output.shape[0]


def main():

    # built tensorboard writer
    writer = SummaryWriter(f"./logs/{str(datetime.now()).split(' ')[1]}")

    # setup
    model = FFNET()
    loss = nn.CrossEntropyLoss()
    dataloader_train = DataLoader(
        ToyDataSet(10000),
        batch_size=64,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # set buffers
    loss_buffer = []
    acc_buffer = []

    for epoch in range(10):
        iterator = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for step, (batch_inputs, batch_targets) in iterator:

            step = step + len(dataloader_train) * epoch

            # forward pass
            optimizer.zero_grad()
            model.apply_mask()
            prediction = model.forward(batch_inputs.float())
            loss_ = loss.forward(prediction, batch_targets)

            # backward pass
            loss_.backward()
            model.apply_mask()
            optimizer.step()

            # metrics
            with torch.no_grad():
                accuracy = get_accuracy(prediction, batch_targets)
                iterator.set_description(str(dict(epoch=epoch, accuracy=accuracy, loss=loss_.item())))
                acc_buffer.append(accuracy)
                loss_buffer.append(loss_.item())

                if (step % 10) == 0:
                    writer.add_scalar("stats/loss", np.mean(loss_buffer), step)
                    writer.add_scalar("stats/acc", np.mean(acc_buffer), step)
                    writer.add_scalar("pruning/unstructured", model.unstructured_sparsity, step)
                    writer.add_scalar("pruning/structured", model.structured_sparsity, step)

                    acc_buffer = []
                    loss_buffer = []

                    writer.flush()

        # prune 7.5 %
        with torch.no_grad():
            model.magnitude_prune_unstructured(0.075)

    writer.close()


if __name__ == '__main__':
    main()
