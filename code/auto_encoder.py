from bert_embedding import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from torch.nn import functional as F


class aeEncoder(nn.Module):
    def __init__(self):
        super(aeEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Tanh(),
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        return codes


class aeDecoder(nn.Module):
    def __init__(self):
        super(aeDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 768),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outputs = self.decoder(inputs)
        return outputs


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = aeEncoder()
        self.decoder = aeDecoder()

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        return codes, decoded


def create_loader(dataset, batch):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=False
    )

    return dataloader


def train_model(device, epochs, dataloader, ae_train, optimizer_ae, scheduler):
    loss_function = nn.MSELoss().to(device)
    epochs_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        data_loss = []

        for data in dataloader:
            inputs = data.view(-1, 768).to(device)
            ae_train.zero_grad()

            # forward
            c, decoded = ae_train(inputs)
            loss = loss_function(decoded, inputs)
            loss.backward(retain_graph=True)
            optimizer_ae.step()
            epoch_loss += loss
            data_loss.append(loss.item())

        epoch_loss /= len(dataloader.dataset)
        epochs_loss.append(epoch_loss.item())
        scheduler.step()
        print('[{}/{}] Loss:'.format(epoch+1, epochs), epoch_loss.item())

    torch.save(ae_train, 'autoencoder_4.pth')
    return epochs_loss, data_loss
