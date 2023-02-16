# tqdm(range(1, epochs+1))
# bert處理後
from bert_embedding import *
import matplotlib.pyplot as plt
import tqdm
# torch
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
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        return codes


class aeDecoder(nn.Module):
    def __init__(self):
        super(aeDecoder, self).__init__()

        self.decoder = nn.Sequential(
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


def create_loader(dataset):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )

    return dataloader


def train_model(device, epochs, dataloader, ae_train, optimizer_ae, scheduler):
    loss_function = nn.MSELoss().to(device)
    log_loss = []

    for epoch in range(epochs):
        total_loss = 0
        data_loss = []

        for data in dataloader:
            inputs = data.view(-1, 768).to(device)
            ae_train.zero_grad()

            # forward
            codes, decoded = ae_train(inputs)
            loss = loss_function(decoded, inputs)
            loss.backward(retain_graph=True)
            optimizer_ae.step()
            total_loss += loss
            data_loss.append(loss.item())

        total_loss /= len(dataloader.dataset)
        log_loss.append(total_loss.item())
        scheduler.step()
        print('[{}/{}] Loss:'.format(epoch+1, epochs), total_loss.item())

    print('average loss:', log_loss)
    plt.plot(data_loss)
    torch.save(ae_train, 'autoencoder1.pth')  # Save
