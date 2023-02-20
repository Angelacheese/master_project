from auto_encoder import *
import matplotlib.pyplot as plt


# ------------------------------------------------------ set_dataset
class setDataset(Dataset):
    def __init__(self):
        self.x = dataset
        # self.y = label

    def __getitem__(self, index):
        return self.x[index]
        # self.y[index]

    def __len__(self):
        return len(self.x)


restaurant_truthful = torch.load("restaurant_truthful.pt")
restaurant_yelp = torch.load("restaurant_yelp_100.pt")
restaurant_deceptive = torch.load("restaurant_deceptive.pt")

# ------------------------------------------------------ training
quantity_truthful = 10
dataset = restaurant_truthful[:quantity_truthful] + restaurant_yelp

batch = 1
dataset_train = setDataset()
print('dataset_train length:', len(dataset_train))
loader_train = create_loader(dataset, batch)


epochs = 100
use_cuda = 1
device = torch.device("cuda" if (
    torch.cuda.is_available() & use_cuda) else "cpu")
ae_train = AutoEncoder().to(device)
optimizer_ae = torch.optim.Adam(ae_train.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_ae, milestones=[10, 40], gamma=0.5)


epochs_loss_train, data_loss_train = train_model(device, epochs, loader_train,
                                                 ae_train, optimizer_ae, scheduler)  # train model


# ------------------------------------------------------ testing
# quantity of testing comment: truth = 20, deceptive = 10
quantity_deceptive = 10
dataset = restaurant_truthful[21:40] + \
    restaurant_deceptive[:quantity_deceptive]
dataset_test = setDataset()
print('dataset_test length:', len(dataset_test))
loader_test = create_loader(dataset_test, batch)


ae_test = torch.load('autoencoder_4.pth')
ae_test.eval()
print(ae_test)


with torch.no_grad():
    loss_function = nn.MSELoss().to(device)
    data_loss_test = []
    for i, data in enumerate(loader_test):
        inputs = data.view(-1, 768).to(device)
        c, outputs = ae_test(inputs)
        loss = loss_function(outputs, inputs)
        data_loss_test.append(loss.item())

    print('data loss:', data_loss_test)

    plt.title("autoencoder loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(epochs_loss_train, color="red", label="y1")
    plt.plot(data_loss_test, ls="--", label="y2")
    plt.savefig('plot4.png')
    plt.show()
