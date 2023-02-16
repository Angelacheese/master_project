from load_dataset import *
from bert_embedding import *
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


# ------------------------------------------------------ training
# Yelp: 400, comment stars > 4, comment useful > 50, business stars > 3.5
txt_path_truthful = '../raw_data/restaurant/truthful'
yelp_path = '../raw_data/domain/results/Yelp_Restaurants_4_50_3.5.csv'


# ------------------------------------------------------ restaurant


bert_truthful = convert_bert_truthful(txt_path_truthful)
bert_yelp = convert_bert_yelp(yelp_path)


# quantity of training comment: truth = 20, yelp = 10
txts_quantity_truthful = 20
dataset = bert_truthful[:txts_quantity_truthful] + bert_yelp


dataset_train = setDataset()
print('dataset_train length:', len(dataset_train))
loader_train = create_loader(dataset)


# hyper-parameter setting
epochs = 10
use_cuda = 1
device = torch.device("cuda" if (
    torch.cuda.is_available() & use_cuda) else "cpu")
ae_train = AutoEncoder().to(device)
optimizer_ae = torch.optim.Adam(ae_train.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_ae, milestones=[10, 40], gamma=0.5)


# train model
loss_data = train_model(device, epochs, loader_train,
                        ae_train, optimizer_ae, scheduler)


# ------------------------------------------------------ testing

# ------------------------------------------------------ restaurant:reataurant

txt_path_deceptive = '../raw_data/restaurant/deceptive_MTurk'
df_txt_deceptive = load_txt_file(txt_path_deceptive)
txt_ids_dict_deceptive, txt_segment_dict_deceptive = txts_convert_dict(
    df_txt_deceptive)
bert_output_deceptive = convert_to_bert(
    txt_ids_dict_deceptive, txt_segment_dict_deceptive)


# quantity of testing comment: truth = 20, deceptive = 10
txts_quantity_deceptive = 10
dataset = bert_truthful[21:40] + \
    bert_output_deceptive[:txts_quantity_deceptive]
dataset_test = setDataset()
print('dataset_test length:', len(dataset_test))
# print('dataset_test[2]:', dataset_test[2])
# print('dataset_test[3]:', dataset_test[3])
loader_test = create_loader(dataset_test)


ae_test = torch.load('autoencoder1.pth')
ae_test.eval()
print(ae_test)


with torch.no_grad():
    loss_function = nn.MSELoss().to(device)
    data_loss = []
    log_loss = []
    total_loss = 0
    for i, data in enumerate(loader_test):
        inputs = data.view(-1, 768).to(device)
        c, outputs = ae_test(inputs)
        loss = loss_function(outputs, inputs)
        data_loss.append(loss.item())
        total_loss += loss

    total_loss /= len(loader_test.dataset)

    print('average loss:', total_loss)
    # print(data_loss)
    plt.plot(data_loss)
    plt.savefig('data_loss_plot1.png')
