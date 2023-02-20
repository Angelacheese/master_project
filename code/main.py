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


# ------------------------------------------------------ preprocessing: convert to bert
# restaurant
# Yelp: 400, comment stars > 4, comment useful > 50, business stars > 3.5
txt_path_truthful = '../raw_data/restaurant/truthful'
yelp_path = '../raw_data/domain/results/Yelp_Restaurants_4_50_3.5.csv'


def convert_bert_txts(txt_path):
    df_txt = load_txt_file(txt_path)
    txt_ids_dict, txt_segment_dict = txts_convert_dict(df_txt)
    bert_truthful = convert_to_bert(txt_ids_dict, txt_segment_dict)
    return bert_truthful


def convert_bert_yelp(yelp_path):
    df_yelp = load_yelp_csv(yelp_path)
    yelp_ids_dict, yelp_segment_dict = yelp_convert_dict(
        df_yelp, 10)   # quantity of training: yelp = 10
    bert_yelp = convert_to_bert(yelp_ids_dict, yelp_segment_dict)
    return bert_yelp


bert_truthful = convert_bert_txts(txt_path_truthful)
bert_yelp = convert_bert_yelp(yelp_path)

# ------------------------------------------------------ training
# quantity of training comment: truth = 10, yelp = 10
txts_quantity_truthful = 10
dataset = bert_truthful[:txts_quantity_truthful] + bert_yelp

batch = 1
dataset_train = setDataset()    # trainingset
print('dataset_train length:', len(dataset_train))
loader_train = create_loader(dataset, batch)   # put in data loader


epochs = 10
use_cuda = 1
device = torch.device("cuda" if (
    torch.cuda.is_available() & use_cuda) else "cpu")
ae_train = AutoEncoder().to(device)
optimizer_ae = torch.optim.Adam(ae_train.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_ae, milestones=[10, 40], gamma=0.5)


epochs_loss, data_loss = train_model(device, epochs, loader_train,
                                     ae_train, optimizer_ae, scheduler)  # train model


# ------------------------------------------------------ preprocessing: convert to bert
txt_path_deceptive = '../raw_data/restaurant/deceptive_MTurk'
bert_deceptive = convert_bert_txts(txt_path_deceptive)

# ------------------------------------------------------ testing
# txt_path_deceptive = '../raw_data/restaurant/deceptive_MTurk'
# df_txt_deceptive = load_txt_file(txt_path_deceptive)
# txt_ids_dict_deceptive, txt_segment_dict_deceptive = txts_convert_dict(
#     df_txt_deceptive)
# bert_output_deceptive = convert_to_bert(
#     txt_ids_dict_deceptive, txt_segment_dict_deceptive)


# quantity of testing comment: truth = 20, deceptive = 10
txts_quantity_deceptive = 10
dataset = bert_truthful[21:40] + \
    bert_deceptive[:txts_quantity_deceptive]
dataset_test = setDataset()
print('dataset_test length:', len(dataset_test))
loader_test = create_loader(dataset_test, batch)


ae_test = torch.load('autoencoder_1.pth')
ae_test.eval()
print(ae_test)


with torch.no_grad():
    loss_function = nn.MSELoss().to(device)
    data_loss_test = []
    total_loss = 0
    for i, data in enumerate(loader_test):
        inputs = data.view(-1, 768).to(device)
        c, outputs = ae_test(inputs)
        loss = loss_function(outputs, inputs)
        data_loss_test.append(loss.item())
        total_loss += loss

    total_loss /= len(loader_test.dataset)
    print('average loss:', total_loss)
    plt.plot(data_loss_test)
    plt.savefig('plot1.png')
