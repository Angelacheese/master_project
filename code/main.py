from load_dataset import *
from bert_embedding import *
from auto_encoder import *
import matplotlib.pyplot as plt
from transformers import logging


logging.set_verbosity_warning()
logging.set_verbosity_error()


# ------------------------------------------------------ dataset

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

## ------------------------------------------------------ restaurant

# data pre-processing
txt_path_truthful = '../raw data/restaurant/truthful'
df_txt_truthful = load_txt_file(txt_path_truthful)
txt_ids_dict_truthful, txt_segment_dict_truthful = txts_convert_dict(
    df_txt_truthful)
bert_output_truthful = convert_to_bert(
    txt_ids_dict_truthful, txt_segment_dict_truthful)
# Yelp: 400, comment stars > 4, comment useful > 50, business stars > 3.5
yelp_path = '../raw data/domain/results/Yelp_Restaurants_4_50_3.5.csv'
df_yelp = load_yelp_csv(yelp_path)
yelp_ids_dict, yelp_segment_dict = yelp_convert_dict(df_yelp, 400)
bert_output_yelp = convert_to_bert(yelp_ids_dict, yelp_segment_dict)


# quantity of training comment: truth = 20, yelp = 400
txts_quantity_truthful = 20
dataset = bert_output_truthful[:txts_quantity_truthful] + bert_output_yelp
dataset_train = setDataset()
print('dataset_train length:', len(dataset_train))
print('dataset_train[201]:', dataset_train[200])
print('dataset_train[201]:', dataset_train[201])
loader_train = create_loader(dataset)


# hyper-parameter setting
epochs = 1
use_cuda = 1
device = torch.device("cuda" if (
    torch.cuda.is_available() & use_cuda) else "cpu")
ae_train = AutoEncoder().to(device)
optimizer_ae = torch.optim.Adam(ae_train.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_ae, milestones=[10, 40], gamma=0.5)


# train model
train_model(device, epochs, loader_train, ae_train, optimizer_ae, scheduler)

# ------------------------------------------------------ testing

# ------------------------------------------------------ restaurant:reataurant

txt_path_deceptive = '../raw data/restaurant/deceptive_MTurk'
df_txt_deceptive = load_txt_file(txt_path_deceptive)
txt_ids_dict_deceptive, txt_segment_dict_deceptive = txts_convert_dict(
    df_txt_deceptive)
bert_output_deceptive = convert_to_bert(
    txt_ids_dict_deceptive, txt_segment_dict_deceptive)


# quantity of testing comment: truth = 200, deceptive = 100
txts_quantity_deceptive = 100
dataset = bert_output_truthful + \
    bert_output_deceptive[:txts_quantity_deceptive]
dataset_test = setDataset()
print('dataset_test length:', len(dataset_test))
print('dataset_test[200]:', dataset_test[200])
print('dataset_test[201]:', dataset_test[201])
loader_test = create_loader(dataset_test)


ae_test = torch.load('autoencoder16.pth')
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
    plt.savefig('data_loss_test_plot16.png')


# ------------------------------------------------------ restaurant:hotel

# txt_path_truthful1 = '../raw data/hotel/negative/truthful'
# df_txt_truthful1 = load_txt_file(txt_path_truthful1)
# txt_ids_dict_truthful1, txt_segment_dict_truthful1 = txts_convert_dict(
#     df_txt_truthful1)
# bert_output_truthful1 = convert_to_bert(
#     txt_ids_dict_truthful1, txt_segment_dict_truthful1)

# txt_path_deceptive = '../raw data/hotel/negative/deceptive_MTurk'
# df_txt_deceptive = load_txt_file(txt_path_deceptive)
# txt_ids_dict_deceptive, txt_segment_dict_deceptive = txts_convert_dict(
#     df_txt_deceptive)
# bert_output_deceptive = convert_to_bert(
#     txt_ids_dict_deceptive, txt_segment_dict_deceptive)


# # quantity of testing comment: truth = 200, deceptive = 100
# txts_quantity_deceptive = 100
# dataset = bert_output_truthful1 + \
#     bert_output_deceptive[:txts_quantity_deceptive]
# dataset_test = setDataset()
# print('dataset_test length:', len(dataset_test))
# print('dataset_test[200]:', dataset_test[200])
# print('dataset_test[201]:', dataset_test[201])
# loader_test = create_loader(dataset_test)


# ae_test = torch.load('autoencoder.pth')
# ae_test.eval()
# print(ae_test)


# with torch.no_grad():
#     loss_function = nn.MSELoss().to(device)
#     data_loss = []
#     log_loss = []
#     total_loss = 0
#     for i, data in enumerate(loader_test):
#         inputs = data.view(-1, 768).to(device)
#         c, outputs = ae_test(inputs)
#         loss = loss_function(outputs, inputs)
#         data_loss.append(loss.item())
#         total_loss += loss

#     total_loss /= len(loader_test.dataset)

#     print('average loss:', total_loss)
#     # print(data_loss)
#     plt.plot(data_loss)
#     plt.savefig('data_loss_test_hotel_neg.png')
