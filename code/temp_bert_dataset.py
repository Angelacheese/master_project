from load_dataset import *
from bert_embedding import *
import torch


def convert_bert_txts(txt_path):
    df_txt = load_txt_file(txt_path)
    txt_ids_dict, txt_segment_dict = txts_convert_dict(df_txt)
    bert_truthful = convert_to_bert(txt_ids_dict, txt_segment_dict)
    return bert_truthful


def convert_bert_yelp(yelp_path):
    df_yelp = load_yelp_csv(yelp_path)
    yelp_ids_dict, yelp_segment_dict = yelp_convert_dict(
        df_yelp, 100)   # quantity of training: yelp = 100
    bert_yelp = convert_to_bert(yelp_ids_dict, yelp_segment_dict)
    return bert_yelp


# txt_path_truthful = '../raw_data/restaurant/truthful'
# yelp_path = '../raw_data/domain/results/Yelp_Restaurants_4_50_3.5.csv'
# txt_path_deceptive = '../raw_data/restaurant/deceptive_MTurk'


# restaurant_truthful = convert_bert_txts(txt_path_truthful)
# restaurant_yelp_100 = convert_bert_yelp(yelp_path)
# restaurant_deceptive = convert_bert_txts(txt_path_deceptive)


# torch.save(restaurant_truthful, 'restaurant_truthful.pt')
# torch.save(restaurant_yelp_100, 'restaurant_yelp_100.pt')
# torch.save(restaurant_deceptive, 'restaurant_deceptive.pt')
