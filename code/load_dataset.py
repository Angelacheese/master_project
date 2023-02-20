import pandas as pd
import os


def load_txt_file(txt_path):    # 每一則評論存成dictionary
    files = os.listdir(txt_path)
    dir_txts = {}
    index = 0

    for file in files:
        position = txt_path + '/' + file
        with open(position, "r", encoding='windows-1252') as f:
            txt = f.read()
            dir_txts[index] = txt
            index += 1
    return dir_txts


def load_yelp_csv(yelp_path):    # 每一則評論存成dataframe
    df_yelp = pd.read_csv(yelp_path, encoding='utf-8')
    print("dfYelpRestaurant:", df_yelp.shape)
    return df_yelp
