# file放在venv_master才讀的到，之後可以再試試放在code file裡。
import pandas as pd
import os


def load_txt_file(txt_path):
    files = os.listdir(txt_path)
    txts = {}   # 每一則評論存成dictionary
    index = 0

    for file in files:
        position = txt_path + '/' + file
        with open(position, "r", encoding='windows-1252') as f:
            txt = f.read()
            txts[index] = txt
            index += 1

    return txts


def load_yelp_csv(yelp_path):
    df_yelp = pd.read_csv(yelp_path, encoding='utf-8')
    print("dfYelpRestaurant:", df_yelp.shape)

    return df_yelp
