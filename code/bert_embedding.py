# # dataset
# from load_dataset import *
# torch
import torch
from torch.utils.data import Dataset, DataLoader
# BERT
from transformers import BertTokenizer, BertModel
from IPython.display import clear_output


pretrained_model = "bert-base-cased"  # 指定預訓練模型
tokenizer = BertTokenizer.from_pretrained(pretrained_model)  # 分詞器
bert_model = BertModel.from_pretrained(
    pretrained_model, output_attentions=True)
clear_output()  # 怕輸出太多文件太大，及時清除 notebook 的輸出


def txts_convert_dict(df_txt_comments):
    # BERT tokens: [CLS] = 101, [SEP] = 102
    ids_embeddings_dir = {}
    ids_index = 0
    segment_embeddings_dir = {}
    segment_index = 0
    maxlen = 100

    for num in df_txt_comments:
        comment = df_txt_comments[num]
        temp_sentence = []

        # 前處理: split, tokenize
        for sentence in comment.lower().split("."):
            token_sentence = tokenizer.tokenize(sentence)
            token_sentence += ["[SEP]"]
            temp_sentence += token_sentence
        token_comment = ["[CLS]"] + temp_sentence

        # 檢查長度符合 maxlen: ['PAD']
        if len(token_comment) < maxlen:
            token_comment = token_comment + \
                ['[PAD]'] * (maxlen - len(token_comment))
        else:
            token_comment = token_comment[:maxlen]

        # 製作 ids, segment masks
        segment_masks = [1 if token !=
                         '[PAD]' else 0 for token in token_comment]
        segment_embeddings_dir[segment_index] = segment_masks
        ids_masks = tokenizer.convert_tokens_to_ids(token_comment)
        ids_embeddings_dir[ids_index] = ids_masks
        segment_index += 1
        ids_index += 1

    return ids_embeddings_dir, segment_embeddings_dir


def yelp_convert_dict(df_yelp_comments, quantity):
    # BERT tokens: [CLS] = 101, [SEP] = 102
    ids_embeddings_dir = {}
    ids_index = 0
    segment_embeddings_dir = {}
    segment_index = 0
    maxlen = 100

    for comment in df_yelp_comments.text[:quantity]:
        temp_sentence = []

        # 前處理: split, tokenize
        for sentence in comment.lower().split("."):
            token_sentence = tokenizer.tokenize(sentence)
            token_sentence += ["[SEP]"]
            temp_sentence += token_sentence
        token_comment = ["[CLS]"] + temp_sentence

        # 檢查長度符合 maxlen: ['PAD']
        if len(token_comment) < maxlen:
            token_comment = token_comment + \
                ['[PAD]'] * (maxlen - len(token_comment))
        else:
            token_comment = token_comment[:maxlen]

        # 製作 ids, segment masks
        segment_masks = [1 if token !=
                         '[PAD]' else 0 for token in token_comment]
        segment_embeddings_dir[segment_index] = segment_masks
        ids_masks = tokenizer.convert_tokens_to_ids(token_comment)
        ids_embeddings_dir[ids_index] = ids_masks
        segment_index += 1
        ids_index += 1

    return ids_embeddings_dir, segment_embeddings_dir


# 把轉成 bert 後的 ids 轉 tensor 型式
def convert_to_bert(ids_embeddings_dir, segment_embeddings_dir):
    bert_output = []

    for num in range(len(ids_embeddings_dir)-1):
        ids_embeddings_dir[num] = torch.tensor(ids_embeddings_dir[num])
        ids_embeddings_dir[num] = torch.reshape(
            ids_embeddings_dir[num], (1, -1))
        segment_embeddings_dir[num] = torch.tensor(segment_embeddings_dir[num])
        segment_embeddings_dir[num] = torch.reshape(
            segment_embeddings_dir[num], (1, -1))
        output = bert_model(
            ids_embeddings_dir[num], attention_mask=segment_embeddings_dir[num])

        bert_output.append(output[0])
    return bert_output


# label = torch.ones(len(ids_embeddings_dir))
# label
