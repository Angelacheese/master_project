# master_project

## code

### 1. load_dataset

> load_txt_file(txt_path) txts
>
> load_yelp_csv(yelp_path) df_yelp

### 2. bert_embedding

> txts_convert_dict(df_txt_comments) ids_embeddings_dir, segment_embeddings_dir
>
> yelp_convert_dict(df_yelp_comments, quantity) ids_embeddings_dir, segment_embeddings_dir
>
> convert_to_bert(ids_embeddings_dir, segment_embeddings_dir) bert_output
>
> convert_bert_truthful(txt_path_truthful) bert_truthful
>
> convert_bert_yelp(yelp_path) bert_yelp

### 3. auto_encoder

> class aeEncoder(nn.Module)
>
> class aeDncoder(nn.Module)
>
> class AutoEncoder(nn.Module)
>
> create_loader(dataset) dataloader
>
> train_model(device, epochs, dataloader, ae_train, optimizer_ae, scheduler)

### Main

> class setDataset(Dataset)
