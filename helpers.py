import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
import random

from transformers import AutoTokenizer, AutoModel
from datasets import Value, Dataset
from torch.utils.data import DataLoader

from TransformerLens.transformer_lens import HookedEncoder


# Function to load JSON file into a Python dictionary
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        # Load JSON data into a dictionary
        data = json.load(file)
    return data

def load_tokenizer_and_models(hf_model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModel.from_pretrained(hf_model_name)
    hf_model.to(device)

    tl_model = HookedEncoder.from_pretrained(hf_model_name, device=device, hf_model=hf_model)
    tl_model.to(device)

    return tokenizer, tl_model


def preprocess(dataset, tokenizer, remove_cols=["title", "text"]):
    def tokenize_fn(inputs):
        return tokenizer(inputs["text"], truncation=True)

    tokenized_dataset = dataset.cast_column("_id", Value(dtype="int32"))
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)

    # Convert to torch tensors
    tokenized_dataset = tokenized_dataset.with_format("torch", columns=["input_ids", "attention_mask", "_id"])

    return tokenized_dataset


def preprocess_queries(queries_df, tokenizer):
    dataset = Dataset.from_pandas(queries_df)    
    tokenized_dataset = preprocess(dataset, tokenizer, remove_cols=["text"])
    dataloader = DataLoader(tokenized_dataset, batch_size=1)

    return dataloader


def preprocess_corpus(corpus_dict, tokenizer):
    corpus_df = pd.DataFrame.from_dict(corpus_dict, orient="index")
    corpus_df.index.name = "_id"
    dataset = Dataset.from_pandas(corpus_df)
    tokenized_dataset = preprocess(dataset, tokenizer, remove_cols=["text"])
    dataloader = DataLoader(tokenized_dataset, batch_size=1)

    return dataloader


'''
Encoding loop for Huggingface models
'''
def encode_hf(model, dataloader, device):
  result = np.empty((0,768))
  all_labels = np.empty(0)

  # send batch to device
  for i, batch in enumerate(tqdm(dataloader)):
    labels = batch.pop("_id")
    batch = {k: v.to(device) for k, v in batch.items()}

    # set model to eval and disable gradient calculations
    model.eval()
    with torch.no_grad():
      # [0] selects the hidden states
      # [:,0,:] selects the CLS token for each vector
      outputs = model(**batch)[0][:,0,:].squeeze(0)

    embeddings = outputs.detach().cpu().numpy()
    # labels = batch["_ids"].detach().cpu().numpy()

    result = np.concatenate((result, embeddings), axis=0)
    all_labels = np.concatenate((all_labels, labels))

  return result, np.asarray(all_labels)


'''
Encoding loop for TransformerLens models
'''
def encode_tl(model, dataloader):
    result = np.empty((0,768))
    all_labels = np.empty(0)
   
    for _, batch in enumerate(tqdm(dataloader)):
        labels = batch.pop("_id")

        # get input ids and attention masks
        input_ids = batch.get("input_ids")
        attn_mask = batch.get("attention_mask")

        outputs = model(input_ids, return_type="embeddings", one_zero_attention_mask=attn_mask)
        embeddings = outputs[:,0,:].squeeze(0).detach().cpu().numpy()

        result = np.concatenate((result, embeddings), axis=0)
        all_labels = np.concatenate((all_labels, labels))


    return result, np.asarray(all_labels)


# Compute ranking scores using dot product
def compute_ranking_scores(query_embedding, doc_embeddings, doc_ids):
    scores = torch.matmul(query_embedding, doc_embeddings.T)
    
    # sort scores and labels
    # sorted_idx = torch.argsort(scores)[::-1]

    # Sort scores and corresponding doc_ids in descending order
    sorted_idx = torch.argsort(scores, descending=True)
    sorted_scores = scores[sorted_idx]

    # Convert indices to list of IDs (doc_ids are strings)
    sorted_doc_ids = [doc_ids[i] for i in sorted_idx.cpu().numpy()]

    # return scores[sorted_idx], doc_ids[sorted_idx].astype(int)
    return sorted_scores, sorted_doc_ids

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)