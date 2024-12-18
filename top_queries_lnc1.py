import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import random
import argparse

from helpers import (
    load_json_file,
    load_tokenizer_and_models,
    preprocess_queries,
    preprocess_corpus,
    set_seed
)
import TransformerLens.transformer_lens.utils as utils

def compute_relative_scores(perturb_type):
    set_seed(42)
    torch.set_grad_enabled(False)
    device = utils.get_device()

    use_reduced_dataset = False
    n_queries = 1
    n_docs = 100

    fbase_path = "data"
    tfc1_add_queries = pd.read_csv(os.path.join(fbase_path, "tfc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
    tfc1_add_baseline_corpus = load_json_file(os.path.join(fbase_path, "tfc1_add_baseline_final_dd_corpus.json"))["corpus"]
    tfc1_add_dd_corpus = load_json_file(os.path.join(fbase_path, "tfc1_add_{}_final_dd_corpus.json".format(perturb_type)))["corpus"]

    target_qids = tfc1_add_queries["_id"].tolist()
    tfc1_add_queries = tfc1_add_queries[tfc1_add_queries["_id"].isin(target_qids)]
    if use_reduced_dataset:
        target_qids = random.sample(tfc1_add_queries["_id"].tolist(), n_queries)
        tfc1_add_queries = tfc1_add_queries[tfc1_add_queries["_id"].isin(target_qids)]

    pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer, tl_model = load_tokenizer_and_models(pre_trained_model_name, device)
    tl_model.eval()

    # Preprocess queries
    queries_dataloader = preprocess_queries(tfc1_add_queries, tokenizer)

    # Use 'guantanamo' as the filler token
    filler_token_id = tokenizer.convert_tokens_to_ids("guantanamo")
    pad_token_id = tokenizer.pad_token_id

    query_relative_changes = {}

    for qid in tqdm(target_qids):
        # Get query embedding
        q_tokenized = list(filter(lambda item: item["_id"] == qid, queries_dataloader))[0]
        q_tokenized["input_ids"] = q_tokenized["input_ids"].to(device)
        q_tokenized["attention_mask"] = q_tokenized["attention_mask"].to(device)

        q_outputs = tl_model(
            q_tokenized["input_ids"],
            return_type="embeddings",
            one_zero_attention_mask=q_tokenized["attention_mask"],
        )
        q_embedding = q_outputs[:,0,:].squeeze(0) # leave on device

        target_docs = tfc1_add_dd_corpus[str(qid)]
        if use_reduced_dataset:
            target_doc_ids = random.sample(list(target_docs.keys()), n_docs)
            target_docs = {doc_id: target_docs[doc_id] for doc_id in target_doc_ids}

        corpus_dataloader = preprocess_corpus(target_docs, tokenizer)

        doc_relative_changes = []

        for batch in corpus_dataloader:
            try:
                doc_id = batch["_id"][0]
                baseline_doc = tfc1_add_baseline_corpus[str(qid)][doc_id]["text"]
                baseline_tokens = tokenizer(baseline_doc, truncation=True, return_tensors="pt")

                b_len = torch.sum(baseline_tokens["attention_mask"])
                cls_tok = baseline_tokens["input_ids"][0][0]
                sep_tok = baseline_tokens["input_ids"][0][-1]

                if perturb_type == "append":
                    filler_tokens = torch.full((b_len,), pad_token_id)  
                    filler_attn_mask = torch.ones((b_len,), dtype=torch.long)  
                    adj_doc = torch.cat((baseline_tokens["input_ids"][0][1:-1], filler_tokens))
                    baseline_tokens["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).unsqueeze(0)
                    baseline_tokens["attention_mask"] = torch.cat((baseline_tokens["attention_mask"][0], filler_attn_mask), dim=0).unsqueeze(0)
                else: # prepend
                    filler_tokens = torch.full((b_len,), pad_token_id)  
                    filler_attn_mask = torch.ones((b_len,), dtype=torch.long)  
                    adj_doc = torch.cat((filler_tokens, baseline_tokens["input_ids"][0][1:-1]))
                    baseline_tokens["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).unsqueeze(0)
                    baseline_tokens["attention_mask"] = torch.cat((filler_attn_mask, baseline_tokens["attention_mask"][0]), dim=0).unsqueeze(0)

                baseline_tokens["input_ids"] = baseline_tokens["input_ids"].to(device)
                baseline_tokens["attention_mask"] = baseline_tokens["attention_mask"].to(device)

                # Compute embedding with PAD tokens
                baseline_with_pad_embeddings = tl_model(
                    baseline_tokens["input_ids"],
                    one_zero_attention_mask=baseline_tokens["attention_mask"],
                    return_type="embeddings",
                )
                baseline_with_pad_embedding = baseline_with_pad_embeddings[:,0,:].squeeze(0)
                baseline_with_pad_score = torch.matmul(q_embedding, baseline_with_pad_embedding.t()).item()

                # Replace PAD with 'guantanamo'
                perturbed_ids = baseline_tokens["input_ids"].masked_fill(baseline_tokens["input_ids"] == pad_token_id, filler_token_id)
                perturbed_tokens = {
                    "input_ids": perturbed_ids,
                    "attention_mask": torch.ones_like(baseline_tokens["attention_mask"]).to(device)
                }

                perturbed_embeddings = tl_model(
                    perturbed_tokens["input_ids"],
                    one_zero_attention_mask=perturbed_tokens["attention_mask"],
                    return_type="embeddings",
                )
                perturbed_embedding = perturbed_embeddings[:,0,:].squeeze(0)
                perturbed_score = torch.matmul(q_embedding, perturbed_embedding.t()).item()

                # Compute relative change
                change = perturbed_score - baseline_with_pad_score
                if baseline_with_pad_score != 0:
                    relative_change = change / abs(baseline_with_pad_score)
                else:
                    relative_change = 0.0

                doc_relative_changes.append(relative_change)

            except Exception as e:
                print("ERROR: {} for query {} and document {}".format(e, qid, doc_id))

        if len(doc_relative_changes) > 0:
            avg_relative_change = sum(doc_relative_changes) / len(doc_relative_changes)
        else:
            avg_relative_change = 0.0

        query_relative_changes[qid] = avg_relative_change

    return query_relative_changes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute relative changes based on ratio normalization.")
    parser.add_argument("perturb_type", type=str, help="The perturbation type (append or prepend).")

    args = parser.parse_args()

    valid_perturb_types = {"append", "prepend"}
    assert args.perturb_type in valid_perturb_types, f"Invalid argument: perturb_type. Must be one of {valid_perturb_types}."

    fbase_path = "data"
    tfc1_add_queries_global = pd.read_csv(os.path.join(fbase_path, "tfc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
    qid_to_text = {row["_id"]: row["text"] for _, row in tfc1_add_queries_global.iterrows()}

    query_relative_changes = compute_relative_scores(args.perturb_type, qid_to_text)

    # Sort queries by average relative change
    sorted_queries = sorted(query_relative_changes.items(), key=lambda x: x[1], reverse=True)

    top_30 = sorted_queries[:30]
    print("Top 30 queries by relative change:")
    for q, ch in top_30:
        print(q, ch)

    # Write all queries with their relative changes
    all_queries_file = "all_queries_relative_scores.csv"
    with open(all_queries_file, "w") as f:
        f.write("query_id,relative_change,query_text\n")
        for q, ch in sorted_queries:
            query_text = qid_to_text[q] if q in qid_to_text else "unknown query"
            f.write(f"{q},{ch},{query_text}\n")
    bottom_33 = sorted_queries[-33:]
    bottom_33_file = "data/lnc1_add_qids_with_text.csv"
    with open(bottom_33_file, "w") as f:
        for q, (norm_ch, raw_ch) in bottom_33:
            query_text = qid_to_text[q] if q in qid_to_text else "unknown query"
            f.write(f"{q},{query_text}\n")