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

def compute_all_scores(perturb_type):
    set_seed
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

    # Use 'guantanamo' as the filler token (example)
    filler_token_id = tokenizer.convert_tokens_to_ids("guantanamo")
    a_token_id = tokenizer.convert_tokens_to_ids("a")
    pad_token_id = tokenizer.pad_token_id
    
    all_scores = []
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

        for batch in corpus_dataloader:
            try:
                doc_id = batch["_id"][0]
                baseline_doc = tfc1_add_baseline_corpus[str(qid)][doc_id]["text"]
                baseline_tokens = tokenizer(baseline_doc, truncation=True, return_tensors="pt")
                # Get score after injecting query term
                perturbed_embeddings = tl_model(
                    batch["input_ids"],
                    one_zero_attention_mask=batch["attention_mask"],
                    return_type="embeddings",
                )

                perturbed_embedding = perturbed_embeddings[:,0,:].squeeze(0)
                orig_perturbed_score = torch.matmul(q_embedding, perturbed_embedding.t()).item()


                # Compute baseline score (no fillers)
                baseline_embeddings = tl_model(
                    baseline_tokens["input_ids"].to(device),
                    one_zero_attention_mask=baseline_tokens["attention_mask"].to(device),
                    return_type="embeddings",
                )
                baseline_embedding = baseline_embeddings[:,0,:].squeeze(0)
                baseline_score = torch.matmul(q_embedding, baseline_embedding.t()).item()

                # Now add padding tokens as per perturb_type
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

                baseline_copy = baseline_tokens.copy()

                # Replace PAD with 'guantanamo'
                perturbed_ids = baseline_copy["input_ids"].masked_fill(baseline_copy["input_ids"] == pad_token_id, filler_token_id)
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
                # Replace PAD with 'a'
                perturbed_ids = baseline_tokens["input_ids"].masked_fill(baseline_tokens["input_ids"] == pad_token_id, a_token_id)
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
                perturbed_score_a = torch.matmul(q_embedding, perturbed_embedding.t()).item()
                all_scores.append((baseline_score,baseline_with_pad_score,orig_perturbed_score,perturbed_score,perturbed_score_a))

            except Exception as e:
                print("ERROR: {} for query {} and document {}".format(e, qid, doc_id))



    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute relative changes based on ratio normalization.")
    parser.add_argument("perturb_type", type=str, help="The perturbation type (append or prepend).")

    args = parser.parse_args()

    valid_perturb_types = {"append", "prepend"}
    assert args.perturb_type in valid_perturb_types, f"Invalid argument: perturb_type. Must be one of {valid_perturb_types}."

    # Load queries to build qid_to_text
    fbase_path = "data"
    tfc1_add_queries_global = pd.read_csv(os.path.join(fbase_path, "tfc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
    qid_to_text = {row["_id"]: row["text"] for _, row in tfc1_add_queries_global.iterrows()}

    all_scores = compute_all_scores(args.perturb_type)

    # Save all scores to a CSV file
    output_csv_path = f"all_scores_{args.perturb_type}.csv"
    df_scores = pd.DataFrame(all_scores, columns=["Baseline score", "Baseline Score after padding", "Perturbed score with query term","Perturbed score with 'guantanamo' tokens", "Perturbed score with 'a' tokens"])
    df_scores.to_csv(output_csv_path, index=False)
    print(f"Scores saved to {output_csv_path}")

    # Compute average scores
    avg_scores = df_scores.mean()

    # Plot the averages with adjusted scale and value labels
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    bars = plt.bar(avg_scores.index, avg_scores.values, color='skyblue')
    plt.title(f"Average Scores for Perturbation Type: {args.perturb_type}")
    plt.ylabel("Average Score")
    plt.xticks(rotation=45)
    plt.ylim(min(avg_scores.values) - 1, max(avg_scores.values) + 1)  # Adjust y-axis for better visibility

    # Add exact values below the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height - 0.5,  # Position slightly below the top of the bar
            f"{height:.2f}", 
            ha="center", 
            va="top", 
            fontsize=10, 
            color="black"
        )

    plt.tight_layout()

    # Save the plot to a file
    plot_file_path = f"average_scores_all_{args.perturb_type}.png"
    plt.savefig(plot_file_path)
    print(f"Plot saved to {plot_file_path}")
    plt.close()

