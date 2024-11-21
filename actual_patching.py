import ir_datasets
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from TransformerLens.transformer_lens import HookedEncoder
from TransformerLens.transformer_lens.utils import get_device

from patching_helpers import (
    get_act_patch_block_every,
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_by_pos,
)

def plot_heatmap(data, title, xlabel, ylabel, xticklabels=None, yticklabels=None, figsize=(8, 6), cmap="coolwarm"):
    """
    Utility function to plot paper-style heatmaps using seaborn.
    """
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)
    sns.heatmap(
        data,
        annot=True,  # Annotate values in cells for debugging
        fmt=".2f",
        cmap=cmap,  # Diverging color map
        cbar=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        linewidths=0.5,
        linecolor="gray",
        vmin=-1,  # Normalize scores to range [-1, 1]
        vmax=1,
    )
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.show()

def load_tokenizer_and_models(hf_model_name, device):
    """
    Load the tokenizer and model (TransformerLens-compatible) for the specified Hugging Face model.
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModel.from_pretrained(hf_model_name)
    hf_model.to(device)
    tl_model = HookedEncoder.from_pretrained(hf_model_name, device=device, hf_model=hf_model)
    return tokenizer, tl_model

def load_dataset(path):
    """
    Load the MS MARCO dataset using ir_datasets.
    """
    dataset = ir_datasets.load(path)
    print(f"Loaded dataset: {path}")
    return dataset

def get_relevant_docs(dataset, query_ids):
    """
    Retrieve relevant documents for a given set of query IDs based on qrels.
    """
    relevant_doc_ids = set()
    for qrel in dataset.qrels_iter():
        if qrel.query_id in query_ids:
            relevant_doc_ids.add(qrel.doc_id)
    
    relevant_docs = []
    for doc_id in relevant_doc_ids:
        doc = dataset.docs_store().get(doc_id)
        if doc:
            relevant_docs.append(doc)
    
    return relevant_docs

def classify_tokens(tokenizer, query, baseline_doc, perturbed_doc):
    """
    Classify each token in the document into one of the token types.
    """
    query_tokens = tokenizer.tokenize(query)
    query_ids = tokenizer.convert_tokens_to_ids(query_tokens)

    baseline_tokens = tokenizer.tokenize(baseline_doc)
    baseline_ids = tokenizer.convert_tokens_to_ids(baseline_tokens)

    perturbed_tokens = tokenizer.tokenize(perturbed_doc)
    perturbed_ids = tokenizer.convert_tokens_to_ids(perturbed_tokens)

    token_types = []
    for idx, token_id in enumerate(perturbed_ids):
        if idx == 0:
            token_types.append("tokCLS")  # CLS token
        elif idx == len(perturbed_ids) - 1:
            token_types.append("tokSEP")  # SEP token
        elif token_id in query_ids and token_id not in baseline_ids:
            token_types.append("tokinj")  # Injected query term
        elif token_id in query_ids:
            token_types.append("tokqterm+")  # Query term present in the baseline
        elif token_id not in query_ids:
            token_types.append("tokother")  # Non-query term
        else:
            token_types.append("tokqterm-")  # Unselected query term

    return token_types

def ranking_metric(patched_doc_embedding, og_score, p_score):
    """
    Computes the ranking metric normalized between -1 and 1.
    """
    patched_score = torch.matmul(q_embedding, patched_doc_embedding.t())
    if p_score - og_score == 0:
        return 0  # Avoid division by zero
    return (patched_score - og_score) / (p_score - og_score)

def main():
    torch.set_grad_enabled(False)
    device = get_device()

    # Load dataset
    dataset = load_dataset("msmarco-document/dev")

    num_queries_to_load = 5
    queries = list(dataset.queries_iter())[:num_queries_to_load]
    query_ids = [query.query_id for query in queries]

    relevant_docs = get_relevant_docs(dataset, query_ids)
    print(f"Number of relevant documents retrieved: {len(relevant_docs)}")

    for query in queries:
        print(f"Processing Query: {query.text}")

        query_doc_ids = [
            qrel.doc_id for qrel in dataset.qrels_iter() if qrel.query_id == query.query_id
        ]
        query_docs = [
            doc for doc in relevant_docs if doc.doc_id in query_doc_ids
        ]
        if not query_docs:
            print("No relevant documents found for this query.")
            continue

        original_doc = query_docs[0].body
        perturbed_doc = original_doc + " additional relevance information"

        print(f"Original Document: {original_doc[:200]}...")  # Truncate for readability

        pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
        tokenizer, tl_model = load_tokenizer_and_models(pre_trained_model_name, device)

        max_seq_length = 512
        tokenized_query = tokenizer(query.text, return_tensors="pt", max_length=max_seq_length, truncation=True)
        tokenized_baseline_doc = tokenizer(original_doc, return_tensors="pt", max_length=max_seq_length, truncation=True)
        tokenized_p_doc = tokenizer(perturbed_doc, return_tensors="pt", max_length=max_seq_length, truncation=True)

        # Ensure token lengths match
        b_len = tokenized_baseline_doc["input_ids"].size(1)
        p_len = tokenized_p_doc["input_ids"].size(1)

        if b_len != p_len:
            size_diff = abs(b_len - p_len)
            filler_token = tokenizer.encode("a", add_special_tokens=False)[0]
            if b_len < p_len:
                filler_tokens = torch.full((1, size_diff), filler_token, dtype=torch.long, device=device)
                filler_attn_mask = torch.ones((1, size_diff), dtype=torch.long, device=device)
                tokenized_baseline_doc["input_ids"] = torch.cat(
                    [tokenized_baseline_doc["input_ids"].to(device), filler_tokens], dim=1
                )
                tokenized_baseline_doc["attention_mask"] = torch.cat(
                    [tokenized_baseline_doc["attention_mask"].to(device), filler_attn_mask], dim=1
                )
            else:
                filler_tokens = torch.full((1, size_diff), filler_token, dtype=torch.long, device=device)
                filler_attn_mask = torch.ones((1, size_diff), dtype=torch.long, device=device)
                tokenized_p_doc["input_ids"] = torch.cat(
                    [tokenized_p_doc["input_ids"].to(device), filler_tokens], dim=1
                )
                tokenized_p_doc["attention_mask"] = torch.cat(
                    [tokenized_p_doc["attention_mask"].to(device), filler_attn_mask], dim=1
                )

        token_types = classify_tokens(tokenizer, query.text, original_doc, perturbed_doc)

        unique_token_types = ["tokCLS", "tokinj", "tokqterm+", "tokqterm-", "tokother", "tokSEP"]
        token_type_indices = {t: i for i, t in enumerate(unique_token_types)}

        # Compute query and baseline embeddings
        q_outputs = tl_model(
            tokenized_query["input_ids"].to(device),
            return_type="embeddings",
            one_zero_attention_mask=tokenized_query["attention_mask"].to(device),
        )
        global q_embedding  # Define q_embedding globally for ranking_metric
        q_embedding = q_outputs[:, 0, :].squeeze(0)

        baseline_outputs = tl_model(
            tokenized_baseline_doc["input_ids"].to(device),
            return_type="embeddings",
            one_zero_attention_mask=tokenized_baseline_doc["attention_mask"].to(device),
        )
        baseline_embedding = baseline_outputs[:, 0, :].squeeze(0)
        baseline_score = torch.matmul(q_embedding, baseline_embedding.t())

        perturbed_outputs, perturbed_cache = tl_model.run_with_cache(
            tokenized_p_doc["input_ids"].to(device),
            one_zero_attention_mask=tokenized_p_doc["attention_mask"].to(device),
            return_type="embeddings",
        )
        perturbed_embedding = perturbed_outputs[:, 0, :].squeeze(0)
        perturbed_score = torch.matmul(q_embedding, perturbed_embedding.t())

        act_patch_block_every = get_act_patch_block_every(
            tl_model,
            device,
            tokenized_baseline_doc,
            perturbed_cache,
            lambda patched_doc_embedding: ranking_metric(patched_doc_embedding, baseline_score, perturbed_score),
        )

        act_patch_attn_head_out_all_pos = get_act_patch_attn_head_out_all_pos(
            tl_model,
            device,
            tokenized_baseline_doc,
            perturbed_cache,
            lambda patched_doc_embedding: ranking_metric(patched_doc_embedding, baseline_score, perturbed_score),
        )

        grouped_data = torch.zeros((len(unique_token_types), act_patch_block_every.shape[1]), device=device)
        for pos, token_type_idx in enumerate([token_type_indices[token] for token in token_types]):
            grouped_data[token_type_idx, :] += act_patch_block_every[:, :, pos].mean(dim=0)

        for i, component in enumerate(["Residual Stream", "Attention Output", "MLP Output"]):
            data = grouped_data.cpu().numpy()
            plot_heatmap(
                data,
                title=f"Importance of {component} (Grouped by Token Type)",
                xlabel="Token Type",
                ylabel="Layer",
                xticklabels=unique_token_types,
                yticklabels=[f"Layer {l}" for l in range(data.shape[1])],
                figsize=(8, 6),
                cmap="coolwarm",
            )


if __name__ == "__main__":
    main()