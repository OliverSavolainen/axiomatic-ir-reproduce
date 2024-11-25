import json
import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pathlib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from TransformerLens.transformer_lens import HookedEncoder
from TransformerLens.transformer_lens.utils import get_device
import random
from collections import defaultdict  # For storing score changes

from patching_helpers import (
    get_act_patch_block_every,
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_by_pos,
)

def plot_heatmap(data, title, xlabel, ylabel, xticklabels=None, yticklabels=None, figsize=(8, 6), cmap="coolwarm_r", save_path=None):
    """
    Utility function to plot paper-style heatmaps using seaborn.
    """
    plt.figure(figsize=figsize)
    sns.set_theme(style="whitegrid")  # Set theme for a cleaner look
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
    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)  # Save the plot to the specified path
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

    plt.close()

def load_tokenizer_and_models(hf_model_name, device):
    """
    Load the tokenizer and model (TransformerLens-compatible) for the specified Hugging Face model.
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModel.from_pretrained(hf_model_name)
    hf_model.to(device)
    tl_model = HookedEncoder.from_pretrained(hf_model_name, device=device, hf_model=hf_model)
    return tokenizer, tl_model

def load_dataset(dataset_name="msmarco", split="dev"):
    """
    Load the MS MARCO dataset using BEIR's GenericDataLoader.

    Args:
        dataset_name (str): Name of the dataset to load (default: "msmarco").
        split (str): Dataset split to load (default: "dev").
    
    Returns:
        corpus (dict): Dictionary containing document IDs and content.
        queries (dict): Dictionary containing query IDs and content.
        qrels (dict): Query relevance judgments.
    """
    # Define dataset download URL and local storage path
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = pathlib.Path("datasets") / dataset_name

    # Download and unzip dataset if not already present
    try:
        data_path = util.download_and_unzip(url, out_dir)
        print(f"Dataset downloaded and unzipped at: {data_path}")
    except Exception as e:
        print(f"Error downloading or unzipping dataset: {e}")
        return None, None, None

    # Load the dataset using BEIR's GenericDataLoader
    try:
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        print(f"Number of documents: {len(corpus)}")
        print(f"Number of queries: {len(queries)}")
        print(f"Number of qrels: {len(qrels)}")
        return corpus, queries, qrels
    except Exception as e:
        print(f"Error loading dataset with BEIR: {e}")
        return None, None, None

def prepare_diagnostic_dataset(corpus, queries, qrels, device):
    """
    Prepare the diagnostic dataset by ranking queries based on score changes.
    """
    diagnostic_dataset = []

    pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer, tl_model = load_tokenizer_and_models(pre_trained_model_name, device)

    for query_id, query_text in queries.items():
        relevant_doc_ids = qrels.get(query_id, {})
        query_docs = [corpus[doc_id] for doc_id in relevant_doc_ids.keys() if doc_id in corpus]

        if not query_docs:
            print(f"No relevant documents found for query {query_id}.")
            continue

        total_score_change = 0
        num_docs = 0
        scored_docs = []
        score_changes = []
        for doc in query_docs:
            original_doc = doc["text"]
            max_seq_length = 512
            tokenized_query = tokenizer(query_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
            tokenized_doc = tokenizer(original_doc, return_tensors="pt", max_length=max_seq_length, truncation=True)

            # Compute embeddings and scores
            q_outputs = tl_model(
                tokenized_query["input_ids"].to(device),
                return_type="embeddings",
                one_zero_attention_mask=tokenized_query["attention_mask"].to(device),
            )
            q_embedding = q_outputs[:, 0, :].squeeze(0)

            doc_outputs = tl_model(
                tokenized_doc["input_ids"].to(device),
                return_type="embeddings",
                one_zero_attention_mask=tokenized_doc["attention_mask"].to(device),
            )
            doc_embedding = doc_outputs[:, 0, :].squeeze(0)
            relevance_score = torch.matmul(q_embedding, doc_embedding.t()).item()
            scored_docs.append((doc, relevance_score))

        # Sort documents by relevance score
        scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:100]

        for doc, baseline_score in scored_docs:
            original_doc = doc["text"]

            # Randomly select a query term for perturbation
            random_query_term = random.choice(query_text.split())
            perturbed_doc = original_doc + f" {random_query_term}"

            # Tokenize perturbed document
            tokenized_p_doc = tokenizer(perturbed_doc, return_tensors="pt", max_length=max_seq_length, truncation=True)

            # Compute perturbed score
            perturbed_outputs = tl_model(
                tokenized_p_doc["input_ids"].to(device),
                return_type="embeddings",
                one_zero_attention_mask=tokenized_p_doc["attention_mask"].to(device),
            )
            perturbed_embedding = perturbed_outputs[:, 0, :].squeeze(0)
            perturbed_score = torch.matmul(q_embedding, perturbed_embedding.t()).item()

            # Compute score change
            score_change = perturbed_score - baseline_score
            total_score_change += score_change
            score_changes.append(score_change)
            num_docs += 1

        if num_docs > 0:
            max_change = max(score_changes)
            min_change = min(score_changes)
            normalized_changes = [(sc - min_change) / (max_change - min_change) if max_change != min_change else 0 for sc in score_changes]

            avg_normalized_score_change = sum(normalized_changes) / len(normalized_changes)
            diagnostic_dataset.append({
                "query_id": query_id,
                "query_text": query_text,
                "avg_score_change": avg_normalized_score_change,
            })

    # Select top 100 queries with the highest average score change
    diagnostic_dataset = sorted(diagnostic_dataset, key=lambda x: x["avg_score_change"], reverse=True)[:100]
    with open("diagnostic_dataset.json", "w") as f:
        json.dump(diagnostic_dataset, f, indent=4)

    print("Diagnostic dataset prepared and saved as 'diagnostic_dataset.json'.")
    return diagnostic_dataset

def perform_activation_patching(diagnostic_dataset, corpus, queries, qrels, device):
    """
    Perform activation patching and generate averaged heatmaps for the diagnostic dataset.
    """
    pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer, tl_model = load_tokenizer_and_models(pre_trained_model_name, device)
    max_seq_length = 512

    os.makedirs("heatmaps", exist_ok=True)

    # Initialize aggregated data structures
    n_layers = tl_model.cfg.n_layers
    n_heads = tl_model.cfg.n_heads
    n_token_types = 6  # Number of unique token types

    # Initialize aggregated data structures
    aggregated_data = {
        "Block Every": torch.zeros((n_token_types, n_layers), device=device),
        "Attn Head All Pos": torch.zeros((n_layers, n_heads), device=device),
        "Attn Head By Pos": torch.zeros((n_token_types, n_layers), device=device),  # Adjust dynamically
    }

    query_count = 0

    for data in diagnostic_dataset:
        query_id = data["query_id"]
        query_text = data["query_text"]

        relevant_doc_ids = qrels.get(query_id, {})
        query_docs = [corpus[doc_id] for doc_id in relevant_doc_ids.keys() if doc_id in corpus]

        if not query_docs:
            print(f"No relevant documents found for query {query_id}.")
            continue

        doc = query_docs[0]
        original_doc = doc["text"]

        random_query_term = random.choice(query_text.split())
        injected_doc = original_doc + f" {random_query_term}"

        tokenized_baseline_doc = tokenizer(original_doc, return_tensors="pt", max_length=max_seq_length, truncation=True)
        tokenized_injected_doc = tokenizer(injected_doc, return_tensors="pt", max_length=max_seq_length, truncation=True)

        # Ensure token lengths match
        b_len = tokenized_baseline_doc["input_ids"].size(1)
        p_len = tokenized_injected_doc["input_ids"].size(1)

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
                tokenized_injected_doc["input_ids"] = torch.cat(
                    [tokenized_injected_doc["input_ids"].to(device), filler_tokens], dim=1
                )
                tokenized_injected_doc["attention_mask"] = torch.cat(
                    [tokenized_injected_doc["attention_mask"].to(device), filler_attn_mask], dim=1
                )

        tokenized_query = tokenizer(query_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
        q_outputs = tl_model(
            tokenized_query["input_ids"].to(device),
            return_type="embeddings",
            one_zero_attention_mask=tokenized_query["attention_mask"].to(device),
        )
        q_embedding = q_outputs[:, 0, :].squeeze(0)

        # Baseline run
        baseline_outputs, baseline_cache = tl_model.run_with_cache(
            tokenized_baseline_doc["input_ids"].to(device),
            one_zero_attention_mask=tokenized_baseline_doc["attention_mask"].to(device),
            return_type="embeddings",
        )
        baseline_score = torch.matmul(q_embedding, baseline_outputs[:, 0, :].t()).item()

        # Perturbed run
        perturbed_outputs, injected_cache = tl_model.run_with_cache(
            tokenized_injected_doc["input_ids"].to(device),
            one_zero_attention_mask=tokenized_injected_doc["attention_mask"].to(device),
            return_type="embeddings",
        )
        perturbed_score = torch.matmul(q_embedding, perturbed_outputs[:, 0, :].t()).item()

        # Patching methods
        act_patch_block_every = get_act_patch_block_every(
            tl_model,
            device,
            tokenized_baseline_doc,
            injected_cache,
            lambda patched_doc_embedding: ranking_metric(
                patched_doc_embedding,
                baseline_score,
                perturbed_score,
            ),
        )

        act_patch_attn_head_out_all_pos = get_act_patch_attn_head_out_all_pos(
            tl_model,
            device,
            tokenized_baseline_doc,
            injected_cache,
            lambda patched_doc_embedding: ranking_metric(
                patched_doc_embedding,
                baseline_score,
                perturbed_score,
            ),
        )

        layer_head_list = [(layer, head) for layer in range(tl_model.cfg.n_layers) for head in range(tl_model.cfg.n_heads)]

        act_patch_attn_head_by_pos = get_act_patch_attn_head_by_pos(
            tl_model,
            device,
            tokenized_baseline_doc,
            injected_cache,
            lambda patched_doc_embedding: ranking_metric(
                patched_doc_embedding,
                baseline_score,
                perturbed_score,
            ),
            layer_head_list,
        )

        # Classify tokens
        token_types = classify_tokens(tokenizer, query_text, original_doc, injected_doc)
        unique_token_types = ["tokCLS", "tokinj", "tokqterm+", "tokqterm-", "tokother", "tokSEP"]
        token_type_indices = {t: i for i, t in enumerate(unique_token_types)}

        print(f"Token types: {token_types}")  # Print classified token types
        print(f"Unique token types: {unique_token_types}")

        for patch_type, patch_data in [
            ("Block Every", act_patch_block_every),
            ("Attn Head All Pos", act_patch_attn_head_out_all_pos),
            ("Attn Head By Pos", act_patch_attn_head_by_pos),
        ]:
            print(f"\nProcessing patch type: {patch_type}")
            print(f"Patch data shape: {patch_data.shape}")

            if patch_type == "Block Every":
                grouped_data = torch.zeros((n_token_types, n_layers), device=device)
                for component_idx in range(3):  # 3 components: resid_pre, attn_out, mlp_out
                    for pos, token_type_idx in enumerate([token_type_indices[token] for token in token_types]):
                        grouped_data[token_type_idx, :] += patch_data[component_idx, :, pos]
                aggregated_data[patch_type] += grouped_data

            elif patch_type == "Attn Head All Pos":
                grouped_data = patch_data.mean(dim=1)  # Average over heads
                aggregated_data[patch_type] += grouped_data.unsqueeze(1)  # Reshape to (n_layers, 1)

            elif patch_type == "Attn Head By Pos":
                grouped_data = torch.zeros((n_token_types, n_layers), device=device)
                for component_idx in range(2):  # 2 components: z and pattern
                    for i, (layer, head) in enumerate(layer_head_list):
                        for pos, token_type_idx in enumerate([token_type_indices[token] for token in token_types]):
                            grouped_data[token_type_idx, layer] += patch_data[component_idx, i, pos]
                aggregated_data[patch_type] += grouped_data

            print(f"Grouped data for {patch_type} after processing:\n{grouped_data}")
            print(f"Aggregated data for {patch_type} after addition:\n{aggregated_data[patch_type]}")

        query_count += 1
        print(f"Queries processed so far: {query_count}")

        # Normalize aggregated data
        for key in aggregated_data:
            print(f"Normalizing aggregated data for {key}. Current shape: {aggregated_data[key].shape}")
            aggregated_data[key] /= query_count
            print(f"Normalized aggregated data for {key}:\n{aggregated_data[key]}")

        # Generate heatmaps for aggregated data
        for patch_type, data in aggregated_data.items():
            for i, component in enumerate(["Residual Stream", "Attention Output", "MLP Output"]):
                plot_heatmap(
                    data.cpu().numpy(),
                    title=f"Aggregated {patch_type} - Importance of {component}",
                    xlabel="Token Type",
                    ylabel="Layer",
                    xticklabels=unique_token_types,
                    yticklabels=[f"L {l}" for l in range(data.shape[1])],
                    save_path=f"heatmaps/{patch_type.lower().replace(' ', '_')}_{component.lower().replace(' ', '_')}.png",
                )

        print("Averaged activation patching and heatmap generation completed.")


def classify_tokens(tokenizer, query, baseline_doc, perturbed_doc):
    """
    Classify each token in the perturbed document into token categories.
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
        elif token_id in query_ids and token_id in baseline_ids:
            token_types.append("tokqterm+")  # Query term in both query and baseline
        elif token_id in query_ids and token_id in baseline_ids and token_id not in perturbed_ids:
            token_types.append("tokqterm-")  # Non-selected query term
        elif token_id not in query_ids:
            token_types.append("tokother")  # Non-query term
        else:
            token_types.append("tokother")  # Fallback to non-query term

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
    """
    1. Use the MS MARCO dataset.
    2. Retrieve the top 100 documents for each query using TAS-B (or equivalent scoring logic).
    3. Randomly select a term from each query to create perturbed documents.
    4. Compute the average score change for all documents per query.
    5. Select the top 100 queries with the highest average score change.
    """

    torch.set_grad_enabled(False)
    device = get_device()

    # Load dataset
    print("Loading dataset...")
    corpus, queries, qrels = load_dataset()
    if corpus is None or queries is None or qrels is None:
        print("Failed to load dataset. Exiting.")
        return

    print("Dataset loaded successfully.")
    print("Preparing diagnostic dataset...")

    # Prepare diagnostic dataset
    diagnostic_dataset = prepare_diagnostic_dataset(corpus, queries, qrels, device)
    if not diagnostic_dataset:
        print("Failed to prepare diagnostic dataset. Exiting.")
        return

    print("Performing activation patching...")

    # Perform activation patching
    perform_activation_patching(diagnostic_dataset, corpus, queries, qrels, device)


if __name__ == "__main__":
    main()