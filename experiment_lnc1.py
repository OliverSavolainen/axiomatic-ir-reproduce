import torch
import numpy as np
import pandas as pd

import os
from tqdm import tqdm
import random
import argparse

import TransformerLens.transformer_lens.utils as utils

from helpers import (
    load_json_file,
    load_tokenizer_and_models,
    preprocess_queries,
    preprocess_corpus,
)

from patching_helpers import (
    get_act_patch_block_every,
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_by_pos
)

def ranking_metric(patched_doc_embedding, query_embedding, og_score=0, p_score=1):
    patched_score = torch.matmul(query_embedding, patched_doc_embedding.t())
    return (patched_score - og_score) / (p_score - og_score)

def run_experiment(experiment_type, perturb_type):
    """
        - experiment_type: what type of experiment to run (options below)
            - block : patches the residual stream (before each layer), attention block, and MLP outputs across individual tokens
            - head_all : patches all attention heads individually across all tokens
            - head_pos : patches all attention heads individually across individual tokens
            - labels : generates the tokenized documents
        - perturb_type :
            - prepend : additional term is injected at the beginning of the document
            - append : additional term is injected at the end of the document
    """
    torch.set_grad_enabled(False)
    device = utils.get_device()

    use_reduced_dataset = False
    n_queries = 1
    n_docs = 100

    # Load queries and docs from files
    fbase_path = "data"

    # Load files
    tfc1_add_queries = pd.read_csv(os.path.join(fbase_path, "lnc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
    tfc1_add_baseline_corpus = load_json_file(os.path.join(fbase_path, "tfc1_add_baseline_final_dd_corpus.json"))["corpus"]
    print(os.path.join(fbase_path, "tfc1_add_baseline_final_dd_corpus.json"))
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

    if not os.path.exists("results_lnc2"):
        os.mkdir("results_lnc2")
    if not os.path.exists(os.path.join("results_lnc2", perturb_type)):
        os.mkdir(f"results_lnc2/{perturb_type}")
    if not os.path.exists("results_lnc2_head_decomp"):
        os.mkdir("results_lnc2_head_decomp")
    if not os.path.exists(os.path.join("results_lnc2_head_decomp", perturb_type)):
        os.mkdir(f"results_lnc2_head_decomp/{perturb_type}")
    if not os.path.exists("results_lnc2_attn_pattern"):
        os.mkdir("results_lnc2_attn_pattern")
    if not os.path.exists(os.path.join("results_lnc2_attn_pattern", perturb_type)):
        os.mkdir(f"results_lnc2_attn_pattern/{perturb_type}")

    filler_token_id = tokenizer.convert_tokens_to_ids("guantanamo")
    pad_token_id = tokenizer.pad_token_id
    # Loop through each query and run activation patching
    for i, qid in enumerate(tqdm(target_qids)):
        print("QID at:", i)
        # Get query embedding
        q_tokenized = list(filter(lambda item: item["_id"] == qid, queries_dataloader))[0]
        query_text = tokenizer.decode(q_tokenized["input_ids"][0], skip_special_tokens=True)
        q_tokenized["input_ids"] = q_tokenized["input_ids"].to(device)
        q_tokenized["attention_mask"] = q_tokenized["attention_mask"].to(device)

        print(f"Query Text: {query_text}")
        q_outputs = tl_model(
            q_tokenized["input_ids"],
            return_type="embeddings",
            one_zero_attention_mask=q_tokenized["attention_mask"],
        )
        q_embedding = q_outputs[:,0,:].squeeze(0) # leave on device
        
        # Get and preprocess documents
        target_docs = tfc1_add_dd_corpus[str(qid)]
        if use_reduced_dataset:
            target_doc_ids = random.sample(target_docs.keys(), n_docs)
            target_docs = {doc_id: target_docs[doc_id] for doc_id in target_doc_ids}
        corpus_dataloader = preprocess_corpus(target_docs, tokenizer)
        
        # batch size = 1 doc
        for j, batch in enumerate(corpus_dataloader):
            print("DOC AT:", j)
            # Get baseline doc
            try:
                doc_id = batch["_id"][0]
                baseline_doc = tfc1_add_baseline_corpus[str(qid)][doc_id]["text"]
                baseline_tokens = tokenizer(baseline_doc, truncation=True, return_tensors="pt")
            

                b_len = torch.sum(baseline_tokens["attention_mask"])
                cls_tok = baseline_tokens["input_ids"][0][0]
                sep_tok = baseline_tokens["input_ids"][0][-1]

                # Let's say we've already determined b_len to be the number of filler tokens we want to insert
                # For example, if we want to append `b_len` tokens at the end, we do:

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

                # Move baseline tokens to device
                baseline_tokens["input_ids"] = baseline_tokens["input_ids"].to(device)
                baseline_tokens["attention_mask"] = baseline_tokens["attention_mask"].to(device)

                if experiment_type != "labels":
                    # Now run the model as before. The embeddings for filler tokens will be zeroed out.
                    baseline_embeddings = tl_model(
                        baseline_tokens["input_ids"],
                        one_zero_attention_mask=baseline_tokens["attention_mask"],
                        return_type="embeddings",
                    )

                    # Now create the perturbed version by replacing PAD with the chosen filler token (e.g., 'a')
                    # We'll decode and re-encode the baseline input_ids to replace PAD tokens with 'a'
                    baseline_ids = baseline_tokens["input_ids"].clone()
                    # Convert the token 'a' to ID

                    # Replace PAD tokens with 'a' tokens in the perturbed run
                    perturbed_ids = baseline_ids.masked_fill(baseline_ids == pad_token_id, filler_token_id)

                    perturbed_tokens = {
                        "input_ids": perturbed_ids,
                        # Use a fully enabled attention mask (all ones) so all tokens, including 'a', are considered
                        "attention_mask": torch.ones_like(baseline_tokens["attention_mask"]).to(device)
                    }

                    # Run perturbed
                    perturbed_embeddings, perturbed_cache = tl_model.run_with_cache(
                        perturbed_tokens["input_ids"],
                        one_zero_attention_mask=perturbed_tokens["attention_mask"],
                        return_type="embeddings",
                    )

                    baseline_embedding = baseline_embeddings[:,0,:].squeeze(0)
                    perturbed_embedding = perturbed_embeddings[:,0,:].squeeze(0)

                    # Compute scores
                    baseline_score = torch.matmul(q_embedding, baseline_embedding.t())
                    perturbed_score = torch.matmul(q_embedding, perturbed_embedding.t())


                # Linear function of score diff, calibrated so that it equals 0 when performance is 
                # same as on clean input, and 1 when performance is same as on corrupted input.
                
                # Patch after each layer (residual stream, attention, MLPs)
                if experiment_type == "block":
                    act_patch_block_every = get_act_patch_block_every(
                        tl_model,
                        device,
                        q_embedding,
                        baseline_score,
                        perturbed_score,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric
                    )
                    detached_block_results = act_patch_block_every.detach().cpu().numpy()
                    
                    if j == 10:
                        baseline_text = tokenizer.decode(baseline_tokens["input_ids"][0], skip_special_tokens=True)
                        print(f"Baseline Document Text: {baseline_doc[:500]}")
                        print(f"Baseline Document Text after: {baseline_text}")
                        print(baseline_score)
                        print(perturbed_score)
                        print(detached_block_results)

                    np.save("results_lnc2/{}/{}_{}_block.npy".format(perturb_type, qid, doc_id), detached_block_results)
                    

                # Patch attention heads
                elif experiment_type == "head_all":
                    act_patch_attn_head_out_all_pos = get_act_patch_attn_head_out_all_pos(
                        tl_model,
                        device,
                        q_embedding,
                        baseline_score,
                        perturbed_score,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric
                    )
                    detached_head_results = act_patch_attn_head_out_all_pos.detach().cpu().numpy()
                    if j == 10:
                        baseline_text = tokenizer.decode(baseline_tokens["input_ids"][0], skip_special_tokens=True)
                        print(f"Baseline Document Text: {baseline_doc[:500]}")
                        print(f"Baseline Document Text after: {baseline_text}")
                        print(baseline_score)
                        print(perturbed_score)
                        print(detached_head_results)

                    np.save("results_lnc2/{}/{}_{}_head.npy".format(perturb_type, qid, doc_id), detached_head_results)

                # Patch heads by position
                elif experiment_type == "head_pos":
                    layer_head_list = [(0,9), (1,6), (0,1), (2,1)]
                    act_patch_attn_head_out_by_pos = get_act_patch_attn_head_by_pos(
                        tl_model,
                        device,
                        q_embedding,
                        baseline_score,
                        perturbed_score,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric,
                        layer_head_list
                    )
                    detached_head_pos_results = act_patch_attn_head_out_by_pos.detach().cpu().numpy()

                    if j == 9:
                        baseline_text = tokenizer.decode(baseline_tokens["input_ids"][0], skip_special_tokens=True)
                        print(f"Baseline Document Text: {baseline_doc[:500]}")
                        print(f"Baseline Document Text after: {baseline_text}")
                        print(baseline_score)
                        print(perturbed_score)
                        print(detached_head_pos_results)
                    np.save("results_lnc2_head_decomp/{}/{}_{}_head_by_pos.npy".format(perturb_type, qid, doc_id), detached_head_pos_results)

                elif experiment_type == "head_attn":
                    # Get attention patterns for head
                    attn_heads = [(0,9), (1,6), (0,1), (2,1)]
                    for layer, head in attn_heads:
                        attn_pattern = perturbed_cache["pattern", layer][:,head].mean(0).detach().cpu().numpy()
                        if j == 9:
                            baseline_text = tokenizer.decode(baseline_tokens["input_ids"][0], skip_special_tokens=True)
                            print(f"Baseline Document Text: {baseline_doc[:500]}")
                            print(f"Baseline Document Text after: {baseline_text}")
                            print(baseline_score)
                            print(perturbed_score)
                            print(detached_head_pos_results)
                        np.save("results_lnc2_attn_pattern/{}/{}_{}_{}_{}_attn_pattern.npy".format(perturb_type, qid, doc_id, layer, head), attn_pattern)

                elif experiment_type == "labels":
                    decoded_tokens = [tokenizer.decode(tok) for tok in baseline_tokens["input_ids"][0]]
                    labels = ["{} {}".format(tok,i) for i, tok in enumerate(decoded_tokens)]
                    with open("results_lnc2/{}/{}_{}_labels.txt".format(perturb_type, qid, doc_id), "w") as f:
                        for item in labels:
                            f.write(str(item) + '\n')
            
            except Exception as e:
                print(baseline_doc)
                print(baseline_tokens)
                token_shape = baseline_tokens["input_ids"].shape
                print(f"Shape of input_ids: {token_shape}")
                print("ERROR: {} for query {} and document {}".format(e, qid, doc_id))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run activation patching with the specific patching experiment and perturbation types.")
    parser.add_argument("experiment_type", default="block",type=str, help="What will be patched (e.g., block).")
    parser.add_argument("perturb_type",default="append", type=str, help="The perturbation to apply (e.g., append).")
    
    args = parser.parse_args()

    valid_exp_types = {"block", "head_all", "head_pos", "head_attn", "labels"}
    valid_perturb_types = {"append", "prepend"}

    assert args.experiment_type in valid_exp_types, f"Invalid argument: experiment_type. Must be one of {valid_exp_types}."
    assert args.perturb_type in valid_perturb_types, f"Invalid argument: perturb_type. Must be one of {valid_perturb_types}."
    
    _ = run_experiment(args.experiment_type, args.perturb_type)