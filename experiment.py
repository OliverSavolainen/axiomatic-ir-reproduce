import torch
import numpy as np
import pandas as pd

import os
from tqdm import tqdm
import random
import glob
import argparse

from functools import partial
import TransformerLens.transformer_lens.utils as utils
from TransformerLens.transformer_lens import patching
from jaxtyping import Float

from helpers import (
    load_json_file,
    load_tokenizer_and_models,
    preprocess_queries,
    preprocess_corpus,
    encode_hf,
    encode_tl,
    compute_ranking_scores
)

from patching_helpers import (
    get_act_patch_block_every,
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_by_pos
)

def ranking_metric(patched_doc_embedding, query_embedding,og_score=0, p_score=1):
    patched_score = torch.matmul(query_embedding, patched_doc_embedding.t())
    return (patched_score - og_score) / (p_score - og_score)


def run_experiment(experiment_type, perturb_type):
    """
        - experiment_type: what type of experiment to run (options below)
            - block : patches the residual stream (before each layer), attention block, and MLP outputs across individual tokens
            - head_all : patches all attention heads individually across all tokens
            - head_pos : patches all attention heads individualls across individual tokens
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
    tfc1_precomputed_scores = pd.read_csv(os.path.join(fbase_path, "tfc1_add_{}_target_qids_scores.csv".format(perturb_type)))
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
    for name, param in tl_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # Preprocess queries
    queries_dataloader = preprocess_queries(tfc1_add_queries, tokenizer)

    # Loop through each query and run activation patching
    for i, qid in enumerate(tqdm(target_qids)):
        if i > 1:
            break
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
        q_embedding = q_outputs[:,0,:].squeeze(0) #.detach().cpu().numpy() # leave on device
        print("q_emb")
        print(q_embedding)
        
        # Get and preprocess documents
        target_docs = tfc1_add_dd_corpus[str(qid)]
        if use_reduced_dataset:
            target_doc_ids = random.sample(target_docs.keys(), n_docs)
            target_docs = {doc_id: target_docs[doc_id] for doc_id in target_doc_ids}
        corpus_dataloader = preprocess_corpus(target_docs, tokenizer)
        tl_model.reset_hooks()

        # batch size = 1 doc
        for j, batch in enumerate(corpus_dataloader):
            if j > 10:
                break
            print("DOC AT:", j)
            try:
                # Get baseline doc
                doc_id = batch["_id"][0]
                baseline_doc = tfc1_add_baseline_corpus[str(qid)][doc_id]["text"]
                print(f"Baseline Document Text: {baseline_doc[:500]}")
                baseline_tokens = tokenizer(baseline_doc, truncation=True, return_tensors="pt")

                
                                
                batch["input_ids"] = batch["input_ids"].to(device)
                batch["attention_mask"] = batch["attention_mask"].to(device)
                # Run perturbed prompt with cache to store activations
                perturbed_embeddings, perturbed_cache = tl_model.run_with_cache(
                    batch["input_ids"],
                    one_zero_attention_mask=batch["attention_mask"],
                    return_type="embeddings",
                )
                tl_model.reset_hooks()
                # Check lengths of pertubred and baseline tokens and adjust if needed
                p_len = torch.sum(batch["attention_mask"])
                b_len = torch.sum(baseline_tokens["attention_mask"])

                adj_n = p_len - b_len
                cls_tok = baseline_tokens["input_ids"][0][0]
                sep_tok = baseline_tokens["input_ids"][0][-1]

                # Hacky thing b/c of the way the diagnostic dataset was created (it was originally created just for prepend)
                # So will always need to adjust the tokens for append
                if perturb_type == "append":
                    filler_tokens = torch.full((adj_n+1,), baseline_tokens["input_ids"][0][1]) # skip CLS token
                    filler_attn_mask = torch.full((adj_n+1,), baseline_tokens["attention_mask"][0][1]) 
                    adj_doc = torch.cat((baseline_tokens["input_ids"][0][2:-1], filler_tokens))
                    baseline_tokens["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
                    baseline_tokens["attention_mask"] = torch.cat((baseline_tokens["attention_mask"][0][1:], filler_attn_mask), dim=0).view(1,-1)
                elif perturb_type == "prepend":
                    # But for prepend, we only need to adjust if the lengths are different
                    if p_len != b_len:
                        filler_tokens = torch.full((adj_n,), baseline_tokens["input_ids"][0][1]) # skip CLS token
                        filler_attn_mask = torch.full((adj_n,), baseline_tokens["attention_mask"][0][1]) 
                        adj_doc = torch.cat((baseline_tokens["input_ids"][0][1:-1],filler_tokens))
                        baseline_tokens["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
                        baseline_tokens["attention_mask"] = torch.cat((baseline_tokens["attention_mask"][0],filler_attn_mask), dim=0).view(1,-1)
                
                baseline_tokens["input_ids"] = baseline_tokens["input_ids"].to(device)
                baseline_tokens["attention_mask"] = baseline_tokens["attention_mask"].to(device)

                # Get baseline doc embedding
                baseline_outputs = tl_model(
                    baseline_tokens["input_ids"],
                    return_type="embeddings",
                    one_zero_attention_mask=baseline_tokens["attention_mask"],
                )
                baseline_embedding = baseline_outputs[:,0,:].squeeze(0) #.detach().cpu().numpy()
                tl_model.reset_hooks()
                # Get scores for baseline and perturbed documents
                baseline_score = torch.matmul(q_embedding, baseline_embedding.t())
                print("baseline score")
                print(baseline_embedding)

                print("Model Device:", next(tl_model.parameters()).device)

                perturbed_score = torch.tensor(tfc1_precomputed_scores[(tfc1_precomputed_scores["qid"] == qid) & (tfc1_precomputed_scores["doc_id"] == int(doc_id))]["p_score"].item())


                '''
                Linear function of score diff, calibrated so that it equals 0 when performance is 
                same as on clean input, and 1 when performance is same as on corrupted input.
                '''


                # Setup directories to save results
                if not os.path.exists("results2"):
                    os.mkdir("results2")
                if not os.path.exists(os.path.join("results2", perturb_type)):
                    os.mkdir(f"results2/{perturb_type}")
                if not os.path.exists("results_head_decomp"):
                    os.mkdir("results_head_decomp")
                if not os.path.exists(os.path.join("results_head_decomp", perturb_type)):
                    os.mkdir(f"results_head_decomp/{perturb_type}")
                if not os.path.exists("results_attn_pattern"):
                    os.mkdir("results_attn_pattern")

                print("Model Device:", next(tl_model.parameters()).device)

                # Check inputs
                print("Query Input IDs Device:", q_tokenized["input_ids"].device)
                print("Query Attention Mask Device:", q_tokenized["attention_mask"].device)

                print("Baseline Input IDs Device:", baseline_tokens["input_ids"].device)
                print("Baseline Attention Mask Device:", baseline_tokens["attention_mask"].device)

                print("Perturbed Input IDs Device:", batch["input_ids"].device)
                print("Perturbed Attention Mask Device:", batch["attention_mask"].device)

                # Check intermediate outputs
                print("Query Embedding Device:", q_embedding.device)
                print("Baseline Embedding Device:", baseline_embedding.device)
                print("Perturbed Embedding Device:", perturbed_embeddings[:, 0, :].squeeze(0).device)

                print("Baseline Outputs Device:", baseline_outputs.device)
                print("Perturbed Outputs Device:", perturbed_outputs.device)

                # Check caches
                if "resid_post" in perturbed_cache:
                    print("Perturbed Cache Sample Device:", perturbed_cache["resid_post", 0].device)
                else:
                    print("Perturbed Cache 'resid_post' not found.")

                # Check adjustments and padding
                if "input_ids" in baseline_tokens and "attention_mask" in baseline_tokens:
                    print("Adjusted Baseline Input IDs Device:", baseline_tokens["input_ids"].device)
                    print("Adjusted Baseline Attention Mask Device:", baseline_tokens["attention_mask"].device)

                if p_len != b_len:
                    filler_tokens = torch.full((adj_n,), baseline_tokens["input_ids"][0][1], device=device)
                    print("Filler Tokens Device:", filler_tokens.device)

                # Ranking computations
                print("Baseline Score Device:", baseline_score.device)
                print("Perturbed Score Device:", perturbed_score.device)
                if "patched_score" in locals():
                    print("Patched Score Device:", patched_score.device)
                else:
                    print("Patched Score not yet computed.")

                
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
                    print(act_patch_block_every)
                    detached_block_results = act_patch_block_every.detach().cpu().numpy()
                    np.save("results2/{}/{}_{}_block.npy".format(perturb_type, qid, doc_id), detached_block_results)
                    

                # Patch attention heads
                elif experiment_type == "head_all":
                    act_patch_attn_head_out_all_pos = get_act_patch_attn_head_out_all_pos(
                        tl_model,
                        device,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric
                    )
                    detached_head_results = act_patch_attn_head_out_all_pos.detach().cpu().numpy()
                    np.save("results/{}/{}_{}_head.npy".format(perturb_type, qid, doc_id), detached_head_results)

                # Patch heads by position
                elif experiment_type == "head_pos":
                    layer_head_list = [(0,9), (1,6), (2,3), (3,8)]
                    act_patch_attn_head_out_by_pos = get_act_patch_attn_head_by_pos(
                        tl_model,
                        device,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric,
                        layer_head_list
                    )
                    detached_head_pos_results = act_patch_attn_head_out_by_pos.detach().cpu().numpy()
                    np.save("results_head_decomp/{}/{}_{}_head_by_pos.npy".format(perturb_type, qid, doc_id), detached_head_pos_results)

                elif experiment_type == "head_attn":
                    # Get attention patterns for head
                    attn_heads = [(0,9), (1,6), (2,3), (3,8)]
                    for layer, head in attn_heads:
                        attn_pattern = perturbed_cache["pattern", layer][:,head].mean(0).detach().cpu().numpy()
                        np.save("results_attn_pattern/{}/{}_{}_{}_{}_attn_pattern.npy".format(perturb_type, qid, doc_id, layer, head), attn_pattern)

                
                elif experiment_type == "labels":
                    decoded_tokens = [tokenizer.decode(tok) for tok in batch["input_ids"][0]]
                    labels = ["{} {}".format(tok,i) for i, tok in enumerate(decoded_tokens)]
                    with open("results2/{}/{}_{}_labels.txt".format(perturb_type, qid, doc_id), "w") as f:
                        for item in labels:
                            f.write(str(item) + '\n')


            except Exception as e:
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