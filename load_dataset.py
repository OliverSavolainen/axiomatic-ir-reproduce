import json
import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pathlib
import torch
from transformers import AutoTokenizer, AutoModel
from TransformerLens.transformer_lens import HookedEncoder
from TransformerLens.transformer_lens.utils import get_device
import random
import csv
from helpers import compute_ranking_scores, set_seed
from tqdm import tqdm


# Ensure the directory exists
output_dir = "data_mmarco_spanish"
os.makedirs(output_dir, exist_ok=True)  

def load_dataset(dataset_name="mmarco", split="dev"):
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
    set_seed(42)
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
    data_path = pathlib.Path("datasets/mmarco/mmarco/spanish")
    print("SPANISH")
 
    try:
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        print(f"Number of documents: {len(corpus)}")
        print(f"Number of queries: {len(queries)}")
        print(f"Number of qrels: {len(qrels)}")
        return corpus, queries, qrels
    except Exception as e:
        print(f"Error loading dataset with BEIR: {e}")
        return None, None, None



def load_tokenizer_and_models(hf_model_name, device):
    """
    Load the tokenizer and model (TransformerLens-compatible) for the specified Hugging Face model.
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModel.from_pretrained(hf_model_name)
    hf_model.to(device)
    tl_model = HookedEncoder.from_pretrained(hf_model_name, device=device, hf_model=hf_model)
    return tokenizer, tl_model

def prepare_diagnostic_dataset(corpus, queries, qrels, device):
    """
    Prepare the diagnostic dataset with a structure similar to the given JSON file.
    """
    selected_query_terms = {}
    corpus_data = {}
    corpus_data_prepend = {}
    corpus_data_append = {}

    doc_ids = []  
    doc_embeddings_list = [] 


    pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer, tl_model = load_tokenizer_and_models(pre_trained_model_name, device)

    # Filter the corpus for documents associated with the queries
    relevant_doc_ids = set()
    for query_id in qrels:
        relevant_doc_ids.update(qrels[query_id].keys())

    # Ensure only relevant documents are considered
    filtered_corpus = {doc_id: corpus[doc_id] for doc_id in relevant_doc_ids if doc_id in corpus}

    max_seq_length = 512
    print("Computing embeddings for relevant documents...")
    # Initialize the progress bar
    with tqdm(total=len(filtered_corpus), desc="Processing Documents") as pbar:
        for doc_id, doc in filtered_corpus.items():
            original_text = doc["text"]
            tokenized_doc = tokenizer(original_text, return_tensors="pt", max_length=max_seq_length, truncation=True)

            # Compute document embedding
            doc_outputs = tl_model(
                tokenized_doc["input_ids"].to(device),
                return_type="embeddings",
                one_zero_attention_mask=tokenized_doc["attention_mask"].to(device),
            )
            doc_embedding = doc_outputs[:, 0, :].squeeze(0).cpu().numpy()  # Convert to NumPy array
            doc_embeddings_list.append(doc_embedding)
            doc_ids.append(doc_id)

            # Update the progress bar
            pbar.update(1)

    # Convert list of embeddings and IDs to arrays
    doc_embeddings = torch.stack([torch.tensor(embedding) for embedding in doc_embeddings_list])
    print(f"Total embeddings calculated: {len(doc_embeddings_list)}")

    with tqdm(total=len(queries), desc="Processing Queries", leave=True) as qbar:
        for query_id, query_text in queries.items():
            selected_query_term = random.choice(query_text.split())
            selected_query_terms[query_id] = selected_query_term
            
            prepend_total_score_change = 0
            append_total_score_change = 0

            num_docs = 0

            scored_docs = []

            prepend_score_changes = []
            append_score_changes = []

            max_seq_length = 512
            
            tokenized_query = tokenizer(query_text, return_tensors="pt", max_length=max_seq_length, truncation=True)

            # Compute query embedding
            q_outputs = tl_model(
                tokenized_query["input_ids"].to(device),
                return_type="embeddings",
                one_zero_attention_mask=tokenized_query["attention_mask"].to(device),
            )
            q_embedding = q_outputs[:, 0, :].squeeze(0)
            
            query_data = {}
            append_query_data = {}
            prepend_query_data = {}

            # Sort documents by relevance score
            top_scores, top_doc_ids = compute_ranking_scores(q_embedding.to(device), doc_embeddings.to(device), doc_ids)
            # Create scored_docs list with the top 100 results
            scored_docs = [(doc_id, corpus[doc_id], score.item()) for doc_id, score in zip(top_doc_ids, top_scores)][:100]

            for doc_id, doc, baseline_score in scored_docs:
                original_doc = doc["text"]
                term_count = original_doc.lower().count(selected_query_term.lower())
                
                # Randomly select a query term for perturbation
                prepend_perturbed_doc = f" {selected_query_term}" + original_doc
                append_perturbed_doc =  original_doc + f" {selected_query_term}"

                # Tokenize perturbed document
                prepend_tokenized_p_doc = tokenizer(prepend_perturbed_doc, return_tensors="pt", max_length=max_seq_length, truncation=True)
                append_tokenized_p_doc = tokenizer(append_perturbed_doc, return_tensors="pt", max_length=max_seq_length, truncation=True)

                # Compute perturbed score

                # Prepend
                prepend_perturbed_outputs = tl_model(
                    prepend_tokenized_p_doc["input_ids"].to(device),
                    return_type="embeddings",
                    one_zero_attention_mask=prepend_tokenized_p_doc["attention_mask"].to(device),
                )
                prepend_perturbed_embedding = prepend_perturbed_outputs[:, 0, :].squeeze(0)
                prepend_perturbed_score = torch.matmul(q_embedding, prepend_perturbed_embedding.t())

                # Append
                append_perturbed_outputs, _ = tl_model.run_with_cache(
                    append_tokenized_p_doc["input_ids"].to(device),
                    return_type="embeddings",
                    one_zero_attention_mask=append_tokenized_p_doc["attention_mask"].to(device),
                )
                append_perturbed_embedding = append_perturbed_outputs[:, 0, :].squeeze(0)
                append_perturbed_score = torch.matmul(q_embedding, append_perturbed_embedding.t())
                
                
                # Prepend - Compute score change
                prepend_score_change = (prepend_perturbed_score.item() - baseline_score) / abs(baseline_score) 
                prepend_total_score_change += prepend_score_change
                prepend_score_changes.append(prepend_score_change)
                

                # Append - Compute score change
                append_score_change = (append_perturbed_score.item() - baseline_score) / abs(baseline_score) 
                append_total_score_change += append_score_change
                append_score_changes.append(append_score_change)

                num_docs += 1
                
                query_data[doc_id] = {
                    "title": doc.get("title", ""),
                    "text": original_doc,
                    "query_term_orignal_ct": term_count
                }

                append_query_data[doc_id] = {
                    "title": doc.get("title", ""),
                    "text": append_perturbed_doc,
                    "query_term_orignal_ct": term_count
                }

                prepend_query_data[doc_id] = {
                    "title": doc.get("title", ""),
                    "text": prepend_perturbed_doc,
                    "query_term_orignal_ct": term_count
                }
            
            if num_docs > 0:
                prepend_avg_normalized_score_change = sum(prepend_score_changes) / len(prepend_score_changes)

                # Append
                append_avg_normalized_score_change = sum(append_score_changes) / len(append_score_changes)

                # Add to corpus data under the query ID

                # Prepend
                corpus_data_prepend[query_id] = {
                    "avg_score_change": append_avg_normalized_score_change,
                    "documents": prepend_query_data
                }
                print("corpus data prepend", len(corpus_data_prepend[query_id]["documents"]))

                # Append
                corpus_data_append[query_id] = {
                    "avg_score_change": append_avg_normalized_score_change,
                    "documents": append_query_data
                }
                print("corpus data append", len(corpus_data_append[query_id]["documents"]))
            corpus_data[query_id] = {
                "avg_score_change": append_avg_normalized_score_change,
                "documents": query_data
            }
            print("corpus data baseline", len(corpus_data[query_id]["documents"]))
            qbar.update(1)

    # Sort queries by average score change and keep the top 100
    top_queries = sorted(corpus_data.items(), key=lambda x: x[1]["avg_score_change"], reverse=True)[:100]
    prepend_top_queries = sorted(corpus_data_prepend.items(), key=lambda x: x[1]["avg_score_change"], reverse=True)[:100]
    append_top_queries = sorted(corpus_data_append.items(), key=lambda x: x[1]["avg_score_change"], reverse=True)[:100]

    # Extract selected query terms for the top 100 queries
    top_selected_query_terms = {query_id: selected_query_terms[query_id] for query_id, _ in top_queries}
    prepend_top_selected_query_terms = {query_id: selected_query_terms[query_id] for query_id, _ in prepend_top_queries}
    append_top_selected_query_terms = {query_id: selected_query_terms[query_id] for query_id, _ in append_top_queries}

    # Update the corpus_data to include only the top queries
    corpus_data = {query_id: data for query_id, data in top_queries}
    corpus_data_prepend = {query_id: data for query_id, data in prepend_top_queries}
    corpus_data_append = {query_id: data for query_id, data in append_top_queries}

    # Baseline
    diagnostic_dataset = {
        "corpus": corpus_data
    }

    # Prepend
    prepend_diagnostic_dataset = {
        "selected_query_terms": prepend_top_selected_query_terms,
        "corpus": corpus_data_prepend
    }

    # Append
    append_diagnostic_dataset = {
        "selected_query_terms": append_top_selected_query_terms,
        "corpus": corpus_data_append
    }

    # Save as JSON
    output_file = os.path.join(output_dir, "tfc1_add_baseline_final_dd_corpus.json")
    prepend_output_file = os.path.join(output_dir, "tfc1_add_prepend_final_dd_corpus.json")
    append_output_file = os.path.join(output_dir, "tfc1_add_append_final_dd_corpus.json")
    csv_output_file = os.path.join(output_dir, "tfc1_add_qids_with_text.csv")

    # Baseline
    with open(output_file, "w") as f:
        json.dump(diagnostic_dataset, f, indent=4)

    # Prepend
    with open(prepend_output_file, "w") as f:
        json.dump(prepend_diagnostic_dataset, f, indent=4)
    
    # Append
    with open(append_output_file, "w") as f:
        json.dump(append_diagnostic_dataset, f, indent=4)
    
    # CSV
    with open(csv_output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for query_id, _ in top_queries:
            writer.writerow([query_id, queries[query_id]])
        
    

    print(f"Diagnostic dataset prepared and saved as '{output_file}'.")
    print(f"Diagnostic dataset prepared and saved as '{prepend_output_file}'.")
    print(f"Diagnostic dataset prepared and saved as '{append_output_file}'.")
    print(f"Top queries saved to CSV at: {csv_output_file}")
    return diagnostic_dataset

def main():
    """
    1. Use the MM MARCO dataset.
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

if __name__ == "__main__":
    main()