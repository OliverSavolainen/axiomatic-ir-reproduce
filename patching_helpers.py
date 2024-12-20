import torch
from tqdm import tqdm

from TransformerLens.transformer_lens import HookedEncoder, ActivationCache
from transformer_lens import patching
import TransformerLens.transformer_lens.utils as utils

from jaxtyping import Float
from typing import Callable
from functools import partial

'''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
'''
def patch_residual_component(
    corrupted_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos,
    clean_cache,
):
    corrupted_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_component


'''
Returns an array of results of patching each position at each layer in the residual
stream, using the value from the clean cache.

The results are calculated using the patching_metric function, which should be
called on the model's logit output.
'''
def get_act_patch_block_every(
    model: HookedEncoder, 
    device,
    q_embedding,
    og_score,
    p_score,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], float]
) -> Float[torch.Tensor, "3 layer pos"]:

    model.reset_hooks()
    _, seq_len = corrupted_tokens["input_ids"].size()
    results = torch.zeros(3, model.cfg.n_layers, seq_len, device=device, dtype=torch.float32)

    # send tokens to device if not already there
    corrupted_tokens["input_ids"] = corrupted_tokens["input_ids"].to(device)
    corrupted_tokens["attention_mask"] = corrupted_tokens["attention_mask"].to(device)

    for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
        print("Patching:", component)
        for layer in tqdm(range(model.cfg.n_layers)):
            for position in range(seq_len):
                hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
                patched_outputs = model.run_with_hooks(
                    corrupted_tokens["input_ids"],
                    one_zero_attention_mask=corrupted_tokens["attention_mask"],
                    return_type="embeddings",
                    fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
                )
                patched_embedding = patched_outputs[:,0,:].squeeze(0)
                results[component_idx, layer, position] = patching_metric(patched_embedding,q_embedding,og_score,p_score)

    return results


'''
Patches the output of a given head (before it's added to the residual stream) at
every sequence position, using the value from the clean cache.
'''
def patch_head_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook, #: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector

def pad_clean_cache(clean_cache, corrupted_tokens, model):
    """
    Vectorizes the padding of clean cache tensors to align with corrupted tokens.

    Args:
        clean_cache (ActivationCache): The clean cache containing activations to be padded.
        corrupted_tokens (dict): Dictionary containing input_ids and attention_mask of corrupted tokens.
        model (HookedEncoder): The model whose layers and dimensions determine cache structure.

    Returns:
        dict: Updated clean_cache with all tensors padded to match corrupted tokens' dimensions.
    """
    # Extract relevant corrupted token sizes
    corrupted_len = corrupted_tokens["input_ids"].size(1)  # Sequence length
    device = corrupted_tokens["input_ids"].device  # Ensure we use the same device

    # Initialize a new dictionary for the padded cache
    padded_clean_cache = {}

    # Iterate over components in the clean cache
    for key, tensor in clean_cache.items():
        if tensor.dim() < 3:  # Skip tensors that don't have positional dimensions
            padded_clean_cache[key] = tensor.to(device)  # Move to correct device
            continue

        # Tensor dimensions: [batch, seq_len, ...]
        clean_len = tensor.size(1)

        if clean_len < corrupted_len:
            # Calculate padding dimensions
            pad_size = corrupted_len - clean_len

            # Generate padding along the sequence dimension
            padding_shape = [tensor.size(0), pad_size, *tensor.shape[2:]]  # Maintain batch and other dimensions
            padding_tensor = torch.zeros(padding_shape, dtype=tensor.dtype, device=device)  # Ensure device matches

            # Concatenate padding along the sequence length dimension
            padded_clean_cache[key] = torch.cat([tensor.to(device), padding_tensor], dim=1)

        elif clean_len > corrupted_len:
            # Trim excess tokens to match corrupted_len
            padded_clean_cache[key] = tensor[:, :corrupted_len, ...].to(device)

        else:
            # No padding needed
            padded_clean_cache[key] = tensor.to(device)

    return padded_clean_cache
'''
Returns an array of results of patching at all positions for each head in each
layer, using the value from the clean cache.

The results are calculated using the patching_metric function, which should be
called on the model's embedding output.
'''
def get_act_patch_attn_head_out_all_pos(
    model: HookedEncoder, 
    device,
    q_embedding,
    og_score,
    p_score,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable
) -> Float[torch.Tensor, "layer head"]:

    model.reset_hooks()
    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
    print("Patching: attn_heads")
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            hook_fn = partial(patch_head_vector, head_index=head, clean_cache=clean_cache)
            patched_outputs = model.run_with_hooks(
                corrupted_tokens["input_ids"],
                one_zero_attention_mask=corrupted_tokens["attention_mask"],
                return_type="embeddings",
                fwd_hooks = [(utils.get_act_name("z", layer), hook_fn)],
            )
            patched_embedding = patched_outputs[:,0,:].squeeze(0)
            results[layer, head] = patching_metric(patched_embedding,q_embedding,og_score,p_score)

    return results


def patch_head_vector_by_pos_pattern(
    corrupted_activation: Float[torch.Tensor, "batch pos head_index pos_q pos_k"],
    hook, #: HookPoint, 
    pos,
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos head_index d_head"]:

    corrupted_activation[:,head_index,pos,:] = clean_cache[hook.name][:,head_index,pos,:]
    return corrupted_activation


def patch_head_vector_by_pos(
    corrupted_activation: Float[torch.Tensor, "batch pos head_index d_head"],
    hook, #: HookPoint, 
    pos,
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos head_index d_head"]:

    corrupted_activation[:, pos, head_index] = clean_cache[hook.name][:, pos, head_index]
    return corrupted_activation


def get_act_patch_attn_head_by_pos(
    model: HookedEncoder, 
    device,
    q_embedding,
    og_score,
    p_score,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable,
    layer_head_list,
) -> Float[torch.Tensor, "layer pos head"]:
    
    model.reset_hooks()
    _, seq_len = corrupted_tokens["input_ids"].size()
    results = torch.zeros(2, len(layer_head_list), seq_len, device=device, dtype=torch.float32)

    for component_idx, component in enumerate(["z", "pattern"]):
        for i, layer_head in enumerate(layer_head_list):
            layer = layer_head[0]
            head = layer_head[1]
            for position in range(seq_len):
                patch_fn = patch_head_vector_by_pos_pattern if component == "pattern" else patch_head_vector_by_pos
                hook_fn = partial(patch_fn, pos=position, head_index=head, clean_cache=clean_cache)
                patched_outputs = model.run_with_hooks(
                    corrupted_tokens["input_ids"],
                    one_zero_attention_mask=corrupted_tokens["attention_mask"],
                    return_type="embeddings",
                    fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
                )
                patched_embedding = patched_outputs[:,0,:].squeeze(0)
                results[component_idx, i, position] = patching_metric(patched_embedding,q_embedding,og_score,p_score)

    return results