"""
Sampling utilitie: temperature, top-k, top-p filtering.
"""

import torch 
import torch.nn.functional as F 
import numpy as np 

def top_k_filter(logits: torch.Tensor, top_k: int):
    if top_k <= 0:
        return logits 
    v, _ = torch.topk(logits, top_k)
    min_v = v[..., -1, None]
    return logits.masked_fill(logits < min_v, -1e10)


def top_p_filter(logits: torch.Tensor, top_p: float):
    if not (0.0 < top_p < 1.0):
        return logits 
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim = -1)
    cum_probs = torch.cumsum(sorted_probs, dim = -1)
    cutoff_mask = cum_probs > top_p 
    first_cutoff = torch.argmax(cutoff_mask.int(), dim = -1)
    cutoff_vals = sorted_logits[range(sorted_logits.size(0)), first_cutoff].unsqueeze(-1)
    return torch.where(logits < cutoff_vals, torch.tensor(-1e10, device = logits.device), logits)


def sample_from_logits(logits: torch.Tensor, temperature=1.0, top_k=0, top_p=0.0):
    """
    Takes 1D tensor logits -> returns (token_id, numpy_prob_vector)
    """
    logits = logits.clone()
    if temperature != 1.0:
        logits = logits / float(temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits.unsqueeze(0), top_p).squeeze(0)
    
    probs = F.softmax(logits, dim = -1).cpu().numpy()
    token = int(np.random.choice(len(probs), p = probs))
    return token, probs