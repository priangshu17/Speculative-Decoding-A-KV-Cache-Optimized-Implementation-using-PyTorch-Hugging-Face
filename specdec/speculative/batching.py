"""
Batching helpers for speculative decoding.
"""

import torch 
from specdec.utils import left_pad_and_stack

def make_prefix_batch(decoder_input_ids, guesses, pad_id):
    """
    Construct list of decoder prefix sequences:
        prefix,
        prefix + x1,
        prefix + x1, x2,
        ...
    Returns stacked (gamma+1, max_len)
    """
    seqs = []
    for i in range(len(guesses) + 1):
        s = decoder_input_ids.clone()
        if i > 0:
            append = torch.tensor([guesses[:i]], dtype = s.dtype, device = s.device)
            s = torch.cat([s, append], dim = 1)
        seqs.append(s[0])
    return left_pad_and_stack(seqs, pad_id)