"""
Utility helpers: padding, encoder replication, simple random seed handling
"""

import torch 
from transformers import PreTrainedModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    import numpy as np
    import random 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def left_pad_and_stack(seqs, pad_id):
    """
    Takes a list of 1D LongTensor sequences,
    returns tensor of shape (batch, max_len)
    """
    max_len = max(s.shape[0] for s in seqs)
    padded = []
    for s in seqs:
        pad_len = max_len - s.shape[0]
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), pad_id, dtype = s.dtype, device = s.device)
            s = torch.cat([pad_tensor, s], dim = 0)
        padded.append(s.unsqueeze(0))
    return torch.cat(padded, dim = 0)  # (B, max_len)


def replicate_encoder_outputs(encoder_outputs, batch_size: int):
    """
    Replicates encoder_outputs.last_hidden_state for batch usage.
    Works for T5/BART style HF seq2seq models.
    """
    if hasattr(encoder_outputs, "last_hidden_state"):
        lh = encoder_outputs.last_hidden_state
        return {"last_hidden_state": lh.expand(batch_size, -1, -1).contiguous()}
    else:
        try:
            lh = encoder_outputs["last_hidden_state"]
            return {"last_hidden_state": lh.expand(batch_size, -1, -1).contiguous()}
        except:
            raise RuntimeError("Unsupported encoder_outputs structure.")
