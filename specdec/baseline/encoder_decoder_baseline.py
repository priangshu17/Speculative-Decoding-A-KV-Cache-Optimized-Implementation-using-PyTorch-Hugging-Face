"""
Baseline Encoder-Decoder autoregressive generation for Mp only.
"""

import torch 
from typing import List, Tuple 
from specdec.sampling import sample_from_logits
from specdec.utils import device


@torch.no_grad()
def baseline_generate_encoder_decoder(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens = 50,
    temperature = 1.0,
    top_k = 0,
    top_p = 0.0
) -> Tuple[List[int], List, float]:
    
    model.eval()
    inputs = tokenizer(prompt, return_tensors = "pt").to(device)
    
    encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
    enc_out = encoder(**inputs)
    
    start_token = tokenizer.pad_token_id or tokenizer.eos_token_id
    dec_input = torch.tensor([[start_token]], device = device)
    
    generated = []
    p_dists = []
    
    import time 
    start = time.perf_counter()
    
    for _ in range(max_new_tokens):
        out = model(encoder_outputs = enc_out, decoder_input_ids = dec_input)
        
        logits = out.logits[:, -1, :].squeeze(0)
        token, probs = sample_from_logits(logits, temperature, top_k, top_p)
        
        generated.append(token)
        p_dists.append(probs)
        
        dec_input = torch.cat([dec_input, torch.tensor([[token]], device = device)], dim = 1)
        
    total_time = time.perf_counter() - start 
    return generated, p_dists, total_time 