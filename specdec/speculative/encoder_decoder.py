"""
KV-cache optimized speculative decoding for encoder-decoder models (Mp, Mq).

Key Ideas / Optimizations:
- Run encoder once for Mp and Mq
- Use 'past_key_values' (KV Cache) and 'use_cache = True' to perform incremental single-token forwards instead of recomputing entire sequences or batching many long padded sequences.
- Generate proposals from Mq by advancing its past_key_values sequentially, which is cheap because each forward only computes a single token.
- Evaluate Mp sequentially with small single-token forwards that update Mp's past_key_values. Acceptance decisions are made using per-token distributions from Mp and q distributions from Mq.
- This approach uses constant memory per step (only KV caches and logits), avoiding large temporary tensors that causes CUDA OOM.


Behavioral Notes:
- The implementation follows the acceptance cascade from this Speculative Decoding paper: for i = 1..gamma, accept xi if u <= p_i(xi) / q_i(xi) (with optional lenience).
- After accepting some guesses, we update Mp's past_key_values incrementally.
- When a guess is rejected (or all guesses accepted), sample one final token from p0 = norm(max(0, p - q)) (or from p if all accepted).
- We record per-token Mp distributions (for JS divergence / fidelity checks).
- Designed for clarity and correctness first; can be further micro-optimized.

Requires:
    torch, transformers, numpy
"""


from typing import List, Tuple, Optional, Dict 
import time 
import numpy as np
import torch 
import torch.nn.functional as F 

from specdec.sampling import sample_from_logits
from specdec.utils import device 
from specdec.metrics import js_divergence

@torch.no_grad()
def speculative_generate_encoder_decoder(
    model_p,
    model_q,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    gamma: int = 4,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    lenience: Optional[float] = None,
):
    """
    KV-cache optimized speculative decoding for encoder-decoder models.
    
    Args:
        model_p: high-quality target model (Mp), a Hugging Face seq2seq model.
        model_q: fast approximator (Mq), a Hugging Face seq2seq model (same tokenizer).
        tokenizer: HF tokenizer used by both models.
        prompt: input text for encoder.
        max_new_tokens: number of tokens to generate.
        gamma: number of speculative proposals per iterations.
        temperature, top_k, top_p: sampling controls applied equally to Mp and Mq when proposing/sampling.
        lenience: optional multiplier on q-probabilities during acceptance check (e.g., >1 to relax).
        
    Returns:
        generated_ids: list of genrerated token ids (integers)
        per_token_p: list of Mp per-token probability vectors (numpy arrays) for each produced token.
        stats: dict with keys:
            - total_time
            - mp_steps (number of single-token Mp forwards performed)
            - mq_steps (number of single-token Mq forwards performed)
            - accepted_tokens (number of proposals accepted)
            - total_proposals (gamma * iterations attempted)
    """
    
    model_p.eval()
    model_q.eval()
    
    # 1. Run encoders once
    enc_inputs = tokenizer(prompt, return_tensors = "pt").to(device)
    # Use model.get_encoder() when available (T5), else model.encoder
    encoder_fn_p = model_p.get_encoder() if hasattr(model_q, "get_encoder") else model_p.encoder
    encoder_fn_q = model_q.get_encoder() if hasattr(model_q, "get_encoder") else model_q.encoder
    
    encoder_out_p = encoder_fn_p(**enc_inputs)
    encoder_out_q = encoder_fn_q(**enc_inputs)
    
    # 2. Initialize decoder starting token (pad or eos depending on tokenizer/model)
    start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # We'll feed single-token inputs and use past_key_values to advance the decoder.
    last_token = torch.tensor([[start_token_id]], device = device, dtype = torch.long)
    
    # Keep track of past_key_values for Mp and Mq (None means empty state)
    past_p = None 
    past_q = None 
    
    
    generated_ids: List[int] = []
    per_token_p: List[np.ndarray] = []
    
    # Stats
    mp_steps = 0  # Single=token forwards on Mp
    mq_steps = 0  # Single-token forwards on Mq
    accepted_tokens = 0
    total_proposals = 0
    
    t_start = time.perf_counter()
    steps = 0
    
    # We'll maintain a "current Mp next-token distribution" (p_current) so each loop we can use it.
    # p_current will be computed on-demand via a single-token forward on Mp using past_p and last_token
    def mp_next_distribution_and_new_past(past, last_token_tensor):
        """
        Run Mp for single-token forward and return (probs_numpy, new_past_key_values).
        """
        # model_p expects encoder_outputs and decoder_input_ids and optional past_key_values
        out = model_p(
            encoder_outputs = encoder_out_p,
            decoder_input_ids = last_token_tensor,
            past_key_values = past,
            use_cache = True,
        )
        # out.logits shape: (batch = 1, seq_len = 1, vocab)
        logits = out.logits[:, -1, :].squeeze(0)  # (vocab,)
        probs = F.softmax(logits / float(1.0), dim = -1).cpu().numpy()  # We'll apply temperature/top-k/top-p when sampling
        new_past = out.past_key_values  # cached for next single-token steps
        return probs, new_past, logits  # return logits (tensor) too if caller wants to apply top-k/top-p/temperature
    
    def mq_generate_gamma(past, last_token_tensor, gamma_count):
        """
        Autoregressively generate gamma_count tokens from Mq, using and updating past.
        Returns guesses (list of token ids), q_dists (list of numpy probs), and final past_q.
        This advances past sequentially and is memory-light.
        """
        guesses = []
        q_dists = []
        cur_past = past 
        cur_token = last_token_tensor
        for _ in range(gamma_count):
            out_q = model_q(
                encoder_outputs = encoder_out_q,
                decoder_input_ids = cur_token,
                past_key_values = cur_past,
                use_cache = True,
            )
            # single-token forward
            logits_q = out_q.logits[:, -1, :].squeeze(0)  # tensor
            # Apply sampling constraints (temperature/top-k/top-p) on logits for token generation
            token_q, probs_q = sample_from_logits(logits_q, temperature=temperature,top_k=top_k, top_p=top_p)
            guesses.append(int(token_q))
            q_dists.append(probs_q)
            cur_past = out_q.past_key_values
            # Next decoder input is the newly generated token
            cur_token = torch.tensor([[token_q]], device = device, dtype = torch.long)
        return guesses, q_dists, cur_past 
    
    # Main Loop: Each iteration proposes gamma tokens (via Mq) and performs the acceptance cascade using Mp.
    while steps < max_new_tokens:
        # (A) Generate gamma proposals and their q-distributions from Mq
        guesses, q_dists, past_q = mq_generate_gamma(past_q, last_token, gamma)
        mq_steps += len(guesses)
        total_proposals += len(guesses)
        
        # (B) Evaluate Mp sequentially during the acceptance cascade.
        # We'll compute Mp's distribution on current prefix (p_current)
        p_current_probs, _, logits_tensor = mp_next_distribution_and_new_past(past_p, last_token)
        mp_steps += 1
        # Note: We have NOT updated past_p yet; past_p still reflects the prefix before any new accepted tokens.
        # We'll store a transient variable past_p_temp which we will update only when we accept tokens.
        past_p_temp = past_p 
        # We also need a variable to hold logits/probs for the current position that we can update as we accept tokens
        p_current = p_current_probs  # Numpy Vector
        
        accepted_in_this_round = 0
        rejected_index = None
        
        
        # Acceptance cascade: iterate through guesses and decide accept/reject sequentially.
        for i, guess in enumerate(guesses):
            q_prob = q_dists[i][guess]  # Scalar
            p_prob = float(p_current[guess])  # Scalar
            # Apply lenience by scaling q_prob upward (makes acceptance easier)
            q_eff = float(q_prob) * (float(lenience) if lenience is not None else 1.0)
            # Draw Uniform random number for acceptance test
            r = np.random.rand()
            ratio = p_prob / (q_eff + 1e-40)
            if ratio >= r:
                # Accept this guesses token
                accepted_in_this_round += 1
                accepted_tokens += 1
                # Append token to outputs
                generated_ids.append(int(guess))
                per_token_p.append(p_current.copy())  # Store Mp's distribution at this accepted step
                
                
                # Advance Mp past by running a single-token forward with the accepted token
                # We call Mp with decoder_input_ids = token and past_key_values = past_p_temp
                token_tensor = torch.tensor([[int(guess)]], device = device, dtype = torch.long)
                out_p_step = model_p(
                    encoder_outputs = encoder_out_p,
                    decoder_input_ids = token_tensor,
                    past_key_values = past_p_temp,
                    use_cache = True,
                )
                mp_steps += 1  # we executed another Mp single-token forward to update cache
                # Update past_p_temp and set p_current to the logits/probs for the next position
                past_p_temp = out_p_step.past_key_values
                logits_next = out_p_step.logits[:, -1, :].squeeze(0)
                p_current = F.softmax(logits_next, dim = -1).cpu().numpy()
                # Update last_token to the accepted token for Mq continuation (Mq's past was already advanced while proposing)
                last_token = torch.tensor([[int(guess)]], device = device, dtype = torch.long)
                # Also update past_p to the new past (we will commit later after fallback sampling)
                # Note: We don't commit to past_p until the end of the iteration; but it's safe to sync it as we accept:
                past_p = past_p_temp 
                # Increment steps and break it done
                steps += 1
                if steps >= max_new_tokens:
                    break 
                # Continue to next guess
            else:
                # Rejection occurs at index i
                rejected_index = i
                # We stop accepting further guesses; we'll apply fallback sampling using p_current and q_dists[i]
                break 
        # If we've reached max_new_tokens during acceptance loop, finish
        if steps >= max_new_tokens:
            break 
        
        # If no rejection occurred (all guesses accepted):
        if rejected_index is None:
            # We have accepted all gamma guesses. At this point p_current is the distribution at the next position
            # (already updated during last accepted token). According to the algorithm, we now need to sample one
            # additional token from p_current (because each speculative iteration produces at least one token).
            # So sample token from p_current (with top-k/top-p/temperature), append it, and update Mp past.
            logits_np = np.log(p_current + 1e-40)
            logits_tensor = torch.tensor(logits_np, device=device)  # on CPU/GPU as necessary
            # SAMPLE using our standardized sampler (which expects logits tensor)
            token_t, _ = sample_from_logits(logits_tensor, temperature=temperature, top_k=top_k, top_p=top_p)
            # append and update past_p by running Mp single-token forward
            generated_ids.append(int(token_t))
            per_token_p.append(p_current.copy())
            # update Mp past with this token
            token_t_tensor = torch.tensor([[int(token_t)]], device=device, dtype=torch.long)
            out_p_after = model_p(
                encoder_outputs=encoder_out_p,
                decoder_input_ids=token_t_tensor,
                past_key_values=past_p,
                use_cache=True,
            )
            mp_steps += 1
            past_p = out_p_after.past_key_values
            last_token = token_t_tensor
            steps += 1
            # continue main loop
            continue

        # If there was a rejection at index rejected_index:
        # p_current is the Mp distribution at that rejection position (i.e., after accepting previous tokens)
        # q_next corresponds to q_dists[rejected_index]
        q_next = q_dists[rejected_index]
        # Form p0(x) = norm(max(0, p_current - q_next))
        p_next = p_current.copy()
        p0 = p_next - q_next
        p0[p0 < 0] = 0.0
        s = p0.sum()
        if s <= 1e-40:
            # degenerate: fallback to p_next normalized
            p_sample = p_next / (p_next.sum() + 1e-40)
        else:
            p_sample = p0 / s

        # Sample token from p_sample (respect top-k/top-p/temperature by converting to logits)
        logits_sample = np.log(p_sample + 1e-40)
        logits_tensor = torch.tensor(logits_sample, device=device)
        token_t, _ = sample_from_logits(logits_tensor, temperature=temperature, top_k=top_k, top_p=top_p)

        # Append sampled token and update Mp past by running single-token forward with that token
        generated_ids.append(int(token_t))
        per_token_p.append(p_next.copy())  # store original p (Mp distribution) for this produced token
        token_t_tensor = torch.tensor([[int(token_t)]], device=device, dtype=torch.long)
        out_p_after = model_p(
            encoder_outputs=encoder_out_p,
            decoder_input_ids=token_t_tensor,
            past_key_values=past_p,
            use_cache=True,
        )
        mp_steps += 1
        past_p = out_p_after.past_key_values
        last_token = token_t_tensor
        steps += 1

        # Important: At this point Mq's past_key_values was already advanced while proposing guesses.
        # However, last_token used by Mq for the next round should reflect the *actual* last token generated
        # (which might differ from the last guessed token if a rejection happened and we sampled token_t).
        # Mq's earlier past (past_q) assumed the guesses were appended; that is fine because Mq proposals were *hypothetical*.
        # To keep Mq in sync with the real generated sequence for future proposals, we must re-align past_q.
        # The simplest approach: reset past_q to None (i.e., force Mq to start proposals from the new last_token)
        # and let Mq autoregressively rebuild its past (cheap because gamma is small).
        # This avoids complex past-key-value surgery. It is a slight extra cost but saves correctness headaches.
        past_q = None

    total_time = time.perf_counter() - t_start
    stats: Dict = {
        "total_time": total_time,
        "mp_steps": mp_steps,
        "mq_steps": mq_steps,
        "accepted_tokens": accepted_tokens,
        "total_proposals": total_proposals,
    }
    return generated_ids, per_token_p, stats