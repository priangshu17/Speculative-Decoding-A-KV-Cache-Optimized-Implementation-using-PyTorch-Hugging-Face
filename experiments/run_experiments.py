"""
Experiment harness compatible with KV-cache optimized speculative decoding.

Usage:
    python experiments/run_experiment.py
"""

import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from specdec.utils import set_seed, device
from specdec.metrics import js_divergence, compute_speedup
from specdec.baseline.encoder_decoder_baseline import baseline_generate_encoder_decoder
from specdec.speculative.encoder_decoder import speculative_generate_encoder_decoder

import matplotlib.pyplot as plt
import csv


def run_experiment():
    set_seed(42)

    # ---------------- CONFIG ----------------
    PROMPT = "Translate to German: The quick brown fox jumps over the lazy dog."
    MODEL_P = "t5-small"    # Mp (target)
    MODEL_Q = "t5-small"    # Mq (proposal model)
    GAMMAS = [1, 2, 4]      # gamma sweep
    N = 3                   # number of sequences per test
    SEQ_LEN = 30            # tokens per sequence
    TEMP = 1.0
    TOP_K = 50
    TOP_P = 0.0
    LENIENCE = None

    OUT_DIR = "./results_kvcache/"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------- LOAD MODELS ----------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_P)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model_p = AutoModelForSeq2SeqLM.from_pretrained(MODEL_P).to(device)
    model_q = AutoModelForSeq2SeqLM.from_pretrained(MODEL_Q).to(device)

    model_p.resize_token_embeddings(len(tokenizer))
    model_q.resize_token_embeddings(len(tokenizer))

    # ---------------- BASELINE ----------------
    print("Running baseline (Mp only)...")
    baseline_time = 0.0
    baseline_dists = []

    for i in range(N):
        prompt_i = f"{PROMPT} (example {i})"
        gen_b, p_b, t_b = baseline_generate_encoder_decoder(
            model_p,
            tokenizer,
            prompt_i,
            max_new_tokens=SEQ_LEN,
            temperature=TEMP,
            top_k=TOP_K,
            top_p=TOP_P,
        )
        baseline_time += t_b
        baseline_dists.append(p_b)
        print(f"  baseline seq {i}: time {t_b:.3f}s")

    print(f"Baseline total time: {baseline_time:.3f}s\n")

    # ---------------- SPECULATIVE ----------------
    results = []

    for gamma in GAMMAS:
        print(f"Running speculative decoding with gamma={gamma} ...")
        total_spec_time = 0.0
        total_mp = 0
        total_mq = 0
        total_acc = 0
        total_prop = 0
        js_vals = []

        for i in range(N):
            prompt_i = f"{PROMPT} (example {i})"

            gen_s, p_s, stats = speculative_generate_encoder_decoder(
                model_p,
                model_q,
                tokenizer,
                prompt_i,
                max_new_tokens=SEQ_LEN,
                gamma=gamma,
                temperature=TEMP,
                top_k=TOP_K,
                top_p=TOP_P,
                lenience=LENIENCE,
            )

            total_spec_time += stats["total_time"]
            total_mp += stats["mp_steps"]
            total_mq += stats["mq_steps"]
            total_acc += stats["accepted_tokens"]
            total_prop += stats["total_proposals"]

            # Compute JS divergence for this sequence
            bdist = baseline_dists[i]
            L = min(len(bdist), len(p_s))
            if L > 0:
                js = np.mean([js_divergence(bdist[k], p_s[k]) for k in range(L)])
                js_vals.append(js)

            print(
                f"  seq {i}: time={stats['total_time']:.3f}s, "
                f"mp_steps={stats['mp_steps']}, mq_steps={stats['mq_steps']}, "
                f"accepted={stats['accepted_tokens']}/{stats['total_proposals']}"
            )

        speedup = compute_speedup(baseline_time, total_spec_time)
        alpha = total_acc / total_prop if total_prop else 0.0
        js_avg = float(np.mean(js_vals)) if js_vals else None

        results.append(
            dict(
                gamma=gamma,
                speedup=speedup,
                empirical_alpha=alpha,
                avg_js_divergence=js_avg,
                spec_time=total_spec_time,
                base_time=baseline_time,
                mp_steps=total_mp,
                mq_steps=total_mq,
            )
        )

        print(f"  >>> gamma={gamma}, speedup={speedup:.3f}, "
              f"alpha={alpha:.3f}, JS={js_avg}\n")

    # ---------------- SAVE CSV ----------------
    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {csv_path}")

    # ---------------- PLOTS ----------------
    gammas = [r["gamma"] for r in results]
    speeds = [r["speedup"] for r in results]
    alphas = [r["empirical_alpha"] for r in results]
    jses = [r["avg_js_divergence"] for r in results]

    plt.plot(gammas, speeds, marker="o")
    plt.xlabel("gamma")
    plt.ylabel("speedup")
    plt.title("Speedup vs gamma")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "speedup.png"))
    plt.close()

    plt.plot(gammas, alphas, marker="o")
    plt.xlabel("gamma")
    plt.ylabel("empirical alpha")
    plt.title("Alpha vs gamma")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "alpha.png"))
    plt.close()

    plt.plot(gammas, jses, marker="o")
    plt.xlabel("gamma")
    plt.ylabel("Avg JS divergence")
    plt.title("JS divergence vs gamma")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "js.png"))
    plt.close()


if __name__ == "__main__":
    run_experiment()
