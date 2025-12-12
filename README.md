# Speculative Decoding — A KV-Cache Optimized Implementation (PyTorch + Hugging Face)

This repository contains a clean, modular, and research-oriented implementation of **Speculative Decoding** for encoder–decoder transformers.  
The project includes:

- A **baseline autoregressive decoder** (Mp)
- A **KV-cache optimized speculative decoding algorithm** using a smaller proposal model (Mq)
- Proper **acceptance–rejection logic** as described in *Fast Inference from Transformers via Speculative Decoding*
- **Speedup benchmarking**, **JS divergence measurement**, **acceptance rate (α)**, and **per-token Mp distribution tracking**
- A full **experiment harness** + **plots** to visualize how γ (gamma) affects performance and fidelity

This implementation is optimized to run efficiently on **consumer GPUs** (e.g., RTX 4050 6GB), with memory-efficient KV-cache decoding and single-token forward passes.

---

## 🚀 Highlights

- **Supports encoder–decoder architectures** (T5, BART, etc.)
- Uses **past_key_values** to ensure *constant memory per decoding step*
- Modular file layout (no monolithic scripts)
- Experiment runner generates:
  - Speedup vs γ plot  
  - α (acceptance rate) vs γ plot  
  - JS divergence vs γ plot  
  - CSV logs for reproducibility  

The codebase is designed for learning, experimentation, and potential extension into research projects.

---

## 📂 Project Structure

IMPLEMENTING_SPECULATIVE_DECODING/
│
├── specdec/
│ ├── init.py
│ ├── utils.py
│ ├── sampling.py
│ ├── metrics.py
│ │
│ ├── baseline/
│ │ ├── init.py
│ │ └── encoder_decoder_baseline.py
│ │
│ ├── speculative/
│ ├── init.py
│ ├── encoder_decoder.py # KV-cache optimized speculative decoding
│ └── batching.py # helpers (kept minimal)
│
├── experiments/
│ ├── init.py
│ └── run_experiment.py # full benchmarking harness
│
├── pyproject.toml # installable package config
├── README.md
└── results/ # saved results & plots


---

## 📘 Background: What Is Speculative Decoding?

Speculative Decoding accelerates transformer inference by:

1. Using a **fast approximator model (Mq)** to *guess* γ future tokens.
2. Using the **large target model (Mp)** to **validate** or **reject** these guesses.
3. If guesses look good, they are accepted *without* calling Mp for every single token.
4. When a guess fails, Mp samples the next token itself.

This allows:

- **Fewer Mp forward passes**
- **Near-identical output quality**
- **Significant speedups** (especially when Mp ≫ Mq)

---

## 📐 Acceptance Rule (Short Summary)

A guess token *xi* from Mq is accepted if:
u ≤ p(xi) / q(xi)

Where:
- `p(xi)` = Mp’s probability  
- `q(xi)` = Mq’s probability  
- `u ~ Uniform(0,1)`  

If rejected:
- We fall back to sampling from **p0(x)** = max(0, p(x) − q(x)) normalized.

This implementation faithfully follows the original paper.

---

## ⚡ KV-Cache Optimization (Why It Matters)

Traditional speculative decoding recomputes large padded batches.  
This implementation avoids that by:

- Calling Mp and Mq with **only one new token at each step**
- Reusing **past_key_values**
- Keeping GPU memory extremely low
- Making speculative decoding viable on **6GB GPUs**

This is a *production-style* speculative decoder.

---

## 📊 Metrics & Plots

Running the experiment script generates:

### **1. Speedup vs Gamma**
Shows how speculative decoding compares to baseline autoregressive decoding.

### **2. Acceptance Rate α vs Gamma**
Measures proposal quality from Mq.

### **3. JS Divergence vs Gamma**
Measures divergence between Mp-only outputs and speculative outputs.

Plots + CSV logs are saved automatically.

---

## ▶️ Running the Code

### 1. Install the package locally
```bash
pip install -e .
```

### 2. Run the Experiments harness
python experiments/run_experiment.py

### 3. View Results
Plots and logs appear under:
results_kvcache/
    speedup.png
    alpha.png
    js.png
    results.csv
    


















