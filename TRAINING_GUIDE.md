# TinyLlama Self-Training & Distillation Guide

This guide details the design decisions, architectural optimization, and training techniques used to train our custom 260K-parameter TinyLlama model for the Raspberry Pi Pico W.

---

## 💡 Why this Approach was Chosen

Running LLMs on standard microcontrollers like the RP2040 is extremely challenging due to memory (264 KB SRAM) and storage (2 MB Flash) limits. 

We used **Knowledge Distillation** and a **Teacher-Student Self-Training Loop** because:
1. **Targeted Knowledge**: Rather than attempting to train a general-purpose model, we specialize the student model's 260K parameters on specific high-value behaviors (name identification, capability self-descriptions, and interactive greetings).
2. **Infinite Data Generation**: A local large model (`gemma4:31b-cloud`) acts as a teacher to dynamically synthesize training examples.
3. **Automated Error Correction**: Standard supervised fine-tuning lacks validation. Our self-training loop tests the student model at the end of each training cycle. If it outputs garbled text or incorrect details, the teacher generates new corrective dialogue patterns targeting those failures to retrain it.

---

## 🧠 Memory Optimization: The GQA Technique

In standard Transformer architectures, the KV (Key-Value) Cache size scales linearly with the sequence length.

### KV Cache Memory Calculation

The KV Cache size for a model is computed as:
$$\text{KV Cache Size} = 2 \times n_{\text{layers}} \times \text{seq\_len} \times d_{\text{kv}} \times \text{sizeof(float)}$$

Where:
* $n_{\text{layers}} = 5$
* $d_{\text{kv}} = \frac{\text{dim} \times n_{\text{kv\_heads}}}{n_{\text{heads}}}$
* $\text{dim} = 64$
* $n_{\text{heads}} = 8$

#### Case A: 4 KV Heads (Standard configuration)
* $n_{\text{kv\_heads}} = 4 \implies d_{\text{kv}} = 32$
* For a sequence length of **128**:
  $$\text{KV Cache Size} = 2 \times 5 \times 128 \times 32 \times 4 \text{ bytes} = 163,840 \text{ bytes (160 KB)}$$
* If we wanted to increase the context window to **256** under this setup:
  $$\text{KV Cache Size} = 2 \times 5 \times 256 \times 32 \times 4 \text{ bytes} = 327,680 \text{ bytes (320 KB)}$$
  *This exceeds the entire 264 KB SRAM of the RP2040, triggering an Out of Memory (OOM) crash.*

#### Case B: Grouped Query Attention (2 KV Heads)
* $n_{\text{kv\_heads}} = 2 \implies d_{\text{kv}} = 16$
* For a sequence length of **256**:
  $$\text{KV Cache Size} = 2 \times 5 \times 256 \times 16 \times 4 \text{ bytes} = 163,840 \text{ bytes (160 KB)}$$
  
By transitioning from MHA (Multi-Head Attention) to **GQA (Grouped Query Attention)** with 2 KV heads, we successfully **doubled the context window (from 128 to 256 tokens) with zero additional memory overhead**.

---

## 📈 Self-Training Loop Pipeline

```
[Start Loop]
    │
    ├── 1. Generate Dataset (Greetings, Name queries, Descriptions)
    │
    ├── 2. Train Student Model (PyTorch + CUDA 12.1)
    │
    ├── 3. Run Validation Prompts on Student Model
    │
    ├── 4. Grade Responses using Teacher Model (gemma4:31b-cloud)
    │     ├── PASS (All prompts correct) ──> Export to model_weights.h ──> [Done]
    │     │
    │     └── FAIL (Any prompt fails)
    │           └── 5. Teacher generates 3 targeted corrective samples
    │                 └── Append to Dataset ──> Loop again (step 2)
```

### PyTorch Training Configurations
* **Optimizer**: AdamW
* **Learning Rate**: 0.005
* **Epochs**: 150 per loop
* **Batch Size**: 32
* **Target masking**: Prompts are masked with `-100` so that loss is only calculated on the response tokens. This focuses backpropagation purely on response quality.

---

## ⚡ Host Acceleration (PyTorch + GPU)

* **GPU Used**: NVIDIA GeForce RTX 4060 Laptop GPU
* **Libraries**: PyTorch compiled with CUDA 12.1 (`torch-2.5.1+cu121`)
* **Benefit**: The parallel matrix multiplication operations are executed natively on GPU tensor cores, reducing the execution time of a training cycle from 40+ minutes on CPU to **under 45 seconds** on GPU. This enables rapid developer iteration.
