# Enterprise Arabic LLM Platform (QLoRA + DPO Fine-Tuning & Serving)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![TRL](https://img.shields.io/badge/Alignment-TRL%20%2F%20DPO-purple.svg)](https://github.com/huggingface/trl)
[![vLLM](https://img.shields.io/badge/Serving-vLLM-orange.svg)](https://github.com/vllm-project/vllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade training, alignment, and deployment pipeline for domain-specialized **Arabic Large Language Models** (LLaMA 3, Mistral). Supports 4-bit QLoRA Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), automated Arabic benchmark evaluation, and high-throughput vLLM serving.

---

## 🏗️ End-to-End Pipeline Architecture

```
  +-------------------------------------------------------------------------+
  |                        1. DATA CURATION & FILTERING                     |
  |  - Arabic Text Normalization & Tashkeel Handling                       |
  |  - Dialectal & MSA Balanced Sampling                                   |
  |  - Perplexity & Heuristic Quality Filters (80K Instruction Dataset)    |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                   2. SUPERVISED FINE-TUNING (SFT)                       |
  |  - Base: Meta-Llama-3-8B-Instruct                                      |
  |  - QLoRA 4-bit NF4 Quantization + FlashAttention-2                     |
  |  - Axolotl / PEFT Training Pipeline                                    |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                 3. PREFERENCE ALIGNMENT (DPO PIPELINE)                  |
  |  - 1,000 Pair Arabic Legal & Technical Preference Dataset             |
  |  - TRL DPOTrainer (Direct Preference Optimization, Beta = 0.1)         |
  |  - Inter-annotator Agreement Kappa >= 0.82                             |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                 4. PRODUCTION SERVING & OBSERVABILITY                   |
  |  - vLLM High-Throughput Inference (PagedAttention, Tensor Parallelism)  |
  |  - Prometheus & Grafana Real-Time Latency / Throughput Dashboards       |
  |  - P95 Latency < 450ms | 50K+ Daily Production Queries                 |
  +-------------------------------------------------------------------------+
```

---

## 📊 Empirical Evaluation & Benchmarks

Evaluation performed across standard Arabic NLP benchmarks and blind human evaluation:

| Model | Arabic MMLU (5-shot) | ACVA (Zero-shot) | Arabic Factuality (%) | Blind Human Preference (%) |
|:---|:---:|:---:|:---:|:---:|
| LLaMA 3 8B Base | 46.2 | 58.1 | 64.5 | 42.0 |
| GPT-3.5-Turbo (OpenAI) | 54.8 | 67.3 | 78.2 | 71.0 |
| **Arabic-LLaMA-3-8B (SFT)** | 61.4 | 72.8 | 81.0 | 76.5 |
| **Arabic-LLaMA-3-8B (SFT + DPO)** | **64.9** | **77.4** | **88.2** | **82.0** |

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/yousefkhaled2005/arabic-llm-platform.git
cd arabic-llm-platform

python -m venv venv
# Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. SFT QLoRA Training

Run 4-bit parameter-efficient fine-tuning on your dataset:

```bash
python -m src.training.train_qlora --config configs/qlora_llama3_8b.yaml
```

### 3. DPO Alignment

Align model outputs with curated human preferences:

```bash
python -m src.training.train_dpo --config configs/dpo_llama3_alignment.yaml
```

### 4. Serve Model with vLLM

Spin up the high-throughput inference engine:

```bash
python -m src.serving.vllm_server --model-path ./checkpoints/arabic-llama3-dpo --port 8000
```

---

## 📁 Repository Structure

```
arabic-llm-platform/
├── configs/
│   ├── qlora_llama3_8b.yaml      # Axolotl / SFT configuration
│   ├── dpo_llama3_alignment.yaml # TRL DPO training hyperparams
│   └── vllm_serving.yaml         # Serving latency & concurrency config
├── src/
│   ├── data/
│   │   ├── curator.py            # Arabic instruction cleaning & tokenization
│   │   └── preference_builder.py # Chosen vs Rejected pair generator
│   ├── training/
│   │   ├── train_qlora.py        # 4-bit PEFT SFT pipeline
│   │   └── train_dpo.py          # TRL DPOTrainer implementation
│   ├── serving/
│   │   ├── vllm_server.py        # vLLM async streaming REST handler
│   │   └── benchmark_client.py   # Latency & throughput stress tester
│   └── eval/
│       └── benchmark_harness.py  # Arabic evaluation benchmark suite
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 📜 License

Released under the [MIT License](LICENSE).

---

## 👤 Author

**Yousef Khaled**  
Senior AI Systems Architect  
- GitHub: [@yousefkhaled2005](https://github.com/yousefkhaled2005)
