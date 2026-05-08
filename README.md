<div align="center">

# 🧠 Difficulty-Aware Adaptive Reasoning Budget Allocation for LLMs

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0+cu128-orange.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

**IS 597 LLM · University of Illinois Urbana-Champaign · Spring 2026**

*Atharva Chaudhari · ac151@illinois.edu*

</div>

---

## 📌 Overview

Large language models use the same reasoning budget for every question, regardless of difficulty. Easy questions waste tokens. Hard questions may not get enough. This project trains a lightweight **budget policy network** that predicts how many reasoning tokens a frozen LLM should use **per question**, before reasoning begins.

> 🏆 **Best result: 80% accuracy at 185 avg tokens** (Model 9, PPO + Qwen2.5-7B)
> outperforming the fixed 192-token baseline of 76% while using **fewer tokens**

---

## 🔧 How It Works

```
Question ──► Policy Network ──► Budget B ──► LLM Reasoning (B tokens) ──► Answer Extraction ──► Final Answer
```

- The **policy network** (small MLP, ~170K params) predicts the token budget per question
- The **frozen LLM** generates reasoning within that budget, then extracts the answer
- Only the policy network is trained. The LLM never changes.

---

## 📁 Repository Structure

```
.
├── model1_reinforce.py          # M1: REINFORCE, 1.5B, 80 train / 20 test
├── model2_actor_critic.py       # M2: Actor-Critic + soft reward, 200/50
├── model3_warmstart_a2c.py      # M3: Warm-start + A2C, 300/100
├── model4_hard_reward.py        # M4: Hard reward + exploration floor
├── model5_oracle_blend.py       # M5: Oracle blend + entropy fix
├── model6_ppo_highest.py        # M6: PPO + highest reward oracle
├── model7_ppo_mincorrect.py     # M7: PPO + minimum correct oracle
├── model8_rich_features.py      # M8: PPO + 446-dim features, 1.5B
├── model9_ppo_7b.py             # M9: PPO + Qwen2.5-7B ⭐ Best Model
├── model10_dpo.py               # M10: DPO + 7B (ablation)
├── requirements.txt
└── README.md
```

---

## 📊 Model Progression

| Model | LLM | Algorithm | Accuracy | Avg Tokens |
|:-----:|:---:|:---------:|:--------:|:----------:|
| M1 | 1.5B | REINFORCE | 15% | 142 |
| M2 | 1.5B | Actor-Critic + soft reward | 30% | 142 |
| M3 | 1.5B | Warm-start + A2C | 30% | 103 |
| M4 | 1.5B | Warm-start + hard reward | 33% | 116 |
| M5 | 1.5B | Actor-Critic + oracle blend | 32% | 130 |
| M6 | 1.5B | PPO + highest reward oracle | 39% | 182 |
| M7 | 1.5B | PPO + min correct oracle | 34% | 167 |
| M8 | 1.5B | PPO + 446-dim features | 49% | 223 |
| **M9** ⭐ | **7B** | **PPO** | **80%** | **185** |
| M10 | 7B | DPO (ablation) | 67% | 142 |

---

## 🎯 Results: Model 9 vs Fixed Budgets

| Method | Avg Tokens | Accuracy |
|:------:|:----------:|:--------:|
| Fixed 96 | 96 | 47% |
| Fixed 128 | 128 | 63% |
| Fixed 160 | 160 | 74% |
| Fixed 192 | 192 | 76% |
| Fixed 224 | 224 | 79% |
| Fixed 256 | 256 | 84% |
| **Policy M9 (PPO)** ⭐ | **185** | **80%** |

The policy beats the fixed 192-token baseline (76%) while using fewer tokens, and matches fixed 224-token accuracy (79%) using **39 fewer tokens per question**.

---

## 🧬 Method

### Question Feature Vector (446 dimensions)

| Component | Dims | Description |
|-----------|:----:|-------------|
| SBERT Embedding | 384 | `all-MiniLM-L6-v2` sentence embedding |
| Handcrafted features | 50 | Token count, digit count, keyword flags |
| Difficulty features (M8+) | 12 | Op count, large numbers, conditional logic, unit conversion, composite score |
| **Total** | **446** | |

### Reward Function

```
R(b) = 𝟙[â = a*] + 0.25 · 𝟙[â = a*] − 0.0005 · b
```

Correct answers earn 1.25 minus a small token penalty. Wrong answers earn only the negative penalty.

### Oracle Label

```
b*(q) = min { b ∈ B : â(b) = a* }
```

The minimum budget at which the LLM answers correctly. If no correct budget exists, the highest-reward budget is used.

### Training Pipeline (Model 9)

```
Step 1 ── Oracle Construction
          Run 7B LLM at all budgets {128,144,160,176,192,210,220}
          Assign minimum correct budget per question

Step 2 ── Warm-Start (10 epochs)
          Supervised MSE regression on oracle labels

Step 3 ── Value Network Pre-Training (5 epochs)
          Pre-train on oracle rewards before PPO begins

Step 4 ── PPO Training (10 epochs)
          Rollout → Advantage → PPO clip [0.8, 1.2] → Oracle blend loss

Step 5 ── Inference
          Deterministic budget from policy mean → clamp → LLM → answer
```

---

## 💡 Key Design Decisions

### Why Earlier Models Failed

| Problem | Root Cause | Fix Applied |
|---------|-----------|-------------|
| Budget collapse (M1-M2) | REINFORCE/Actor-Critic gradients too noisy | PPO clip introduced in M6 |
| Stuck at 30% (M2) | Soft reward — partial credit masks failure | Hard binary reward in M4 |
| Warm-start doesn't help (M3) | Oracle labels: highest reward, not efficient | Minimum correct label in M5+ |
| Budget std = 3.8 (M4) | Entropy too low | Entropy coef raised 0.002 → 0.02 in M5 |
| PPO oscillates (M6-M7) | Value network starts from random weights | Value pre-training added in M8+ |
| 1.5B ceiling at 49% (M8) | Model capacity, not policy quality | Upgraded to 7B in M9 |

### Why DPO Failed (M10)

- Only **74/300** questions form valid preference pairs — 7B solves 94.7% of questions at every budget so most have no rejected budget
- Without a per-step reward signal, budget collapses from 142 → 106 tokens across 20 epochs
- PPO receives reward on **every question every rollout** — naturally prevents collapse

### Why M9 Works

All five fixes active simultaneously for the first time:

1. ✅ PPO clipped surrogate objective (Schulman et al., 2017)
2. ✅ Value network pre-training before PPO begins
3. ✅ Oracle warm-start with minimum correct budget labels
4. ✅ 446-dimensional difficulty-aware feature vector
5. ✅ Oracle blend loss keeps deterministic head learning during PPO

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/atharvasatishchaudhari/Difficulty-Aware-Reinforcement-Learning-for-Adaptive-Reasoning-Budget-Allocation-in-LLMs.git
cd Difficulty-Aware-Reinforcement-Learning-for-Adaptive-Reasoning-Budget-Allocation-in-LLMs

# Install PyTorch with CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install numpy transformers datasets sentence-transformers accelerate
```

### Requirements

```
torch>=2.11.0+cu128
transformers>=4.40.0
datasets>=2.18.0
sentence-transformers>=2.7.0
accelerate>=0.30.0
numpy>=1.26.0
```

---

## 🚀 Running the Models

### Run Model 9 (Best Model)

```bash
python model9_ppo_7b.py
```

This will:
1. Load GSM8K (300 train, 100 test questions)
2. Build oracle labels — cached after first run in `oracle_labels_v15.json`
3. Run warm-start for 10 epochs
4. Pre-train value network for 5 epochs
5. Run PPO for 10 epochs, saving best checkpoint
6. Evaluate final policy against all fixed budget baselines
7. Print final results comparison table

### Run Any Other Model

```bash
python model1_reinforce.py
python model2_actor_critic.py
python model8_rich_features.py
# etc.
```

### Cache Behavior

LLM inference results are cached to disk to avoid redundant GPU calls.

```
./cache_ppo_vXX/        ← LLM generation cache
oracle_labels_vXX.json  ← oracle budget labels
```

Set `clear_cache_on_start = True` in the `Config` dataclass to rerun from scratch (default: `True`).

---

## 🖥️ Hardware Requirements

| Component | Minimum | Used in This Work |
|-----------|:-------:|:-----------------:|
| GPU (1.5B models) | 8GB VRAM | NVIDIA H200 NVL 150GB |
| GPU (7B models) | 40-80GB VRAM | NVIDIA H200 NVL 150GB |
| CUDA | 11.8+ | 12.8 |
| RAM | 32GB+ | — |

> ⚠️ Models 9 and 10 use Qwen2.5-7B-Instruct in FP16 and require at least 40GB VRAM.

---

## 🔩 Configuration

All hyperparameters are set via a `Config` dataclass at the top of each script:

```python
@dataclass
class Config:
    # Models
    llm_name: str = "Qwen/Qwen2.5-7B-Instruct"
    embed_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_batch_size: int = 8

    # Oracle
    oracle_budgets: Tuple[int, ...] = (128, 144, 160, 176, 192, 210, 220)

    # Training
    warmstart_epochs: int = 10
    rl_epochs: int = 10
    ppo_epochs: int = 2
    ppo_epsilon: float = 0.2
    policy_learning_rate: float = 3e-4
    value_learning_rate: float = 8e-4
    entropy_coef: float = 0.02

    # Reward
    reward_token_penalty: float = 0.0005
    exact_match_bonus: float = 0.25

    # Budget range
    min_budget: int = 96
    max_budget: int = 256

    # Dataset
    train_subset: int = 300
    test_subset: int = 100
```

---

## 📚 Dataset

**GSM8K** — Cobbe et al., 2021 · [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)

- 7,473 training / 1,319 test questions
- Grade-school arithmetic word problems
- Loaded via Hugging Face `datasets`
- First N questions used in order for reproducibility

```python
from datasets import load_dataset
ds = load_dataset("gsm8k", "main")
```

---

## 📖 Citation

```bibtex
@misc{chaudhari2026adaptive,
  title   = {Difficulty-Aware Adaptive Reasoning Budget Allocation for LLMs},
  author  = {Chaudhari, Atharva},
  year    = {2026},
  note    = {IS 597 LLM Course Project, University of Illinois Urbana-Champaign},
  url     = {https://github.com/atharvasatishchaudhari/Difficulty-Aware-Reinforcement-Learning-for-Adaptive-Reasoning-Budget-Allocation-in-LLMs}
}
```

---

## 📝 References

- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in LLMs. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Aggarwal, P. & Welleck, S. (2025). L1: Controlling Reasoning Length with RL. [arXiv:2503.04697](https://arxiv.org/abs/2503.04697)
- Konda, V. & Tsitsiklis, J. (2000). Actor-Critic Algorithms. *NeurIPS 13*.

---

## 🤖 AI Disclosure

Claude (Anthropic) was used to assist with report writing and formatting. All experimental design, model implementation, training, results, and analysis are original work by the author.

---

<div align="center">

*University of Illinois Urbana-Champaign · IS 597 LLM · Spring 2026*

</div>
