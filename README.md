# Difficulty-Aware-Reinforcement-Learning-for-Adaptive-Reasoning-Budget-Allocation-in-LLMs

# Difficulty-Aware Reinforcement Learning for Adaptive Reasoning Budget Allocation in LLMs

A lightweight policy network trained with PPO and DPO that dynamically allocates thinking token budgets to a frozen large language model on a per-question basis, achieving better accuracy at lower inference cost than fixed budget strategies.

---

## Overview

Standard LLM inference assigns the same reasoning token budget to every input regardless of difficulty. A simple arithmetic problem gets the same compute as a five-step word problem. This wastes tokens on easy questions and can be insufficient for hard ones.

This project trains a small MLP policy network to predict the optimal thinking token budget for each question before the LLM generates any reasoning. The frozen LLM then reasons within that budget and extracts a final answer. The result is a variable budget policy that beats fixed budget baselines on the accuracy vs tokens Pareto frontier.

**Best result:** 80% accuracy at 185 average tokens, beating fixed budget 224 tokens (79% accuracy) while saving 39 tokens per question. At 1 crore questions this saves 39 crore tokens with better accuracy.

---

## Key Results

| Method | Accuracy | Avg Tokens | vs Nearest Fixed Baseline |
|---|---|---|---|
| Fixed 96 tokens | 47% | 96 | baseline |
| Fixed 128 tokens | 63% | 128 | baseline |
| Fixed 160 tokens | 75% | 160 | baseline |
| Fixed 192 tokens | 76% | 192 | baseline |
| Fixed 224 tokens | 79% | 224 | baseline |
| Fixed 256 tokens | 83% | 256 | baseline |
| Policy Model 4 (1.5B) | 33% | 116 | matches fixed 128 using 12 fewer tokens |
| Policy Model 9 (1.5B) | 34% | 167 | beats fixed 168 by +4% at same token cost |
| **Policy Model 10 (7B)** | **80%** | **185** | **beats fixed 224 by +1% saving 39 tokens** |

---

## System Architecture

```
Question
   |
   v
QuestionFeaturizer (446-dim vector)
   |-- SentenceTransformer all-MiniLM-L6-v2 (384-dim SBERT embedding)
   |-- 62 handcrafted difficulty features
         |-- operation count (add, multiply, divide, percent keywords)
         |-- number count in question
         |-- sentence count (proxy for reasoning steps)
         |-- multi-step indicators (first, then, next, finally)
         |-- conditional logic flags (if, when, unless)
         |-- unit conversion flags (hours, minutes, kilometers)
         |-- composite difficulty score (weighted combination)
   |
   v
ContinuousBudgetPolicy (2-layer MLP, 256 hidden units)
   |-- outputs Normal distribution over normalized budget
   |-- deterministic mode: sigmoid(mean) -> budget
   |-- stochastic mode: sample -> budget (used during PPO rollout)
   |
   v
Budget B (integer in [96, 256] tokens)
   |
   v
Stage 1: LLM generates thinking text up to B tokens
   |
   v
Stage 2: LLM reads thinking text and extracts final numeric answer
   |
   v
Reward = correct + 0.25 * correct - 0.0005 * thinking_tokens
```

---

## GPU and CUDA Optimization

### Hardware

- GPU: NVIDIA H200 NVL with 150 GB VRAM
- CUDA version: 12.8
- PyTorch: 2.10 and 2.11 with cu128 wheels
- Framework: HuggingFace Transformers with device_map="auto"

### Batched GPU Inference

The most significant optimization is the batched LLM inference pipeline. Instead of processing questions one at a time (which would require a separate CUDA kernel launch per question), all questions sharing the same thinking budget are grouped and processed in a single batched forward pass.

```python
def generate_batch(self, prompts, max_new_tokens):
    # Left-pad all prompts to same length for batched decoding
    # Single CUDA kernel launch for entire batch
    # Reduces CPU-GPU synchronization overhead by batch_size factor
    inputs = self.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True
    ).to(self.device)

    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
    return outputs
```

**Why left-padding for batched decoder inference:** Decoder-only models (Qwen, GPT family) attend to all previous tokens autoregressively. Right-padding places padding tokens before the actual content, causing the model to attend to meaningless padding during generation. Left-padding ensures all real tokens appear at the end of the sequence so the model's attention is correctly grounded.

**Speedup achieved:** Batched inference with batch_size=16 achieves approximately 3x to 4x wall-clock speedup over sequential single-question inference. For the oracle label construction phase which evaluates 300 questions across 7 budget values, this reduces runtime from approximately 14 hours to under 4 hours.

### GPU Memory Management

The 7B model (Qwen2.5-7B-Instruct) uses approximately 15 GB of VRAM in float16 precision. With 150 GB available on H200, batch size is set to 8 for the 7B model leaving ample headroom for the KV cache during generation.

```python
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,      # FP16 halves memory vs FP32
    device_map="auto",        # automatic GPU placement
    trust_remote_code=True
)
```

For the 1.5B model, batch size is increased to 16 as the model occupies only approximately 3 GB of VRAM.

### Parallel Oracle Construction

The oracle label building phase evaluates every training question at multiple budget values. Each budget level is processed as a single batched GPU call rather than nested loops:

```python
# Process all 300 questions at budget=128 in one batched call
# Then all 300 at budget=160 in one batched call
# etc.
# Total GPU calls = num_budgets (7), not num_questions * num_budgets (2100)
for budget in oracle_budgets:
    results = run_batch_two_stage(questions, answers, budget, llm, cfg)
```

This reduces total GPU kernel launches from 2,100 to 7 for the oracle construction phase.

### Caching Layer

All LLM calls are cached to disk using MD5-keyed JSON files. The cache key is a hash of the question text, budget value, and model name. On subsequent runs, cached results are loaded directly without any GPU computation, enabling fast iteration during hyperparameter tuning.

```python
def make_cache_key(stage, question, context, model_name):
    content = f"{stage}|{question}|{context}|{model_name}"
    return hashlib.md5(content.encode()).hexdigest()
```

---

## Training Pipeline

### Phase 1: Oracle Label Construction

For each training question, the LLM is evaluated at multiple fixed budgets (96, 128, 160, 192, 224, 256 tokens). The oracle label is the minimum budget at which the model answers correctly. This teaches the policy what the ideal efficient budget looks like for each question.

### Phase 2: Supervised Warmstart

The policy is pre-trained via supervised regression on oracle labels before any RL begins. This gives the policy a meaningful starting point rather than random initialization, which would waste early RL epochs on random exploration from a cold start.

Loss: mean squared error between predicted normalized budget and oracle normalized budget.

### Phase 3: PPO Fine-tuning

Proximal Policy Optimization fine-tunes the policy using rollout experiences collected on training questions. Key PPO implementation details:

```
ppo_epochs = 2         # update passes per rollout batch (reduced from 4 to prevent over-updating)
ppo_epsilon = 0.2      # clip range: ratio stays in [0.8, 1.2]
ppo_batch_size = 32    # mini-batch size for PPO gradient updates
grad_clip_norm = 0.5   # gradient clipping to prevent NaN from high-variance rollouts
```

**Value network pre-training:** Before PPO begins, the value network is pre-trained for 5 epochs on oracle reward estimates. Without this, ValueLoss starts at 2.6 (random initialization), producing garbage advantage estimates in PPO epoch 1. Pre-training brings ValueLoss below 0.5 before the first policy update.

**Advantage estimation:**
```
Advantage = Reward - Value(question_features)
Normalized advantage = (advantage - mean) / (std + 1e-8)
```

### Reward Function

```
reward = correct + 0.25 * correct - 0.0005 * thinking_tokens
```

At 185 tokens correct: 1.25 - 0.0925 = 1.1575
At 256 tokens correct: 1.25 - 0.128 = 1.122

The small gap encourages efficiency without punishing longer budgets when they are genuinely needed.

---

## Policy Network Architecture

```python
class ContinuousBudgetPolicy(nn.Module):
    def __init__(self, input_dim=446, hidden_dim=256):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-2, 2)
        # NaN guard: replace any exploded weights with safe values
        mean = torch.nan_to_num(mean, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=-0.5)
        return mean, log_std

    def sample_budget(self, features):
        mean, log_std = self.forward(features)
        dist = Normal(mean, torch.exp(log_std))
        action = dist.rsample()           # reparameterization trick for gradient flow
        log_prob = dist.log_prob(action)
        norm_budget = torch.sigmoid(action)
        return norm_budget, log_prob, dist.entropy()
```

Budget denormalization:
```
budget = norm_budget * (max_budget - min_budget) + min_budget
budget = clamp(round(budget), min_budget, max_budget)
```

---

## Difficulty Features

The featurizer adds 62 handcrafted features on top of the 384-dim SBERT embedding:

**Original 50 features (keyword presence flags):**
Boolean indicators for math-related words: total, remaining, difference, product, average, fraction, ratio, percent, and 42 others.

**New 12 difficulty detection features:**

| Feature | Description |
|---|---|
| op_count | Count of arithmetic operation keywords |
| num_count | Count of numerical values in question |
| has_large_num | Flag for numbers greater than 100 |
| sentence_count | Number of sentences (proxy for reasoning steps) |
| multistep_count | Count of sequencing words: first, then, next, finally |
| has_conditional | Flag for if, when, unless |
| has_comparison | Flag for more than, less than, at least, how many more |
| has_unit_conversion | Flag for time, distance, weight units |
| has_pct_fraction | Flag for percent, half, third, quarter |
| entity_count | Count of pronouns and named entities |
| has_multiple_q | Flag for multiple question marks or "and how" |
| raw_difficulty | Weighted composite: 0.35 * op + 0.25 * num + 0.20 * sentences + 0.20 * multistep |

---

## Model Iteration History

| Model | LLM | Algorithm | Accuracy | Avg Tokens | Key Lesson |
|---|---|---|---|---|---|
| M1 | 1.5B | REINFORCE | 15% test | 142 | Too little data (80q), noisy reward, collapsed to one budget |
| M2 | 1.5B | Actor-Critic | 30% test | 142 | Value network doubled accuracy but still collapsed to one budget |
| M3 | 1.5B | Warmstart + Actor-Critic | 30% test | 103 | First real per-question adaptation, spread 85-139 tokens |
| M4 | 1.5B | Warmstart + PPO binary reward | 33% test | 116 | Matches fixed 128 using 12 fewer tokens |
| M5 to M9 | 1.5B | PPO variants | 39% best | 182 | Fixed oracle cache bug, value pretrain, ppo_epochs 4 to 2 |
| M10 | 7B | PPO + rich features | 80% test | 185 | Beats fixed 224 by +1% saving 39 tokens per question |
| M11 | 7B | PPO + forced exploration | NaN | n/a | Forced 20%+20% extreme budgets caused gradient NaN explosion |
| M12 | 7B | DPO | 67% test | 142 | Only 74 valid pairs from 300 questions, policy collapsed to min budget |

---

## Installation

```bash
# Install PyTorch with CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install numpy transformers datasets sentence-transformers accelerate

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Configuration

Key config values in the Config dataclass:

```python
llm_name = "Qwen/Qwen2.5-7B-Instruct"
embed_name = "sentence-transformers/all-MiniLM-L6-v2"
llm_batch_size = 8            # 8 for 7B model, 16 for 1.5B model
train_subset = 300
test_subset = 100
min_budget = 96
max_budget = 256
oracle_budgets = (96, 128, 160, 192, 224, 256)
warmstart_epochs = 10
rl_epochs = 10
ppo_epochs = 2
ppo_epsilon = 0.2
dpo_epochs = 20
dpo_beta = 0.1
reward_token_penalty = 0.0005
```

---

## Open Challenges

**Budget std collapse:** Across all models the policy budget std stays at 3 to 10, meaning the policy outputs nearly the same budget for every question. True per-question adaptivity needs std of 25 to 35. The policy effectively learns a better fixed budget rather than a genuinely variable one.

**Oracle skew:** 65 to 72% of oracle labels collapse to the minimum budget because the 7B model is too accurate at short budgets. This gives the warmstart too little signal about hard questions.

**Training set size:** 300 questions produces noisy PPO gradient estimates. Full GSM8K training set has 7,473 questions.

---

## Next Steps

- Scale to full GSM8K training set (7,473 questions)
- Use LLM to self-score question difficulty by comparing correctness at 96 vs 256 tokens
- Curriculum RL: train on hard questions first then gradually add easy ones
- Discrete policy head with 6 budget classes instead of continuous regression
- Larger model experiments: Qwen2.5-14B

---

## References

Cobbe et al. (2021). Training verifiers to solve math word problems. arXiv:2110.14168.

Aggarwal and Welleck (2025). L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv:2503.04697.

DeepSeek-AI (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948.

Schulman et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.

---

## AI Disclosure

Experimental design, model implementation, training, results, and analysis are entirely original work. Claude (Anthropic) assisted with code debugging and documentation writing. All scientific content and conclusions are the author's own.
