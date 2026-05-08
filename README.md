# Difficulty-Aware-Reinforcement-Learning-for-Adaptive-Reasoning-Budget-Allocation-in-LLMs

Large language models typically use the same reasoning budget for every question, regardless of difficulty. Easy questions waste tokens; hard questions may not get enough. This project trains a lightweight budget policy network that predicts the number of reasoning tokens a frozen LLM should use per question, before reasoning begins.
The system uses a two-stage inference pipeline:

The frozen LLM generates a reasoning chain under the predicted token budget
The same LLM reads the reasoning and produces a final numeric answer

A small trainable MLP (the policy network) decides the budget B for each question based on its features. Only the policy network is trained. The LLM remains completely frozen throughout.
The approach is evaluated on GSM8K across 10 model configurations spanning two LLM scales (1.5B and 7B), three training algorithms (REINFORCE, PPO, DPO), and progressively richer question features.
Best result: 80% accuracy at 185 avg tokens (Model 9, PPO + Qwen2.5-7B), outperforming the fixed 192-token baseline of 76% while using fewer tokens.

Repository Structure
.
├── model1_reinforce.py          # M1: REINFORCE, 1.5B, 80 train / 20 test
├── model2_actor_critic.py       # M2: Actor-Critic + soft reward, 200/50
├── model3_warmstart_a2c.py      # M3: Warm-start + A2C, 300/100
├── model4_hard_reward.py        # M4: Hard reward + exploration floor
├── model5_oracle_blend.py       # M5: Oracle blend + entropy fix
├── model6_ppo_highest.py        # M6: PPO + highest reward oracle
├── model7_ppo_mincorrect.py     # M7: PPO + minimum correct oracle
├── model8_rich_features.py      # M8: PPO + 446-dim features, 1.5B
├── model9_ppo_7b.py             # M9: PPO + Qwen2.5-7B (best model)
├── model10_dpo.py               # M10: DPO + 7B (ablation)
├── requirements.txt
└── README.md

Model Progression
ModelLLMAlgorithmAccuracyAvg TokensM11.5BREINFORCE15%142M21.5BActor-Critic + soft reward30%142M31.5BWarm-start + A2C30%103M41.5BWarm-start + hard reward33%116M51.5BActor-Critic + oracle blend32%130M61.5BPPO + highest reward oracle39%182M71.5BPPO + min correct oracle34%167M81.5BPPO + 446-dim features49%223M97BPPO (best)80%185M107BDPO (ablation)67%142

Method
Two-Stage Inference Pipeline
Question → Policy Network → Budget B → LLM Reasoning (B tokens) → LLM Answer Extraction → Final Answer
Policy Network

Two-layer MLP with 256 hidden units per layer
~170,000 trainable parameters
Input: 446-dimensional question feature vector
Output: Gaussian distribution over normalized budget → sampled during training, mean at inference
Budget clamped to [96, 256] tokens

Question Feature Vector (446 dimensions)
ComponentDimensionsDescriptionSBERT Embedding384all-MiniLM-L6-v2 sentence embeddingOriginal handcrafted50Token count, digit count, keyword flagsDifficulty features (M8+)12Operation count, large number flag, conditional logic, unit conversion, composite scoreTotal446
Reward Function
R(b) = 𝟙[â = a*] + 0.25 · 𝟙[â = a*] − 0.0005 · b
Correct answers earn 1.25 minus a small token penalty. Wrong answers earn only the negative token penalty.
Oracle Label Construction
For each training question, the LLM is run at every candidate budget in {128, 144, 160, 176, 192, 210, 220}. The oracle label is the minimum budget that produces a correct answer. If no budget is correct, the highest-reward budget is used.
Training Pipeline (Model 9)

Oracle construction: Run frozen 7B LLM at all candidate budgets on 300 training questions (batched GPU inference, 8 questions at a time)
Warm-start (10 epochs): Supervised MSE regression on oracle labels with reward-weighted and deviation-weighted loss
Value network pre-training (5 epochs): Pre-train value network on oracle rewards before PPO begins
PPO training (10 epochs):

Rollout: sample budgets from policy, run LLM, compute rewards and advantages
PPO clip update: ratio clamped to [0.8, 1.2]
Oracle blend loss: supervised MSE on deterministic head alongside PPO loss
Entropy bonus: entropy coefficient 0.02 to prevent budget collapse


Inference: deterministic budget from policy mean → clamp → LLM → answer


Key Design Decisions
What Fixed the Budget Collapse (M1-M5)

M1-M2: Budget collapsed to a single constant value because REINFORCE/Actor-Critic gradients are too noisy
M3: Oracle warm-start introduced — gives policy a strong starting point
M4: Hard binary reward replaces soft partial credit — cleaner gradient signal
M5: Entropy coefficient raised 0.002 → 0.02, oracle blend loss added
M6+: PPO clip ratio [0.8, 1.2] prevents oscillation entirely

Why DPO Failed (M10)

Only 74/300 questions form valid preference pairs (7B solves 94.7% of questions at every budget, so most have no rejected budget)
No per-step reward signal → budget collapses from 142 tokens to 106 tokens across 20 epochs
PPO receives reward signal on every question every rollout → naturally prevents collapse

Why M9 Works
All five fixes active simultaneously:

PPO clipped surrogate objective (Schulman et al., 2017)
Value network pre-training before PPO begins
Oracle warm-start with minimum correct budget labels
446-dimensional difficulty-aware feature vector
Oracle blend loss keeps deterministic head learning during PPO


Results: Model 9 vs Fixed Budgets
MethodAvg TokensAccuracyFixed 969647%Fixed 12812863%Fixed 16016074%Fixed 19219276%Fixed 22422479%Fixed 25625684%Policy M9 (PPO)18580%
The policy beats fixed 192-token baseline (76%) while using fewer tokens, and matches fixed 224-token accuracy (79%) using 39 fewer tokens per question.

Installation
bash# Clone the repository
git clone https://github.com/YOUR_USERNAME/adaptive-reasoning-budget.git
cd adaptive-reasoning-budget

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy transformers datasets sentence-transformers accelerate
Requirements
torch>=2.11.0+cu128
transformers>=4.40.0
datasets>=2.18.0
sentence-transformers>=2.7.0
accelerate>=0.30.0
numpy>=1.26.0

Running the Models
Run Model 9 (Best Model)
bashpython model9_ppo_7b.py
This will:

Load GSM8K (300 train, 100 test questions)
Build oracle labels (cached after first run)
Run warm-start for 10 epochs
Pre-train value network for 5 epochs
Run PPO for 10 epochs
Evaluate final policy against fixed budget baselines
Print final results table

Run Any Other Model
bashpython model1_reinforce.py
python model2_actor_critic.py
# ... etc
Cache Behavior
Each model caches LLM inference results to disk to avoid redundant GPU calls. Cache is stored in ./cache_ppo_vXX/ and oracle labels in oracle_labels_vXX.json. To clear cache and rerun from scratch, set clear_cache_on_start: bool = True in the Config dataclass (default is True).

Hardware Requirements
ComponentMinimumUsed in This WorkGPU24GB VRAM (for 1.5B)NVIDIA H200 NVL 150GBGPU80GB VRAM (for 7B)NVIDIA H200 NVL 150GBCUDA11.8+12.8RAM32GB—

Note: The 7B model (Models 9, 10) requires at least 40-80GB VRAM in FP16. The 1.5B model (Models 1-8) runs on smaller GPUs with ~8GB VRAM.


Configuration
All hyperparameters are controlled via a Config dataclass at the top of each script. Key parameters for Model 9:
python@dataclass
class Config:
    llm_name: str = "Qwen/Qwen2.5-7B-Instruct"
    embed_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_batch_size: int = 8
    oracle_budgets: Tuple[int, ...] = (128, 144, 160, 176, 192, 210, 220)
    warmstart_epochs: int = 10
    rl_epochs: int = 10
    ppo_epochs: int = 2
    ppo_epsilon: float = 0.2
    policy_learning_rate: float = 3e-4
    value_learning_rate: float = 8e-4
    entropy_coef: float = 0.02
    reward_token_penalty: float = 0.0005
    exact_match_bonus: float = 0.25
    min_budget: int = 96
    max_budget: int = 256
    train_subset: int = 300
    test_subset: int = 100

Dataset
GSM8K (Grade School Math 8K) — Cobbe et al., 2021

7,473 training questions, 1,319 test questions
Grade-school arithmetic word problems
Loaded via Hugging Face datasets library
First N questions used in order (not random sampling) to ensure reproducibility

pythonfrom datasets import load_dataset
ds = load_dataset("gsm8k", "main")


}
