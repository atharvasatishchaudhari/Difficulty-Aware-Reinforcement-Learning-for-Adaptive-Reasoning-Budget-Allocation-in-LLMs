# Difficulty-Aware-Reinforcement-Learning-for-Adaptive-Reasoning-Budget-Allocation-in-LLMs
Large language models use the same reasoning budget for every question, regardless of difficulty. Easy questions waste tokens. Hard questions may not get enough. This project trains a lightweight budget policy network that predicts how many reasoning tokens a frozen LLM should use per question, before reasoning begins.

🏆 Best result: 80% accuracy at 185 avg tokens (Model 9, PPO + Qwen2.5-7B)
outperforming the fixed 192-token baseline of 76% while using fewer tokens

🔧 How It Works
Question ──► Policy Network ──► Budget B ──► LLM Reasoning (B tokens) ──► Answer Extraction ──► Final Answer

The policy network (small MLP, ~170K params) predicts the token budget per question
The frozen LLM generates reasoning within that budget, then extracts the answer
Only the policy network is trained. The LLM never changes.

📁 Repository Structure
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


