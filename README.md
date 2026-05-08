# Difficulty-Aware-Reinforcement-Learning-for-Adaptive-Reasoning-Budget-Allocation-in-LLMs
Large language models use the same reasoning budget for every question, regardless of difficulty. Easy questions waste tokens. Hard questions may not get enough. This project trains a lightweight budget policy network that predicts how many reasoning tokens a frozen LLM should use per question, before reasoning begins.

🏆 Best result: 80% accuracy at 185 avg tokens (Model 9, PPO + Qwen2.5-7B)
outperforming the fixed 192-token baseline of 76% while using fewer tokens

