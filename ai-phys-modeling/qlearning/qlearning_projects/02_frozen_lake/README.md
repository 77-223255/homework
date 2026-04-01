# Frozen Lake Q-Learning 🧊

Learn about **stochastic environments** and **sparse rewards** with a slippery frozen lake!

---

## 🎯 What You'll Learn

| Concept | How It's Used Here |
|---------|-------------------|
| **Stochastic transitions** | Actions don't always work (ice is slippery!) |
| **Sparse rewards** | Only get reward when reaching goal |
| **Exploration** | Must explore to find safe paths |
| **Grid world** | Classic RL environment type |

---

## 🧊 The Game

```
S F F F       S = Start (safe)
F H F H       F = Frozen (safe)
F F F H       H = Hole (fall in = game over!)
H F F G       G = Goal (+1 reward)
```

**Goal:** Cross the frozen lake without falling in holes!

---

## ⚠️ The Twist: Slippery Ice!

| Action | What You Want | What Might Happen |
|--------|---------------|-------------------|
| Move Up | Go up | 33% up, 33% left, 33% right |
| Move Down | Go down | 33% down, 33% left, 33% right |
| Move Left | Go left | 33% left, 33% up, 33% down |
| Move Right | Go right | 33% right, 33% up, 33% down |

**This is called a STOCHASTIC environment!**

**💡 Key Insight:** Even "bad" moves (like trying to walk into a wall) can be useful because you might slip in a good direction!

---

## 📊 Reward Structure:

| Situation | Reward | Meaning |
|-----------|--------|---------|
| 🏁 Reach goal | +1.0 | Win! |
| 💀 Fall in hole | -1.0 | Lose! |
| ⬜ Normal move | 0.0 | Continue playing |
| 🧱 Hit wall | 0.0 | Might slip elsewhere! |

---

## 🚀 How to Run

### Step 1: Train the Agent
```bash
cd /home/admin/Desktop/rl_projects/02_frozen_lake
conda activate qlearning
python train.py
```

### Step 2: Watch It Play
```bash
python play.py
```

---

## 🧠 Key Differences from Tic-Tac-Toe

| Feature | Tic-Tac-Toe | Frozen Lake |
|---------|-------------|-------------|
| **Transitions** | Deterministic | Stochastic (random) |
| **Rewards** | Frequent | Sparse (only at goal) |
| **State size** | ~5,000 states | 16 states |
| **Difficulty** | Medium | Easy states, hard randomness |

---

## 🎯 Expected Results

After training:
- Agent should reach goal ~70-75% of the time (not 100% due to slippery ice!)
- Agent learns safe paths that avoid holes
- Agent learns to recover from slips

---

## 💡 Experiments to Try

1. **Non-slippery mode** — Set `is_slippery=False`. Can agent reach 100%?
2. **Different maps** — Create harder/easier maps
3. **Discount factor** — What if γ = 0.5 (short-term thinking)?
4. **More holes** — Make the map harder

---

## 📊 Comparison: Deterministic vs Stochastic

### Deterministic (like Tic-Tac-Toe)
```
State → Action → Next State (ALWAYS same result)
```

### Stochastic (Frozen Lake)
```
State → Action → Next State (RANDOM result!)
```

**This is why Q-learning is powerful — it works even when outcomes are uncertain!**

---

*Created by Punk ⚡*