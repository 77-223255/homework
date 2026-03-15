# Deep Q-Networks (DQN) Introduction 🧠

Learn how **neural networks replace Q-tables** to handle complex environments!

---

## 🎯 What You'll Learn

| Concept | How It's Used Here |
|---------|-------------------|
| **Neural networks** | Approximate Q-values instead of storing them |
| **Experience replay** | Learn from past experiences multiple times |
| **Target network** | Stabilize training |
| **Continuous states** | Handle infinite state spaces |

---

## 🤔 Why Do We Need DQN?

### The Problem with Q-Tables:

| Game | States | Q-Table Size |
|------|--------|--------------|
| Frozen Lake | 16 | Tiny ✅ |
| Blackjack | 280 | Small ✅ |
| Tic-Tac-Toe | ~5,000 | Medium ✅ |
| **CartPole** | **INFINITE** | **Impossible! ❌** |

**CartPole states are continuous numbers:**
- Cart position: -4.8 to 4.8
- Cart velocity: -∞ to ∞
- Pole angle: -0.4 to 0.4 radians
- Pole velocity: -∞ to ∞

**You can't make a Q-table for infinite states!**

---

## 🧠 The DQN Solution

### Key Idea: Approximate Q-values with a Neural Network

```
Q-Table Approach:
  State → Look up in table → Q-values
  
DQN Approach:
  State → Neural Network → Q-values
```

**Instead of storing Q-values, we LEARN to predict them!**

---

## 🏗️ DQN Architecture

```
Input: State (4 numbers for CartPole)
    ↓
Hidden Layer 1 (24 neurons, ReLU)
    ↓
Hidden Layer 2 (24 neurons, ReLU)
    ↓
Output: Q-values for each action (2 actions)
```

---

## 🔄 Key Innovations

### 1. Experience Replay

```python
# Store experiences in memory
memory = [(state, action, reward, next_state, done), ...]

# Sample random batch to learn
batch = random.sample(memory, 32)
```

**Why?** Breaks correlation, reuses experiences!

---

### 2. Target Network

```python
# Two networks:
q_network       # Gets updated every step
target_network  # Gets updated every N steps

# Use target_network for calculating targets
target = reward + γ × target_network.predict(next_state)
```

**Why?** Prevents unstable training!

---

## 🎮 The Game: CartPole

```
    ┌─────────────────┐
    │       ╱│        │
    │      ╱ │        │  ← Pole
    │     ╱  │        │
    │    ╱   │        │
    │   ───────       │
    │   │ Cart │      │  ← Cart (move left/right)
    └───┴─────┴───────┘
        ←     →
```

**Goal:** Keep pole balanced for as long as possible!

| Observation | Range |
|-------------|-------|
| Cart Position | -4.8 to 4.8 |
| Cart Velocity | -∞ to ∞ |
| Pole Angle | -0.4 to 0.4 rad |
| Pole Velocity | -∞ to ∞ |

| Action | What |
|--------|------|
| 0 | Push cart left |
| 1 | Push cart right |

---

## 🚀 How to Run

### Step 1: Install PyTorch
```bash
pip install torch
```

### Step 2: Train the Agent
```bash
cd /home/admin/Desktop/rl_projects/04_dqn_intro
conda activate qlearning
python train.py
```

---

## 📊 Expected Results

| Training Progress | Score |
|-------------------|-------|
| Episode 1-100 | ~10-30 (random) |
| Episode 100-300 | ~50-100 (learning) |
| Episode 300-500 | ~150-200 (improving) |
| **Solved!** | **195+ average** |

---

## 🧠 What Makes DQN Special

| Feature | Q-Table | DQN |
|---------|---------|-----|
| **State space** | Finite only | **Infinite!** |
| **Memory** | Store all Q-values | Store weights only |
| **Generalization** | None | **Similar states → similar Q-values** |
| **Learning** | Direct update | **Gradient descent** |

---

## 📈 The Learning Process

```
Episode 1:    Score = 12   (random exploration)
Episode 50:   Score = 45   (starting to learn)
Episode 200:  Score = 120  (getting better)
Episode 400:  Score = 185  (almost solved!)
Episode 500:  Score = 210  (SOLVED! 🎉)
```

---

## 💡 Key Concepts to Understand

1. **Neural networks approximate functions** — they learn patterns
2. **Experience replay breaks correlation** — randomize training data
3. **Target networks stabilize learning** — don't chase a moving target
4. **Continuous states require function approximation** — can't enumerate everything

---

## 🧪 Experiments to Try

1. **Remove experience replay** — Watch training become unstable
2. **Remove target network** — See scores oscillate wildly
3. **Change network size** — Bigger isn't always better
4. **Different environments** — Try LunarLander, MountainCar

---

## 📚 What's Next?

After understanding DQN, you can learn:
- **Double DQN** — Reduce overestimation
- **Dueling DQN** — Better value estimation
- **Prioritized Experience Replay** — Learn from important experiences
- **A3C / PPO** — State-of-the-art algorithms

---

*Created by Punk ⚡*