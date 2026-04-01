# Q-Learning Projects 🎮🧠

A collection of reinforcement learning projects for learning Q-learning from basics to neural networks.

---

## 📁 Project Structure

```
Desktop/
├── qlearning_original_project/    # Original maze navigation project
│   ├── RL_brain.py               # Q-learning agent implementation
│   ├── maze_env.py               # 4x4 maze environment with tkinter GUI
│   └── run_this.py               # Training script
│
├── qlearning_projects/            # Progressive learning collection
│   ├── 01_tic_tac_toe/           # ⭐ Beginner - Q-learning basics
│   ├── 02_frozen_lake/           # ⭐⭐ Beginner+ - Stochastic worlds
│   ├── 03_blackjack/             # ⭐⭐ Intermediate - Probability
│   └── 04_dqn_intro/             # ⭐⭐⭐ Intermediate - Neural networks
│
└── README.md                      # This file
```

---

## 🎯 Two Ways to Learn

### 1️⃣ Original Project (`qlearning_original_project/`)

A **visual maze navigation** project with tkinter GUI. Watch an agent learn to navigate a 4x4 maze in real-time!

| File | Purpose |
|------|---------|
| `RL_brain.py` | Q-learning agent with Q-table |
| `maze_env.py` | Maze environment with visual display |
| `run_this.py` | Training script (100 episodes) |

**Features:**
- 🎨 Visual GUI (tkinter)
- 🔴 Agent (red) navigates to 🟡 Goal (yellow)
- ⬛ Avoid hells (black obstacles)
- 📊 Real-time Q-table visualization

---

### 2️⃣ Progressive Projects (`qlearning_projects/`)

Four projects that progressively teach reinforcement learning concepts:

| # | Project | Concept | Difficulty |
|---|---------|---------|------------|
| 01 | Tic-Tac-Toe | Q-learning basics | ⭐ Beginner |
| 02 | Frozen Lake | Stochastic environments | ⭐⭐ Beginner+ |
| 03 | Blackjack | Probability & risk | ⭐⭐ Intermediate |
| 04 | DQN Intro | Neural networks | ⭐⭐⭐ Intermediate |

---

## 🚀 Quick Start

### Option A: Run the Original Maze Project

```bash
cd ~/Desktop/qlearning_original_project
conda activate qlearning
python run_this.py
```

**What happens:**
1. A window opens with a 4x4 maze
2. Agent (red square) explores randomly
3. Over 100 episodes, it learns the optimal path
4. Final Q-table shows learned values

---

### Option B: Follow the Progressive Path

```bash
# Step 1: Start with Tic-Tac-Toe (simplest)
cd ~/Desktop/qlearning_projects/01_tic_tac_toe
conda activate qlearning
python train.py    # Train the agent
python play.py     # Play against it

# Step 2: Move to Frozen Lake (stochastic)
cd ~/Desktop/qlearning_projects/02_frozen_lake
python train.py
python play.py

# Step 3: Try Blackjack (probability)
cd ~/Desktop/qlearning_projects/03_blackjack
python train.py
python play.py

# Step 4: Explore DQN (neural networks)
cd ~/Desktop/qlearning_projects/04_dqn_intro
pip install torch    # Need PyTorch for DQN
python train.py
```

---

## 🧠 Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│                    Q-LEARNING JOURNEY                        │
└─────────────────────────────────────────────────────────────┘

Original Project (Maze)           Progressive Projects
       │                                 │
       ▼                                 ▼
┌──────────────┐              ┌──────────────────┐
│   Q-Table    │              │  Tic-Tac-Toe     │ ⭐
│   Visual     │              │  (basics)        │
└──────────────┘              └────────┬─────────┘
       │                               │
       │                               ▼
       │                     ┌──────────────────┐
       │                     │  Frozen Lake     │ ⭐⭐
       │                     │  (stochastic)    │
       │                     └────────┬─────────┘
       │                              │
       │                              ▼
       │                     ┌──────────────────┐
       │                     │  Blackjack       │ ⭐⭐
       │                     │  (probability)   │
       │                     └────────┬─────────┘
       │                              │
       ▼                              ▼
┌──────────────┐              ┌──────────────────┐
│   Finite     │              │  DQN             │ ⭐⭐⭐
│   States     │              │  (neural nets)   │
└──────────────┘              └──────────────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │  Infinite States │
                             │  (continuous)    │
                             └──────────────────┘
```

---

## 📊 Project Comparison

| Feature | Original (Maze) | Tic-Tac-Toe | Frozen Lake | Blackjack | DQN |
|---------|-----------------|-------------|-------------|-----------|-----|
| **States** | ~20 | ~5,000 | 16 | ~280 | ∞ |
| **Actions** | 4 | 9 | 4 | 2 | 2 |
| **Visual** | ✅ tkinter | ❌ | ❌ | ❌ | ❌ |
| **Stochastic** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Neural Net** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Difficulty** | ⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 🎮 What You'll Learn

### From the Original Project:
- Q-table fundamentals
- State-action value functions
- ε-greedy exploration
- Visual reinforcement learning

### From Progressive Projects:

| Project | Key Concepts |
|---------|--------------|
| **Tic-Tac-Toe** | Q-learning basics, state representation, self-play |
| **Frozen Lake** | Stochastic transitions, sparse rewards, exploration |
| **Blackjack** | Probability, risk management, state aggregation |
| **DQN** | Neural networks, experience replay, target networks |

---

## 📦 Requirements

### Basic Requirements (All Projects):
```bash
pip install numpy pandas
```

### For Original Project (Maze):
```bash
# tkinter comes with Python on most systems
# If not: sudo apt-get install python3-tk (Ubuntu/Debian)
```

### For DQN Project:
```bash
pip install torch
# or with conda:
conda install pytorch
```

---

## 📈 Expected Results

### Original Maze Project:
- Training: 100 episodes
- Expected: Agent learns optimal path to goal
- Win rate: ~100% after training

### Progressive Projects:

| Project | Training Episodes | Expected Win Rate |
|---------|-------------------|-------------------|
| Tic-Tac-Toe | 10,000 | ~100% (never loses) |
| Frozen Lake | 10,000 | ~70-75% (slippery ice!) |
| Blackjack | 100,000 | ~42% (dealer advantage) |
| DQN | 500 | 195+ avg score (solved) |

---

## 🧪 Experiments to Try

### Beginner:
1. Change learning rate in any project
2. Train for more/less episodes
3. Modify reward values

### Intermediate:
1. Remove experience replay from DQN
2. Create custom Frozen Lake maps
3. Add card counting to Blackjack

### Advanced:
1. Implement Double DQN
2. Try different neural network architectures
3. Apply to new environments

---

## 📁 File Reference

### Original Project Files:

| File | Lines | Description |
|------|-------|-------------|
| `RL_brain.py` | ~200 | Q-learning agent with Q-table |
| `maze_env.py` | ~250 | Tkinter maze environment |
| `run_this.py` | ~100 | Training script |

### Progressive Project Files (per project):

| File | Description |
|------|-------------|
| `agent.py` | Q-learning or DQN agent |
| `train.py` | Training script |
| `play.py` | Test/play script |
| `game.py` / `*_env.py` | Environment/game logic |
| `README.md` | Project-specific docs |

---

## 🔗 Quick Links

| Resource | Path |
|----------|------|
| Original Project | `~/Desktop/qlearning_original_project/` |
| Tic-Tac-Toe | `~/Desktop/qlearning_projects/01_tic_tac_toe/` |
| Frozen Lake | `~/Desktop/qlearning_projects/02_frozen_lake/` |
| Blackjack | `~/Desktop/qlearning_projects/03_blackjack/` |
| DQN Intro | `~/Desktop/qlearning_projects/04_dqn_intro/` |

---

## 💡 Tips

1. **Start simple** — Run the maze project first to see visual learning
2. **Read the code** — Every important line is commented
3. **Experiment** — Change parameters and observe results
4. **Progress gradually** — Each project builds on previous concepts

---

## 🎯 Summary

| Project Type | Best For |
|--------------|----------|
| **Original (Maze)** | Visual learning, understanding Q-tables |
| **Progressive** | Step-by-step mastery of RL concepts |

**Recommendation:** Start with the visual maze project, then work through the progressive projects in order.

---

Happy Learning! 🚀

*Created by Punk ⚡*
