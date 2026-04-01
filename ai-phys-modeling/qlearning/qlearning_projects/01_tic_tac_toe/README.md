# Tic-Tac-Toe Q-Learning 🎮

Learn Q-learning by teaching an AI to play Tic-Tac-Toe!

---

## 🎯 What You'll Learn

| Concept | How It's Used Here |
|---------|-------------------|
| **State representation** | Board positions as strings |
| **Reward design** | +1 for win, -1 for loss, 0 for draw |
| **Self-play** | Agent plays against itself |
| **Q-table** | Learns best move for each position |
| **Exploitation** | Uses learned knowledge to win |

---

## 🎮 The Game

```
 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8
```

- Agent is **X** (plays first)
- Opponent is **O**
- Win: Get 3 in a row

---

## 🚀 How to Run

### Step 1: Train the Agent
```bash
python train.py
```

### Step 2: Play Against It
```bash
python play.py
```

---

## 🧠 How Q-Learning Works Here

### State
The board is represented as a string:
```
X . .      "X.."
. O .  →   ".O."
. . X      "..X"
```

### Actions
Actions are positions 0-8 (where to place X)

### Rewards
| Outcome | Reward |
|---------|--------|
| Win | +1 |
| Lose | -1 |
| Draw | 0.5 (small reward for not losing) |
| Ongoing | 0 |

### Learning Process
1. Agent looks at current board (state)
2. Chooses a move (action) using ε-greedy
3. Observes result (reward + new state)
4. Updates Q-table: "That move was good/bad"
5. Repeats until it learns optimal strategy

---

## 📊 Training Progress

After training, you'll see:
- Win rate over time
- Q-table for key positions
- Best moves for each situation

---

## 🎯 Expected Results

After ~10,000 games:
- Agent should **never lose** (win or draw)
- Agent learns to:
  - Take center if available
  - Block opponent's winning moves
  - Create winning opportunities

---

## 💡 Experiments to Try

1. **Change learning rate** — What happens if α = 0.5 instead of 0.1?
2. **Change exploration** — What if ε = 0.5 (more random)?
3. **Train longer** — Does 100,000 games improve performance?
4. **Play first** — Can you beat the trained agent?

---

*Created by Punk ⚡*