# Blackjack Q-Learning 🃏

Learn Q-learning with a classic card game — understand **probability** and **risk/reward decisions**!

---

## 🎯 What You'll Learn

| Concept | How It's Used Here |
|---------|-------------------|
| **Probability** | Cards are random, agent learns odds |
| **Risk vs Reward** | Hit (risky) or Stand (safe)? |
| **State aggregation** | Infinite deck possibilities → finite states |
| **Decision making** | Optimal play under uncertainty |

---

## 🃏 The Game

### Rules (Simplified):
1. You and dealer each get 2 cards
2. Your goal: Get closer to 21 than dealer without going over
3. **Hit** = Get another card
4. **Stand** = Stop drawing
5. **Bust** = Go over 21 = You lose!
6. Dealer must hit until 17+

### Card Values:
| Card | Value |
|------|-------|
| 2-10 | Face value |
| J, Q, K | 10 |
| A (Ace) | 1 or 11 |

---

## 🧠 The Agent's Challenge

**State:** (Your total, Dealer's visible card, Do you have a usable Ace?)

**Example:** (15, 10, False) = You have 15, dealer shows 10, no Ace

**Actions:**
- 0 = Stand (stop drawing)
- 1 = Hit (draw another card)

**The Question:** Should I risk hitting and maybe bust, or stand and hope dealer busts?

---

## 🚀 How to Run

### Step 1: Train the Agent
```bash
cd /home/admin/Desktop/rl_projects/03_blackjack
conda activate qlearning
python train.py
```

### Step 2: Play Against It
```bash
python play.py
```

---

## 📊 What Makes This Different

| Feature | Tic-Tac-Toe | Frozen Lake | Blackjack |
|---------|-------------|-------------|-----------|
| **State space** | ~5,000 | 16 | ~280 |
| **Randomness** | None | Movement | Cards dealt |
| **Opponent** | Direct | Environment | Dealer |
| **Risk** | No risk | Slippery ice | **Bust = lose!** |
| **Optimal play** | 100% win | ~75% win | **~42% win** |

---

## 🎯 Expected Results

After training:
- Win rate: ~42-43%
- Draw rate: ~8-9%
- Loss rate: ~48-49%

**Wait, less than 50% win?** Yes! The dealer has an advantage in Blackjack.

---

## 💡 The Key Lesson: Risk Management

| Situation | Safe Play | Risky Play |
|-----------|-----------|------------|
| You have 12 | Stand (wait for dealer to bust) | Hit (might improve) |
| You have 16 | Stand (risky!) | Hit (very risky!) |
| You have 18 | Stand (usually best) | Hit (probably bust) |

**The agent learns when to take risks and when to play safe!**

---

## 📈 Learning Progression

1. **Early training:** Random decisions (50% win would be lucky!)
2. **Mid training:** Learns basic rules (don't bust, hope dealer busts)
3. **Late training:** Near-optimal strategy (~42% win rate)

---

## 🧪 Experiments to Try

1. **Count cards** — Track which cards have been played
2. **Different rules** — Change dealer threshold (hit until 18?)
3. **Betting** — Add betting decisions
4. **Card counting** — Can agent learn it?

---

## 📚 Beyond This Project

This is the foundation for:
- **Card counting systems**
- **Game theory**
- **Monte Carlo methods**
- **Deep Q-Learning for complex games**

---

*Created by Punk ⚡*