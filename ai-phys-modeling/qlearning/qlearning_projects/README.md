# qlearning_projects

渐进式 Q-Learning 学习项目。

## 学习路径

| 序号 | 项目 | 核心概念 |
|------|------|----------|
| [`01_tic_tac_toe`](./01_tic_tac_toe/) | Q-Table 基础 |
| [`02_frozen_lake`](./02_frozen_lake/) | 随机环境 |
| [`03_blackjack`](./03_blackjack/) | 概率与风险 |
| [`04_dqn_intro`](./04_dqn_intro/) | 深度 Q 网络 |

## 通用结构

每个项目包含：
- `agent.py` — Q-Learning / DQN 智能体
- `train.py` — 训练脚本
- `play.py` — 测试/对弈脚本
- `game.py` / `*_env.py` — 环境逻辑

## 运行

```bash
cd <project_dir>
python train.py
python play.py
```
