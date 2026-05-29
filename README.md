# Reinforcement Learning Trading Agent (Phase 1)

A Reinforcement Learning based Trading Agent built using Deep Q Learning (DQN) and PyTorch.

This project was created to understand how reinforcement learning works internally by building a custom trading environment from scratch instead of relying on prebuilt Gymnasium environments.

---

# Project Goal

The goal of this project is to train an AI agent that can learn trading behaviour using historical stock market data.

The agent learns to take one of three actions:

* Buy
* Sell
* Hold

based on market conditions and reward signals.

---

# What I Learned

This project helped me understand:

* How custom reinforcement learning environments work
* State, action, and reward design
* Why traditional Q-Learning does not scale well
* How Deep Q Networks (DQN) approximate Q-values using neural networks
* Replay memory and target networks
* Epsilon-greedy exploration
* Reward engineering and its effect on agent behaviour

---

# Features Implemented (Phase 1)

* Custom trading environment
* Buy / Sell / Hold actions
* Portfolio tracking
* Reward based on portfolio value change
* Replay memory
* Target network synchronization
* Deep Q Learning using PyTorch
* Model saving/loading
* Reward visualization graphs

---

# Tech Stack

* Python
* PyTorch
* Pandas
* NumPy
* Matplotlib

---

# State Space

The agent currently uses:

* Price vs Moving Average
* Daily Price Change
* Volume vs Average Volume
* Holding Status

as the state representation.

---

# Reward System

The reward is based on portfolio value improvement after taking an action.

```python
reward = new_portfolio - old_portfolio / old_portfolio
```

This encourages the agent to make decisions that improve total portfolio value over time.

---

# Results (Phase 1)

The trained agent was able to outperform a simple Buy & Hold strategy during a major downtrend by significantly reducing losses.

Example result:

* Agent Return: -6.2%
* Buy & Hold Return: -55.6%

---
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/b57b4f12-4951-466a-8728-c1620498c472" />

# Future Improvements (Phase 2)

Planned improvements:

* Add technical indicators (RSI, MACD, Bollinger Bands)
* Improve reward engineering
* Add train/test split
* Integrate financial news sentiment analysis
* Build interactive dashboard using Streamlit
* Experiment with advanced RL algorithms

---

# Project Structure

```bash
trading_agent/
│
├── agent.py
├── trading_environment.py
├── dqn.py
├── experience_replay.py
├── hyperparameter.yml
├── runs/
└── dataset.csv
```

---

# Inspiration & Learning Resources

This project was heavily inspired by:

* Sentdex Reinforcement Learning tutorials
* JohnnyCode DQN implementation videos

The concepts learned from those resources were adapted into a completely custom trading environment.

---

# Status

Phase 1 Complete 

