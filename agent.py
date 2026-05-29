import torch
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from torch import nn
import yaml
from experience_replay import ReplayMemory
from dqn import DQN
from trading_environment import TradingEnvironment
import itertools
import os
from datetime import datetime, timedelta

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# ── load and prepare data ──────────────────────────
df = pd.read_csv(r'C:\Users\apoor\OneDrive\Documents\reinforcement_learning_q_learning\trading_agent\ANDHRAPAP.NS.csv')
df['ma20']          = df['Close'].rolling(20).mean()
df['price_vs_ma20'] = df['Close'] / df['ma20']
df['price_change']  = df['Close'].pct_change().clip(-0.1, 0.1)
df['avg_volume']    = df['Volume'].rolling(20).mean()
df['volume_vs_avg'] = (df['Volume'] / df['avg_volume']).clip(0, 3)
df = df.dropna().reset_index(drop=True)

print(f"Total trading days: {len(df)}")

class Agent:
    def __init__(self, df, hyperparameter_set):
        self.df = df

        with open('trading_agent\hyperparameter.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters         = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set  = hyperparameter_set
        self.learning_rate_a     = hyperparameters['learning_rate_a']
        self.discount_factor_g   = hyperparameters['discount_factor_g']
        self.network_sync_rate   = hyperparameters['network_sync_rate']
        self.replay_memory_size  = hyperparameters['replay_memory_size']
        self.mini_batch_size     = hyperparameters['mini_batch_size']
        self.epsilon_init        = hyperparameters['epsilon_init']
        self.epsilon_decay       = hyperparameters['epsilon_decay']
        self.epsilon_min         = hyperparameters['epsilon_min']
        self.fc1_nodes           = hyperparameters['fc1_nodes']

        # file paths
        RUNS_DIR = "runs"
        os.makedirs(RUNS_DIR, exist_ok=True)
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.png')
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')

        self.loss_fn  = nn.MSELoss()
        self.optimizer = None

    def run(self, is_training=True):
        env         = TradingEnvironment(self.df)    # ← create env here
        num_actions = env.action_size
        num_states  = env.state_size

        rewards_per_episode = []
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            start_time             = datetime.now()
            last_graph_update_time = start_time

            epsilon        = self.epsilon_init
            memory         = ReplayMemory(self.replay_memory_size)
            epsilon_history = []
            step_count     = 0
            best_reward    = -9999999

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=self.learning_rate_a
            )
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()


        episodes = 1000 if is_training else 1

        for episode in range(episodes):
            state          = env.reset()
            terminated     = False
            episode_reward = 0.0

            while not terminated:
                # epsilon greedy action selection
                if is_training and random.random() < epsilon:
                    action = random.randint(0, num_actions - 1)
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

                new_state, reward, terminated = env.step(action.item())

                episode_reward += reward

                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            # Add inside run() after rewards_per_episode.append():
            if is_training and episode % 100 == 0:
                final_portfolio = env.cash + env.shares * env.df.iloc[-1]['Close']
                print(f"Episode {episode} | "
                    f"Reward: {episode_reward:.1f} | "
                    f"Portfolio: ₹{final_portfolio:.2f} | "
                    f"Epsilon: {epsilon:.3f}")
            
            # save model if best reward
            if is_training:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    print(f"Episode {episode} | New best reward: {episode_reward:.1f} | Model saved")

                # save graph every 10 seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # optimize if enough memory
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

        # Add at the end of run() after training loop:
        initial_price  = self.df.iloc[0]['Close']
        final_price    = self.df.iloc[-1]['Close']
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100

        # calculate agent's final portfolio in last episode
        final_portfolio  = env.cash + env.shares * env.df.iloc[-1]['Close']
        agent_return     = ((final_portfolio - env.starting_cash) / env.starting_cash) * 100

        print(f"\n{'='*50}")
        print(f"Starting Capital:    ₹{env.starting_cash:,.2f}")
        print(f"Final Portfolio:     ₹{final_portfolio:,.2f}")
        print(f"Agent Return:        {agent_return:.1f}%")
        print(f"Buy & Hold Return:   {buy_hold_return:.1f}%")
        if is_training:
            print(f"Best Episode Reward: {best_reward:.1f}")
        print(f"{'='*50}") 
        
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states       = torch.stack(states)
        actions      = torch.stack(actions)
        new_states   = torch.stack(new_states)
        rewards      = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                       target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(
            dim=1,
            index=actions.unsqueeze(dim=1)
        ).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])

        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


if __name__ == '__main__':
    agent = Agent(df, 'trading_agent')
    agent.run(is_training=True)