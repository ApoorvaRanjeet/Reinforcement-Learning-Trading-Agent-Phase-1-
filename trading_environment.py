import torch
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class TradingEnvironment:
    def __init__(self, df, starting_cash=10000):
        self.df            = df
        self.starting_cash = starting_cash
        self.action_size   = 3   # buy, sell, hold
        self.state_size    = 4   # price_vs_ma20, price_change, volume_vs_avg, holding

    def reset(self):
        self.cash         = self.starting_cash
        self.shares       = 0
        self.current_step = 0

        row = self.df.iloc[0]

        state = torch.tensor([
            row['price_vs_ma20'],
            row['price_change'],
            row['volume_vs_avg'],
            0                        # holding = 0 at start
        ], dtype=torch.float, device=device)   # ← device added

        return state

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        old_portfolio = self.cash + self.shares * current_price

        # take action
        if action == 1:        # BUY
            if self.cash >= current_price:
                self.shares   += 1
                self.cash     -= current_price
        elif action == 2:      # SELL
            if self.shares > 0:
                self.shares   -= 1
                self.cash     += current_price
        # action 0 = HOLD → nothing changes

        # move to next day
        self.current_step += 1

        # calculate reward
        next_price    = self.df.iloc[self.current_step]['Close']
        new_portfolio = self.cash + self.shares * next_price
        reward = (new_portfolio - old_portfolio) / old_portfolio

        # check if episode over
        terminated = self.current_step >= len(self.df) - 1   # ← self.df

        # get new state
        row       = self.df.iloc[self.current_step]
        holding   = 1 if self.shares > 0 else 0
        new_state = torch.tensor([
            row['price_vs_ma20'],
            row['price_change'],
            row['volume_vs_avg'],
            holding
        ], dtype=torch.float, device=device)

        return new_state, reward, terminated


# ── test block ──────────────────────────────────────
if __name__ == '__main__':

    # load and prepare data here
    df = pd.read_csv(r'C:\Users\apoor\OneDrive\Documents\reinforcement_learning_q_learning\trading_agent\ANDHRAPAP.NS.csv')
    df['ma20']          = df['Close'].rolling(20).mean()
    df['price_vs_ma20'] = df['Close'] / df['ma20']
    df['price_change']  = df['Close'].pct_change().clip(-0.1, 0.1)
    df['avg_volume']    = df['Volume'].rolling(20).mean()
    df['volume_vs_avg'] = (df['Volume'] / df['avg_volume']).clip(0, 3)
    df = df.dropna().reset_index(drop=True)

    print(f"Total trading days: {len(df)}")

    # test environment
    env   = TradingEnvironment(df)
    state = env.reset()
    print(f"\nInitial state: {state}")
    print(f"State shape:   {state.shape}")

    # test one step each action
    print("\n--- Testing BUY ---")
    new_state, reward, terminated = env.step(1)
    print(f"New state:    {new_state}")
    print(f"Reward:       {reward:.2f}")
    print(f"Terminated:   {terminated}")
    print(f"Cash:         {env.cash:.2f}")
    print(f"Shares:       {env.shares}")

    print("\n--- Testing SELL ---")
    new_state, reward, terminated = env.step(2)
    print(f"New state:    {new_state}")
    print(f"Reward:       {reward:.2f}")
    print(f"Terminated:   {terminated}")
    print(f"Cash:         {env.cash:.2f}")
    print(f"Shares:       {env.shares}")

    print("\n--- Testing HOLD ---")
    new_state, reward, terminated = env.step(0)
    print(f"New state:    {new_state}")
    print(f"Reward:       {reward:.2f}")
    print(f"Terminated:   {terminated}")