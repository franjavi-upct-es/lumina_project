import asyncio
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.agent.ppo_agent import PPOAgent, PPOConfig
from backend.cognition.agent.uncertainty_gate import UncertaintyGate
from backend.config.constants import ACTION_DIM, NEXUS_OUTPUT_DIM, TARGET_TICKERS
from backend.data_engine.storage.timescale import get_timescale_store


class MultiAssetTradingEnv:
    def __init__(self, ticker_data, initial_capital=0.0):
        self.ticker_data = ticker_data  # dict: ticker -> {prices: [], dates: []}
        self.tickers = sorted(list(ticker_data.keys()))
        self.initial_capital = initial_capital

        # Align all tickers to a master timeline
        all_dates = set()
        for t in self.tickers:
            all_dates.update(ticker_data[t]["dates"])
        self.timeline = sorted(list(all_dates))
        self.n_steps = len(self.timeline)

        # Pre-process data into aligned matrices
        self.price_matrix = np.zeros((self.n_steps, len(self.tickers)), dtype=np.float32)
        for i, date in enumerate(self.timeline):
            for j, ticker in enumerate(self.tickers):
                # Simple forward fill for missing data (e.g. before IPO)
                # In real scenario, we'd handle "not yet active"
                d_idx = ticker_data[ticker]["date_to_idx"].get(date)
                if d_idx is not None:
                    self.price_matrix[i, j] = ticker_data[ticker]["prices"][d_idx]
                else:
                    if i > 0:
                        self.price_matrix[i, j] = self.price_matrix[i - 1, j]
                    else:
                        self.price_matrix[i, j] = 0.0

        self.reset()

    def reset(self):
        self.t = 0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.positions = np.zeros(len(self.tickers), dtype=np.float32)  # fraction of current equity
        self.portfolio_values = np.zeros(len(self.tickers), dtype=np.float32)  # in dollars
        return self._get_obs_for_ticker(0)

    def _get_obs_for_ticker(self, ticker_idx):
        # Synthetic market state + bookkeeping
        # Bookkeeping: [pos, equity_norm, drawdown, progress]
        self.tickers[ticker_idx]
        price = self.price_matrix[self.t, ticker_idx]

        rng = np.random.default_rng(self.t + ticker_idx)
        market_state = rng.standard_normal(NEXUS_OUTPUT_DIM).astype(np.float32) * 0.1
        # Add price signal
        if price > 0:
            market_state[0] = np.log(price) * 0.1

        portfolio_state = np.array(
            [
                self.positions[ticker_idx],
                self.equity / self.initial_capital,
                1.0 - self.equity / self.peak_equity,
                self.t / self.n_steps,
            ],
            dtype=np.float32,
        )

        return np.concatenate([market_state, portfolio_state])

    def _get_obs_all(self):
        """Batched observation: (n_tickers, NEXUS_OUTPUT_DIM + 4).

        Preserves the per-ticker rng seeding (``self.t + ticker_idx``) so
        behavior matches the single-ticker path exactly.
        """
        n = len(self.tickers)
        prices = self.price_matrix[self.t]  # (n,)
        market_state = np.empty((n, NEXUS_OUTPUT_DIM), dtype=np.float32)
        for j in range(n):
            rng = np.random.default_rng(self.t + j)
            market_state[j] = rng.standard_normal(NEXUS_OUTPUT_DIM).astype(np.float32) * 0.1
        # Price signal in slot 0 (only where price > 0)
        mask = prices > 0
        market_state[mask, 0] = np.log(prices[mask]) * 0.1

        equity_norm = self.equity / self.initial_capital
        drawdown = 1.0 - self.equity / self.peak_equity
        progress = self.t / self.n_steps
        portfolio_state = np.stack(
            [
                self.positions,
                np.full(n, equity_norm, dtype=np.float32),
                np.full(n, drawdown, dtype=np.float32),
                np.full(n, progress, dtype=np.float32),
            ],
            axis=1,
        )
        return np.concatenate([market_state, portfolio_state], axis=1)

    def step(self, action_vector, ticker_idx):
        # Simplified execution for one ticker
        direction = np.clip(action_vector[0], -1, 1)
        np.clip(action_vector[1], -1, 1)
        size_factor = (np.clip(action_vector[2], -1, 1) + 1.0) * 0.5

        target_pos = direction * size_factor  # fraction of equity

        # Execute trade at current price
        # For simplicity in this mess: we rebalance to target_pos
        # Note: Total target_pos across all tickers should be <= 1.0 for no leverage
        # but here we just let it be.

        self.positions[ticker_idx] = target_pos

        # After going through all tickers, we will advance time.
        # This function only records the intent for the current ticker.
        return

    def advance_time(self):
        if self.t >= self.n_steps - 1:
            return True  # Done

        p_now = self.price_matrix[self.t]
        p_next = self.price_matrix[self.t + 1]

        # Calculate PnL based on positions
        total_pnl = 0
        for j in range(len(self.tickers)):
            if p_now[j] > 0 and p_next[j] > 0:
                ret = (p_next[j] - p_now[j]) / p_now[j]
                # PnL = equity_at_start_of_step * position_fraction * return
                total_pnl += self.equity * self.positions[j] * ret

        self.equity += total_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.t += 1

        # Check bankruptcy
        if self.equity < self.initial_capital * 0.1:
            return True
        return False


async def main():
    # 1. Load Data — sequential per ticker. The TimescaleDB query is a
    # heavy time_bucket aggregation over a 1-minute hypertable; firing
    # them in parallel saturates the DB and hits command_timeout.
    store = get_timescale_store()
    await store.connect()

    tickers = TARGET_TICKERS
    start_date = datetime(1928, 1, 1, tzinfo=UTC)
    end_date = datetime(2026, 1, 1, tzinfo=UTC)

    ticker_data = {}
    logger.info("Loading data for all tickers...")
    for t in tickers:
        df = await store.get_historical_window(t, start_date, end_date, freq="1d")
        if not df.is_empty():
            dates = df.select("time").to_numpy().squeeze().tolist()
            prices = df.select("close").to_numpy().squeeze().astype(np.float32)
            ticker_data[t] = {
                "prices": prices,
                "dates": dates,
                "date_to_idx": {d: i for i, d in enumerate(dates)},
            }
    await store.disconnect()

    # 2. Setup Env and Agent — auto-select GPU when available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    env = MultiAssetTradingEnv(ticker_data, initial_capital=100.0)
    policy = PolicyNetwork(state_dim=NEXUS_OUTPUT_DIM + 4, action_dim=ACTION_DIM)
    agent = PPOAgent(
        policy, UncertaintyGate(), config=PPOConfig(lr=1e-3, batch_size=512), device=device
    )

    # 3. Training Loop (Single long episode per "generation")
    n_generations = 20
    history_equity = []
    n_tickers = len(env.tickers)

    logger.info(f"Starting multi-asset training for {n_generations} generations...")

    for gen in range(n_generations):
        env.reset()
        done = False
        gen_equity = [env.equity]

        while not done:
            # Batched act: one forward pass for all tickers × MC samples
            obs_batch = env._get_obs_all()  # (n_tickers, D)
            actions, log_probs, values, uncertainties, _vetoed = agent.act_batch(obs_batch)

            for j in range(n_tickers):
                env.step(actions[j], j)
                agent.record(
                    obs_batch[j],
                    actions[j],
                    float(log_probs[j]),
                    float(values[j]),
                    0.0,
                    False,
                    float(uncertainties[j]),
                )

            # Now advance time and get real reward
            prev_equity = env.equity
            done = env.advance_time()
            reward = (env.equity - prev_equity) / env.initial_capital

            # Update rewards in the buffer for the steps just taken
            # (Last len(tickers) entries)
            for k in range(1, len(env.tickers) + 1):
                agent.buffer.rewards[-k] = reward
                if done:
                    agent.buffer.dones[-k] = True

            # advance_time returns done=True on the terminal step *without*
            # incrementing self.t — skip the append in that case to keep
            # gen_equity aligned with env.timeline (both end at length n_steps).
            if not done:
                gen_equity.append(env.equity)

            if len(agent.buffer) >= 2048:
                agent.update()

        history_equity.append(env.equity)
        logger.info(
            f"Gen {gen + 1}/{n_generations} | Final Equity: ${env.equity:.2f} | Steps: {env.t}"
        )

        if gen == n_generations - 1:
            # Save the final curve
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(env.timeline, gen_equity, label="Portfolio Equity")
            plt.yscale("log")
            plt.title(f"Century-Long Multi-Asset Portfolio ($100 to ${env.equity:.2f})")
            plt.ylabel("Equity ($) - Log Scale")
            plt.grid(True)

            plt.subplot(2, 1, 2)
            for j, t in enumerate(env.tickers):
                plt.plot(env.timeline, env.price_matrix[:, j], label=t, alpha=0.5)
            plt.yscale("log")
            plt.ylabel("Asset Prices - Log Scale")
            plt.legend(ncol=3, fontsize="small")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig("portfolio_mess_results.png")
            logger.success("Portfolio results saved to portfolio_mess_results.png")


if __name__ == "__main__":
    asyncio.run(main())
