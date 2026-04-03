import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


OUTPUT_DIR = "output/Ant-v4/sac"
# OUTPUT_DIR = "output/HalfCheetah-v4/sac"
# SAVE_DIR = "HalfCheetah_plots/sac/grid-search-grouped"
SAVE_DIR = "Ant_plots/sac/grid-search-grouped"
SMOOTHING = 0.9

os.makedirs(SAVE_DIR, exist_ok=True)


def _ema_smooth(values, weight):
    if weight == 0:
        return values
    smoothed, last = [], values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def load_all_runs():
    runs = []

    for subdir in os.listdir(OUTPUT_DIR):
        run_dir = os.path.join(OUTPUT_DIR, subdir)
        if not os.path.isdir(run_dir):
            continue

        config_path = os.path.join(run_dir, "config.yaml")
        returns_path = os.path.join(run_dir, "returns.npy")

        if not (os.path.exists(config_path) and os.path.exists(returns_path)):
            continue

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # tau = config.get("tau")
        # batch_size = config.get("batch_size")

        # data = np.load(returns_path)
        # steps = data[:, 0]
        # returns = _ema_smooth(data[:, 1], SMOOTHING)

        tau = config.get("tau")
        batch_size = config.get("batch_size")

        data = np.load(returns_path)
        steps = data[:, 0]
        returns = data[:, 1]

        mask = steps <= 400000
        steps = steps[mask]
        returns = returns[mask]

        if len(steps) == 0:
            return None
        
        returns = _ema_smooth(returns, SMOOTHING)

        # 2. interpolation
        dense_steps = np.linspace(steps.min(), steps.max(), 300)
        returns = np.interp(dense_steps, steps, returns)
        steps = dense_steps

        runs.append({
            "tau": tau,
            "batch_size": batch_size,
            "steps": steps,
            "returns": returns
        })

    return runs


def plot_group_by_tau(runs):
    grouped = defaultdict(list)

    for r in runs:
        grouped[r["tau"]].append(r)

    for tau, group in grouped.items():
        if len(group) < 2:
            continue 

        fig, ax = plt.subplots(figsize=(8, 5))

        for r in sorted(group, key=lambda x: x["batch_size"]):
            label = f"batch_size={r['batch_size']}"
            ax.plot(r["steps"], r["returns"], label=label)

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episodic Return")
        ax.set_title(f"SAC — tau={tau} (varying batch-size)")
        ax.set_ylim(-500, 4500)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()

        save_path = os.path.join(SAVE_DIR, f"group_tau_{tau}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")


def plot_group_by_batch_size(runs):
    grouped = defaultdict(list)

    for r in runs:
        grouped[r["batch_size"]].append(r)
    for batch_size, group in grouped.items():
        if len(group) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for r in sorted(group, key=lambda x: x["tau"]):
            label = f"tau={r['tau']}"
            ax.plot(r["steps"], r["returns"], label=label)

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episodic Return")
        ax.set_title(f"SAC — batch-size={batch_size} (varying tau)")
        ax.set_ylim(-500, 4500)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()

        save_path = os.path.join(SAVE_DIR, f"group_batch_size_{batch_size}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")


def main():
    runs = load_all_runs()

    runs.sort(key=lambda x: (x["tau"], x["batch_size"]))

    plot_group_by_tau(runs)
    plot_group_by_batch_size(runs)

if __name__ == "__main__":
    main()