import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ====== 配置 ======
OUTPUT_DIR = "output/HalfCheetah-v4/ppo"
SAVE_DIR = "plot/grid-search-grouped"
SMOOTHING = 0.9

os.makedirs(SAVE_DIR, exist_ok=True)


# ====== EMA smoothing ======
def _ema_smooth(values, weight):
    if weight == 0:
        return values
    smoothed, last = [], values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


# ====== 读取所有 run ======
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

        clip = config.get("clip_coef")
        epoch = config.get("update_epochs")

        data = np.load(returns_path)
        steps = data[:, 0]
        returns = _ema_smooth(data[:, 1], SMOOTHING)

        runs.append({
            "clip": clip,
            "epoch": epoch,
            "steps": steps,
            "returns": returns
        })

    return runs


# ====== 分组1：固定 clip_coef ======
def plot_group_by_clip(runs):
    grouped = defaultdict(list)

    for r in runs:
        grouped[r["clip"]].append(r)

    for clip, group in grouped.items():
        if len(group) < 2:
            continue  # 至少2条线才有意义

        fig, ax = plt.subplots(figsize=(8, 5))

        for r in sorted(group, key=lambda x: x["epoch"]):
            label = f"epoch={r['epoch']}"
            ax.plot(r["steps"], r["returns"], label=label)

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episodic Return")
        ax.set_title(f"PPO — clip-coef={clip} (varying update-epochs)")
        ax.set_ylim(-500, 3100)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()

        save_path = os.path.join(SAVE_DIR, f"group_clip_{clip}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")


# ====== 分组2：固定 update_epochs ======
def plot_group_by_epoch(runs):
    grouped = defaultdict(list)

    for r in runs:
        grouped[r["epoch"]].append(r)

    for epoch, group in grouped.items():
        if len(group) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for r in sorted(group, key=lambda x: x["clip"]):
            label = f"clip={r['clip']}"
            ax.plot(r["steps"], r["returns"], label=label)

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episodic Return")
        ax.set_title(f"PPO — update-epochs={epoch} (varying clip-coef)")
        ax.set_ylim(-500, 3100)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()

        save_path = os.path.join(SAVE_DIR, f"group_epoch_{epoch}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")


# ====== 主函数 ======
def main():
    runs = load_all_runs()

    # 排序（让图更整齐）
    runs.sort(key=lambda x: (x["clip"], x["epoch"]))

    plot_group_by_clip(runs)
    plot_group_by_epoch(runs)


if __name__ == "__main__":
    main()