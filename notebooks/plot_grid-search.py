import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


# ====== 配置 ======
OUTPUT_DIR = "/home/tree/RRL_cw2/output/HalfCheetah-v4/ppo"
SAVE_DIR = "plots/grid-search"
SMOOTHING = 0.9  # ⚠️ EMA建议用0~1


# ====== 创建保存目录 ======
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


# ====== 单个 run 画图并保存 ======
def plot_single_run(run_dir, smoothing=0.0):
    config_path = os.path.join(run_dir, "config.yaml")
    returns_path = os.path.join(run_dir, "returns.npy")

    if not os.path.exists(config_path) or not os.path.exists(returns_path):
        return None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    clip_coef = config.get("clip_coef")
    update_epochs = config.get("update_epochs")

    data = np.load(returns_path)
    steps = data[:, 0]
    returns = data[:, 1]

    returns_smooth = _ema_smooth(returns, smoothing)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, returns_smooth, label="run")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episodic Return")
    ax.set_title(f"PPO — clip-coef={clip_coef} update-epochs={update_epochs}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    # ====== 保存 ======
    filename = f"ppo_clip{clip_coef}_epoch{update_epochs}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300)

    plt.close()  # ⚠️ 很重要，防止内存爆

    print(f"Saved: {save_path}")

    return steps, returns_smooth, clip_coef, update_epochs


# ====== 主函数 ======
def main():
    all_runs = []

    for subdir in os.listdir(OUTPUT_DIR):
        run_dir = os.path.join(OUTPUT_DIR, subdir)
        if not os.path.isdir(run_dir):
            continue

        result = plot_single_run(run_dir, SMOOTHING)
        if result is not None:
            all_runs.append(result)

    # ====== 合并图 ======
    fig, ax = plt.subplots(figsize=(8, 5))

    # 可选：排序（更美观）
    all_runs.sort(key=lambda x: (x[2], x[3]))

    for steps, returns, clip_coef, update_epochs in all_runs:
        label = f"clip={clip_coef}, epoch={update_epochs}"
        ax.plot(steps, returns, label=label)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episodic Return")
    ax.set_title("PPO Comparison (All Runs)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    # ====== 保存总图 ======
    save_path = os.path.join(SAVE_DIR, "ppo_all_runs.png")
    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()