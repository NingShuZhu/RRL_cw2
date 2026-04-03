import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


# OUTPUT_DIR = "output/Ant-v4/ppo"
OUTPUT_DIR = "output/HalfCheetah-v4/ppo"
# OUTPUT_DIR = "output/Ant-v4/sac"
# SAVE_DIR = "Ant_plots/sac/grid-search"
SAVE_DIR = "HalfCheetah_plots/ppo/grid-search"
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


def plot_single_run(run_dir):
    config_path = os.path.join(run_dir, "config.yaml")
    returns_path = os.path.join(run_dir, "returns.npy")

    if not os.path.exists(config_path) or not os.path.exists(returns_path):
        return None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    clip_coef = config.get("clip_coef")
    update_epochs = config.get("update_epochs")

    # tau = config.get("tau")
    # batch_size = config.get("batch_size")

    # data = np.load(returns_path)
    # steps = data[:, 0]
    # returns = data[:, 1]

    # returns_smooth = _ema_smooth(returns, smoothing)

    # dense_steps = np.linspace(steps.min(), steps.max(), 300)
    # returns = np.interp(dense_steps, steps, returns)
    # steps = dense_steps

    data = np.load(returns_path)
    steps = data[:, 0]
    returns = data[:, 1]

    # mask = steps <= 400000
    # steps = steps[mask]
    # returns = returns[mask]

    # if len(steps) == 0:
    #     return None

    # 1. smoothing
    returns = _ema_smooth(returns, SMOOTHING)

    # 2. interpolation
    dense_steps = np.linspace(steps.min(), steps.max(), 300)
    returns = np.interp(dense_steps, steps, returns)
    steps = dense_steps

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, returns, label="run")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episodic Return")
    ax.set_title(f"PPO — clip-coef={clip_coef} update-epochs={update_epochs}")
    # ax.set_title(f"SAC — tau={tau} batch_size={batch_size}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()


    filename = f"ppo_clip{clip_coef}_epoch{update_epochs}.png"
    # filename = f"sac_tau{tau}_batch{batch_size}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"Saved: {save_path}")

    return steps, returns, clip_coef, update_epochs
    # return steps, returns, tau, batch_size


def main():
    all_runs = []

    for subdir in os.listdir(OUTPUT_DIR):
        run_dir = os.path.join(OUTPUT_DIR, subdir)
        if not os.path.isdir(run_dir):
            continue

        result = plot_single_run(run_dir, SMOOTHING)
        if result is not None:
            all_runs.append(result)

    fig, ax = plt.subplots(figsize=(8, 5))

    all_runs.sort(key=lambda x: (x[2], x[3]))

    for steps, returns, clip_coef, update_epochs in all_runs:
        label = f"clip={clip_coef}, epoch={update_epochs}"
        ax.plot(steps, returns, label=label)

    # for steps, returns, tau, batch_size in all_runs:
    #     label = f"tau={tau}, batch={batch_size}"
    #     ax.plot(steps, returns, label=label)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episodic Return")
    ax.set_title("PPO Comparison (All Runs)")
    # ax.set_title("SAC Comparison (All Runs)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "ppo_all_runs.png")
    # save_path = os.path.join(SAVE_DIR, "sac_all_runs.png")
    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()