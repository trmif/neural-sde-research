import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_data(A_true, n_obs, n_steps, sigma_obs=0.1):
    d = A_true.shape[0]
    dt = 1.0 / n_steps
    x = torch.zeros(n_obs, d)

    with torch.no_grad():
        for _ in range(n_steps):
            drift = torch.sigmoid(x @ A_true.T)
            x = x + dt * drift + dt**0.5 * torch.randn_like(x)

    return x + sigma_obs * torch.randn_like(x)


def plot_mesh_sizes(results: dict[int, list[float]], title: str = ""):
    smooth = lambda x: np.convolve(x, np.ones(20) / 20, mode="valid")
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    plt.figure(figsize=(7, 5))
    for (n_steps, losses), color in zip(results.items(), colors): plt.plot(smooth(losses), color=color, linewidth=2, label=f"h=1/{n_steps}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title or "Effect of discretization step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("symlog")
    plt.tight_layout()
    plt.show()


def plot_sample_sizes(results: dict[int, list[float]], title: str = ""):
    smooth = lambda x: np.convolve(x, np.ones(20) / 20, mode="valid")
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(results)))

    plt.figure(figsize=(7, 5))
    for (n_obs, losses), color in zip(results.items(), colors): plt.plot(smooth(losses), color=color, linewidth=2, label=f"n={n_obs}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title or "Effect of sample size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("symlog")
    plt.tight_layout()
    plt.show()


def plot_comparison(results: dict[str, list[float]], title: str = ""):
    smooth = lambda x: np.convolve(x, np.ones(20) / 20, mode="valid")
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    plt.figure(figsize=(7, 5))
    for (label, losses), color in zip(results.items(), colors): plt.plot(smooth(losses), color=color, linewidth=2, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title or "Model comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("symlog")
    plt.tight_layout()
    plt.show()