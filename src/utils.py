import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def free_energy(x1, kl, y_obs, sigma_obs=0.1):
    recon_term = 0.5 / (sigma_obs ** 2) * ((y_obs - x1) ** 2).sum(-1).mean()
    return kl + recon_term

def generate_data(A_true, n_obs, n_steps, sigma_obs=0.1):

    d = A_true.shape[0]
    dt = 1.0 / n_steps
    x = torch.zeros(n_obs, d)

    with torch.no_grad():
        for _ in range(n_steps):
            drift = torch.sigmoid(x @ A_true.T)
            x = x + dt * drift + dt**0.5 * torch.randn_like(x)

    return x + sigma_obs * torch.randn_like(x)



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


def plot_samples(Y_data, samples):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(Y_data[:, 0], Y_data[:, 1], alpha=0.3, s=10, label="Y_data")
    axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, label="samples")
    axes[0].set_title("X[0] vs X[1]")
    axes[0].legend()
    axes[1].hist(Y_data[:, 0].numpy(), bins=50, alpha=0.5, label="Y_data", density=True)
    axes[1].hist(samples[:, 0].numpy(), bins=50, alpha=0.5, label="samples", density=True)
    axes[1].set_title("Marginal X[0]")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def load_wind_data(path, hist_h, h):
    df = pd.read_csv(path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)
    df['ws100'] = np.sqrt(df.U100**2 + df.V100**2)
    df['ws10'] = np.sqrt(df.U10**2 + df.V10**2)
    hh = df.TIMESTAMP.dt.hour
    df['hs'] = np.sin(2 * np.pi * hh / 24)
    df['hc'] = np.cos(2 * np.pi * hh / 24)
    return df.dropna(subset=['TARGETVAR']).reset_index(drop=True)

def build_wind_windows(df, hist_h, h):
    P = df.TARGETVAR.values.astype('float32')
    w1 = df.ws100.values.astype('float32')
    w0 = df.ws10.values.astype('float32')
    hs = df.hs.values.astype('float32')
    hc = df.hc.values.astype('float32')
    CTX, WS, P0, Y = [], [], [], []
    for t in range(hist_h, len(df) - h):
        CTX.append(np.concatenate([P[t-hist_h:t], [hs[t], hc[t]]]))
        WS.append(np.stack([w1[t+1:t+1+h], w0[t+1:t+1+h]], axis=1))
        P0.append(P[t])
        Y.append(P[t+1:t+1+h])
    return (np.array(CTX, dtype='float32'), np.array(WS, dtype='float32'),
            np.array(P0, dtype='float32'), np.array(Y, dtype='float32'))