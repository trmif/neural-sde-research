import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.utils import generate_data
from src.models.eulerSDE import EulerSDE, free_energy

st.title("Beyond Infinity: Neural SDEs in Practice")

page = st.sidebar.selectbox("Эксперимент", ["Synthetic", "Wind", "Diabetes"])

if page == "Synthetic":
    st.sidebar.header("Гиперпараметры")
    d = st.sidebar.slider("Размерность d", 2, 20, 10)
    n_obs = st.sidebar.slider("Число наблюдений", 1000, 10000, 5000)
    n_steps = st.sidebar.slider("Шагов Эйлера", 8, 64, 32)
    n_iters = st.sidebar.slider("Итераций", 100, 2000, 500)
    hidden = st.sidebar.slider("Hidden size", 32, 128, 64)
    lr = st.sidebar.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)

    if st.sidebar.button("Запустить обучение"):
        torch.manual_seed(42)
        A_true = torch.randn(d, d)
        Y_data = generate_data(A_true, n_obs=n_obs, n_steps=n_steps)

        class ModelDrift(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d + 1, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, d))

            def forward(self, t, x):
                t_emb = t.expand(x.shape[0], 1) if x.dim() > 1 else t.unsqueeze(0)
                return self.net(torch.cat([x, t_emb], dim=-1))

        class VariationalDrift(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d * 2 + 1, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, d))

            def forward(self, t, x, y, step=0):
                t_emb = t.expand(x.shape[0], 1) if x.dim() > 1 else t.unsqueeze(0)
                return self.net(torch.cat([x, y, t_emb], dim=-1))

        class Diffusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d + 1, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, d))

            def forward(self, t, x):
                t_emb = t.expand(x.shape[0], 1) if x.dim() > 1 else t.unsqueeze(0)
                return torch.exp(self.net(torch.cat([x, t_emb], dim=-1)))

        model = EulerSDE(
            d=d,
            model_drift=ModelDrift(),
            variational_drift=VariationalDrift(),
            diffusion=Diffusion())

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        progress = st.progress(0)
        loss_placeholder = st.empty()

        for i in range(n_iters):
            optimizer.zero_grad()
            idx = torch.randint(0, len(Y_data), (128,))
            y_batch = Y_data[idx]
            x0 = torch.zeros(128, d)
            x1, kl = model(x0, n_steps, y=y_batch)
            loss = free_energy(x1, kl, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            progress.progress((i + 1) / n_iters)
            if i % 50 == 0:
                loss_placeholder.metric("Loss", f"{loss.item():.4f}")

        st.subheader("Кривая обучения")
        smooth = np.convolve(losses, np.ones(20) / 20, mode='valid')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(losses, alpha=0.3, color='steelblue')
        ax.plot(smooth, color='steelblue', lw=2)
        ax.set_xlabel("Итерация")
        ax.set_ylabel("Free Energy")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Сэмплирование")
        model.eval()
        with torch.no_grad():
            idx = torch.randint(0, len(Y_data), (1000,))
            x0 = torch.zeros(1000, d)
            samples, _ = model(x0, n_steps, y=Y_data[idx])

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
        st.pyplot(fig)

elif page == "Wind":
    st.header("...")

elif page == "Diabetes":
    st.header("...")