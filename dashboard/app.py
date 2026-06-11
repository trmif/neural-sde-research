from pathlib import Path
import json

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

try:
    BASE = Path(__file__).parent
except NameError:
    BASE = Path.cwd()
CACHE = BASE / "cache_attempts"

st.set_page_config(page_title="Beyond Infinity: Neural SDEs in Practice", layout="wide", page_icon="🌀")

DOMAINS = {
    "Акции": "stocks",
    "Энергопотребление": "energy",
    "Глюкоза / инсулин": "glucose",
    "Ветрогенерация": "wind",
}
SYNTHETIC_TAB = "Синтетика (статья)"

NOTES = {
    "stocks": "Лог-доходности AAPL — почти белый шум: предсказуемого тренда мало, "
              "веер почти плоский. Честная картина того, где SDE не творит чудес.",
    "energy": "Дневное потребление с недельной/годовой сезонностью — нелинейная "
              "динамика, где Neural SDE обходит persistence.",
    "glucose": "CGM каждые 5 минут. Видно, как доверительный интервал расширяется с "
               "горизонтом — это и есть выученное распределение риска (гипо/гипер).",
    "wind": "Мощность ветрогенерации с будущим NWP-прогнозом ветра как экзогеном. "
            "Веер расширяется при росте неопределённости погоды.",
}

RED, BLACK, GREY = "#d62728", "#000000", "#7f7f7f"


@st.cache_data
def load(domain):
    z = np.load(CACHE / f"{domain}.npz")
    meta = json.load(open(CACHE / f"{domain}.json", encoding="utf-8"))
    return {k: z[k] for k in z.files}, meta


def forecast_chart(hist, fut, mean, lo, hi, K, H, units):
    cur = hist[-1]
    fstep = np.arange(0, H + 1)
    df_fc = pd.DataFrame({
        "шаг": fstep,
        "прогноз": np.concatenate([[cur], mean]),
        "lo": np.concatenate([[cur], lo]),
        "hi": np.concatenate([[cur], hi]),
        "факт_буд": np.concatenate([[cur], fut]),
    })
    df_h = pd.DataFrame({"шаг": np.arange(-K + 1, 1), "история": hist})

    band = alt.Chart(df_fc).mark_area(opacity=0.20, color=RED).encode(
        x=alt.X("шаг:Q", title="шаг (0 = «сейчас»)"),
        y=alt.Y("lo:Q", title=units), y2="hi:Q")
    l_hist = alt.Chart(df_h).mark_line(color=BLACK, strokeWidth=2).encode(x="шаг:Q", y="история:Q")
    l_fut = alt.Chart(df_fc).mark_line(color=BLACK, strokeDash=[5, 3], point=True).encode(
        x="шаг:Q", y="факт_буд:Q")
    l_mean = alt.Chart(df_fc).mark_line(color=RED, strokeWidth=2.5).encode(x="шаг:Q", y="прогноз:Q")
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GREY, strokeDash=[2, 2]).encode(x="x:Q")
    return (band + rule + l_hist + l_fut + l_mean).properties(height=420).interactive()


def dist_chart(samples, real, mean, units):
    df = pd.DataFrame({"v": samples})
    hist = alt.Chart(df).mark_bar(opacity=0.6, color=RED).encode(
        x=alt.X("v:Q", bin=alt.Bin(maxbins=30), title=f"прогноз в горизонте, {units}"),
        y=alt.Y("count()", title="сэмплов"))
    r_real = alt.Chart(pd.DataFrame({"v": [real]})).mark_rule(
        color=BLACK, strokeWidth=2).encode(x="v:Q")
    r_mean = alt.Chart(pd.DataFrame({"v": [mean]})).mark_rule(
        color=RED, strokeDash=[4, 2], strokeWidth=2).encode(x="v:Q")
    return (hist + r_mean + r_real).properties(height=300)


def render(domain):
    data, meta = load(domain)
    K, H = meta["K"], meta["H"]
    st.caption(NOTES[domain])

    skill = (meta["rmse_pers"] - meta["rmse_nsde"]) / meta["rmse_pers"] * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Точек в тесте", f"{meta['n_test']:,}")
    c2.metric("RMSE Neural SDE", f"{meta['rmse_nsde']:.4g}")
    c3.metric("RMSE persistence", f"{meta['rmse_pers']:.4g}")
    c4.metric("Выигрыш vs persistence", f"{skill:+.1f}%",
              delta_color="normal" if skill >= 0 else "inverse")

    A = data["hist"].shape[0]
    a = st.slider("Опорный момент прогноза", 1, A, 1, key=f"a_{domain}") - 1

    hist, fut = data["hist"][a], data["fut"][a]
    mean, lo, hi = data["mean"][a], data["lo"][a], data["hi"][a]
    samples = data["samples"][a]

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Вероятностный прогноз на {H} шагов вперёд**")
        st.altair_chart(forecast_chart(hist, fut, mean, lo, hi, K, H, meta["units"]),
                        use_container_width=True)
        st.caption(
            "Чёрная сплошная — история (факт), чёрная пунктир — реальное будущее, "
            "красная — средний прогноз Neural SDE, красная заливка — доверительный "
            "интервал (10–90% по MC-сэмплам). Видно, как неопределённость растёт с горизонтом."
        )
    with right:
        st.markdown(f"**Выученное распределение в точке +{H}**")
        real_H, mean_H = float(fut[-1]), float(mean[-1])
        st.altair_chart(dist_chart(samples, real_H, mean_H, meta["units"]),
                        use_container_width=True)
        q = float((samples <= real_H).mean()) * 100
        st.caption(
            f"Гистограмма {len(samples)} сэмплов Neural SDE в финальном горизонте. "
            f"Чёрная линия — реальное значение ({real_H:.3g}), красная пунктир — "
            f"средний прогноз ({mean_H:.3g}). Факт попал в **{q:.0f}-й перцентиль** "
            "предсказанного распределения."
        )


# ── Шапка ─────────────────────────────────────────────────────────────────
st.title("Применение Neural SDE")
st.markdown(
    "Neural SDE в прогнозировании будущего на реальных данных"
)

if not CACHE.exists() or not list(CACHE.glob("*.npz")):
    st.error("Нет данных в cache_attempts/. Сначала запусти `python prepare_attempts.py`.")
    st.stop()

tabs = st.tabs([SYNTHETIC_TAB] + list(DOMAINS.keys()))

with tabs[0]:
    st.header("Synthetic — воспроизводим эксперимент из статьи")
    st.markdown("Воспроизводим секцию 6 из статьи на синтетических данных")

    col1, col2, col3 = st.columns(3)
    with col1:
        s_d = st.slider("Размерность d", 2, 20, 10, key="s_d")
        s_n_obs = st.slider("Наблюдений", 1000, 10000, 5000, key="s_n_obs")
    with col2:
        s_n_steps = st.slider("Шагов Эйлера", 8, 64, 32, key="s_n_steps")
        s_n_iters = st.slider("Итераций", 100, 2000, 500, key="s_n_iters")
    with col3:
        s_hidden = st.slider("Hidden size", 32, 128, 64, key="s_hidden")
        s_lr = st.select_slider("LR", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, key="s_lr")

    if st.button("Запустить", key="s_run"):
        import torch
        import torch.nn as nn
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils import generate_data
        from src.models.eulerSDE import EulerSDE, free_energy

        torch.manual_seed(42)
        A_true = torch.randn(s_d, s_d)
        Y_data = generate_data(A_true, n_obs=s_n_obs, n_steps=s_n_steps)

        class ModelDrift(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(s_d + 1, s_hidden), nn.Tanh(),
                    nn.Linear(s_hidden, s_hidden), nn.Tanh(),
                    nn.Linear(s_hidden, s_d))

            def forward(self, t, x):
                t_emb = t.expand(x.shape[0], 1) if x.dim() > 1 else t.unsqueeze(0)
                return self.net(torch.cat([x, t_emb], dim=-1))

        class VariationalDrift(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(s_d * 2 + 1, s_hidden), nn.Tanh(),
                    nn.Linear(s_hidden, s_hidden), nn.Tanh(),
                    nn.Linear(s_hidden, s_d))

            def forward(self, t, x, y, step=0):
                t_emb = t.expand(x.shape[0], 1) if x.dim() > 1 else t.unsqueeze(0)
                return self.net(torch.cat([x, y, t_emb], dim=-1))

        class Diffusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(s_d + 1, s_hidden), nn.Tanh(),
                    nn.Linear(s_hidden, s_hidden), nn.Tanh(),
                    nn.Linear(s_hidden, s_d))

            def forward(self, t, x):
                t_emb = t.expand(x.shape[0], 1) if x.dim() > 1 else t.unsqueeze(0)
                return torch.exp(self.net(torch.cat([x, t_emb], dim=-1)))

        model = EulerSDE(
            d=s_d,
            model_drift=ModelDrift(),
            variational_drift=VariationalDrift(),
            diffusion=Diffusion())

        optimizer = torch.optim.Adam(model.parameters(), lr=s_lr)
        losses = []
        progress = st.progress(0)
        loss_placeholder = st.empty()

        for i in range(s_n_iters):
            optimizer.zero_grad()
            idx = torch.randint(0, len(Y_data), (128,))
            y_batch = Y_data[idx]
            x0 = torch.zeros(128, s_d)
            x1, kl = model(x0, s_n_steps, y=y_batch)
            loss = free_energy(x1, kl, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            progress.progress((i + 1) / s_n_iters)
            if i % 50 == 0:
                loss_placeholder.metric("Loss", f"{loss.item():.4f}")

        import matplotlib.pyplot as plt

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
            x0 = torch.zeros(1000, s_d)
            samples, _ = model(x0, s_n_steps, y=Y_data[idx])

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


for tab, (name, domain) in zip(tabs, DOMAINS.items()):
    with tab:
        st.header(name)
        render(domain)