"""
Единый офлайн-пайплайн для дашборда «Наши попытки применить Neural SDE».

Одна модель (controlled Neural SDE: дрейф-сеть + Эйлер–Маруяма + KL) обучается на
4 доменах в одинаковой one-step постановке. Затем для нескольких опорных моментов
делается МНОГОШАГОВЫЙ вероятностный прогноз: SDE прогоняется на H шагов вперёд
M раз (Монте-Карло) → веер траекторий. Это и даёт «выученное распределение».

Для каждого домена в cache_attempts/<domain>.npz сохраняется ОДИНАКОВЫЙ формат:
    hist     (A, K)      историческая траектория (факт)
    fut      (A, H)      реальное будущее (факт)
    mean     (A, H)      средний прогноз
    lo, hi   (A, H)      доверительный интервал (10–90% по сэмплам)
    samples  (A, M)      распределение прогноза в финальном горизонте
+ метрики (one-step) в cache_attempts/<domain>.json.

Запуск: python prepare_attempts.py   (один домен: --only wind)
"""
import argparse
import json
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    BASE = Path(__file__).parent
except NameError:
    BASE = Path.cwd()
OUT = BASE / "cache_attempts"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_STEPS, DT = 64, 1.0 / 64
M_PRED = 200        # сэмплов Монте-Карло для распределения
N_ANCHORS = 12      # опорных моментов прогноза
TRAIN_RATIO = 0.8


# ── Общая модель (controlled Neural SDE) ──────────────────────────────────
class ConditionalDrift1D(nn.Module):
    def __init__(self, ctx_dim, hidden=96, activation="tanh"):
        super().__init__()
        act = nn.Tanh if activation == "tanh" else nn.SiLU
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + 1, hidden), act(),
            nn.Linear(hidden, hidden), act(),
            nn.Linear(hidden, hidden // 2), act(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, ctx):
        return self.net(torch.cat([x, ctx], dim=-1))


def simulate(net, ctx, n_steps=N_STEPS, dt=DT):
    B = ctx.shape[0]
    X = torch.zeros(B, 1, device=ctx.device)
    kl = torch.zeros(B, device=ctx.device)
    sqrt_dt = dt ** 0.5
    for _ in range(n_steps):
        drift = net(X, ctx)
        kl = kl + 0.5 * (drift ** 2).squeeze(-1) * dt
        X = X + drift * dt + sqrt_dt * torch.randn_like(X)
    return X.squeeze(-1), kl


def train_nsde(Xtr, ytr, hidden=96, activation="tanh", n_iter=1200, lr=1e-3,
               batch=128, obs_std=0.10, beta_max=0.03):
    net = ConditionalDrift1D(Xtr.shape[1], hidden, activation).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr * 0.05)
    for it in range(1, n_iter + 1):
        idx = np.random.randint(0, len(Xtr), batch)
        xb = torch.tensor(Xtr[idx], dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(ytr[idx], dtype=torch.float32, device=DEVICE)
        beta = min(beta_max, max(0.0, (it - n_iter * 0.2) / (n_iter * 0.4) * beta_max))
        x_t, kl = simulate(net, xb)
        loss = -((-0.5 / obs_std ** 2 * (x_t - yb) ** 2) - beta * kl).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
    return net


def predict_onestep(net, Xte, ysc, m=64):
    """One-step предсказание (для метрик): среднее по m сэмплам."""
    net.eval()
    ctx = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    preds = []
    with torch.no_grad():
        for _ in range(m):
            x_t, _ = simulate(net, ctx)
            preds.append(x_t.cpu().numpy())
    return ysc.inverse_transform(np.stack(preds).mean(0).reshape(-1, 1)).ravel()


def rollout(net, xsc, ysc, build_ctx, raw_series, anchors, K, H, m=M_PRED):
    """Многошаговый MC-прогноз. Возвращает samples (A, H, m) в raw-шкале target."""
    net.eval()
    A = len(anchors)
    out = np.zeros((A, H, m), dtype=np.float32)
    with torch.no_grad():
        for ai, t in enumerate(anchors):
            win = np.tile(raw_series[t - K + 1:t + 1].astype(np.float32), (m, 1))  # (m, K)
            for h in range(1, H + 1):
                ctx_raw = build_ctx(win, t, h)                      # (m, ctx_dim)
                ctx_s = xsc.transform(ctx_raw).astype(np.float32)
                x_t, _ = simulate(net, torch.tensor(ctx_s, device=DEVICE))
                nxt = ysc.inverse_transform(x_t.cpu().numpy().reshape(-1, 1)).ravel()
                out[ai, h - 1, :] = nxt
                win = np.concatenate([win[:, 1:], nxt.reshape(-1, 1)], axis=1)
    return out


def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def split_scale(X, y):
    n = int(len(X) * TRAIN_RATIO)
    xsc, ysc = StandardScaler(), StandardScaler()
    Xtr = xsc.fit_transform(X[:n]); Xte = xsc.transform(X[n:])
    ytr = ysc.fit_transform(y[:n].reshape(-1, 1)).ravel()
    return Xtr, Xte, ytr, y[n:], xsc, ysc, n


def finalize(domain, label, units, net, xsc, ysc, Xte, yte, pers_te,
             build_ctx, raw_series, to_units, K, H, anchor_lo, anchor_hi):
    # --- метрики one-step ---
    pred = predict_onestep(net, Xte, ysc)
    pred_u, true_u, pers_u = to_units(pred), to_units(yte), to_units(pers_te)
    r_nsde, r_pers = rmse(true_u, pred_u), rmse(true_u, pers_u)

    # --- многошаговый вероятностный прогноз от A опорных точек ---
    anchors = np.linspace(anchor_lo, anchor_hi, N_ANCHORS).astype(int)
    samp = rollout(net, xsc, ysc, build_ctx, raw_series, anchors, K, H)  # (A,H,m) raw
    samp_u = to_units(samp)
    hist = np.stack([to_units(raw_series[t - K + 1:t + 1]) for t in anchors])      # (A,K)
    fut = np.stack([to_units(raw_series[t + 1:t + 1 + H]) for t in anchors])       # (A,H)
    mean = samp_u.mean(2); lo = np.quantile(samp_u, 0.1, 2); hi = np.quantile(samp_u, 0.9, 2)
    final_samples = samp_u[:, -1, :]                                               # (A,m)

    np.savez_compressed(
        OUT / f"{domain}.npz",
        hist=hist.astype(np.float32), fut=fut.astype(np.float32),
        mean=mean.astype(np.float32), lo=lo.astype(np.float32), hi=hi.astype(np.float32),
        samples=final_samples.astype(np.float32),
    )
    with open(OUT / f"{domain}.json", "w") as f:
        json.dump({
            "label": label, "units": units, "K": int(K), "H": int(H),
            "n_anchors": int(N_ANCHORS), "n_test": int(len(yte)),
            "rmse_nsde": r_nsde, "rmse_pers": r_pers,
        }, f, ensure_ascii=False, indent=2)
    print(f"[{domain}] готово — anchors={N_ANCHORS}, K={K}, H={H}, "
          f"RMSE NSDE={r_nsde:.4g} vs persistence={r_pers:.4g}")


# ── 1. Акции AAPL ─────────────────────────────────────────────────────────
def prepare_stocks():
    import yfinance as yf

    def dl(t, tries=4):
        for k in range(tries):
            d = yf.download(t, start="2018-01-01", end="2024-12-31",
                            auto_adjust=True, progress=False)
            if d is not None and not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                return d
            time.sleep(3 * (k + 1))
        raise RuntimeError(f"Yahoo не отдал {t}")

    print("[stocks] загружаю с Yahoo...")
    W = 20
    stock, qqq, vix = dl("AAPL"), dl("QQQ"), dl("^VIX")
    df = pd.DataFrame(index=stock.index)
    df["close"], df["volume"] = stock["Close"], stock["Volume"]
    df["qqq"], df["vix"] = qqq["Close"], vix["Close"]
    df = df.dropna()
    df["r"] = np.log(df["close"] / df["close"].shift(1))
    df["qqq_r"] = np.log(df["qqq"] / df["qqq"].shift(1))
    df["vix_r"] = np.log(df["vix"] / df["vix"].shift(1))
    df["vol_r"] = np.log(df["volume"] / df["volume"].shift(1))
    df["roll_std"] = df["r"].rolling(W).std()
    df["roll_mean"] = df["r"].rolling(W).mean()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    r = df["r"].values.astype(np.float32)
    qr, vr, vol = df["qqq_r"].values, df["vix_r"].values, df["vol_r"].values
    X, y, pers = [], [], []
    for i in range(len(df) - W - 1):
        j = i + W - 1
        extra = [r[i:i + W].std(), r[i:i + W].mean(), qr[j], vr[j], vol[j]]
        X.append(np.concatenate([r[i:i + W], extra]).astype(np.float32))
        y.append(r[i + W]); pers.append(r[j])
    X, y, pers = np.array(X, np.float32), np.array(y, np.float32), np.array(pers, np.float32)
    Xtr, Xte, ytr, yte, xsc, ysc, n = split_scale(X, y)
    net = train_nsde(Xtr, ytr, hidden=96, activation="tanh", obs_std=0.10, beta_max=0.02)

    # rollout по абсолютному индексу ряда r; экзогены qr/vr/vol замораживаем у якоря
    def build_ctx(win, t, h):
        m = win.shape[0]
        extra = np.stack([win.std(1), win.mean(1),
                          np.full(m, qr[t]), np.full(m, vr[t]), np.full(m, vol[t])], 1)
        return np.concatenate([win, extra], 1).astype(np.float32)

    lo_a, hi_a = n + W, len(r) - W - 2
    finalize("stocks", "Акции Apple (AAPL)", "лог-доходность",
             net, xsc, ysc, Xte, yte, pers[n:], build_ctx, r, lambda a: a,
             K=W, H=10, anchor_lo=lo_a, anchor_hi=hi_a)


# ── 2. Энергопотребление ──────────────────────────────────────────────────
def prepare_energy():
    elec_dir = next((p for p in [BASE / "electric-data", BASE.parent / "electric-data"]
                     if p.exists()), None)
    txt = sorted(elec_dir.glob("household_power_consumption*.txt"))[0]
    print(f"[energy] читаю {txt.name}...")
    raw = pd.read_csv(txt, sep=";", na_values="?", low_memory=False)
    raw["dt"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    raw["Global_active_power"] = pd.to_numeric(raw["Global_active_power"], errors="coerce")
    raw = raw.dropna(subset=["Global_active_power"]).sort_values("dt")
    daily = raw.set_index("dt")["Global_active_power"].resample("D").sum()
    series = daily[daily > 10].values.astype(np.float32)

    W = 14
    log_s = np.log1p(series).astype(np.float32)

    def feats(window, di):
        return [np.sin(2 * np.pi * (di % 7) / 7), np.cos(2 * np.pi * (di % 7) / 7),
                np.sin(2 * np.pi * ((di // 30) % 12) / 12),
                np.cos(2 * np.pi * ((di // 30) % 12) / 12),
                window.mean(), window.std() + 1e-6]

    X, y, pers = [], [], []
    for i in range(len(log_s) - W - 1):
        X.append(np.concatenate([log_s[i:i + W], feats(log_s[i:i + W], i + W)]).astype(np.float32))
        y.append(log_s[i + W]); pers.append(log_s[i + W - 1])
    X, y, pers = np.array(X, np.float32), np.array(y, np.float32), np.array(pers, np.float32)
    Xtr, Xte, ytr, yte, xsc, ysc, n = split_scale(X, y)
    net = train_nsde(Xtr, ytr, hidden=128, activation="silu", lr=3e-4, obs_std=0.05, beta_max=0.05)

    def build_ctx(win, t, h):
        di = t + h
        ex = np.array([np.sin(2 * np.pi * (di % 7) / 7), np.cos(2 * np.pi * (di % 7) / 7),
                       np.sin(2 * np.pi * ((di // 30) % 12) / 12),
                       np.cos(2 * np.pi * ((di // 30) % 12) / 12)], np.float32)
        m = win.shape[0]
        extra = np.concatenate([np.tile(ex, (m, 1)),
                                win.mean(1, keepdims=True), win.std(1, keepdims=True) + 1e-6], 1)
        return np.concatenate([win, extra], 1).astype(np.float32)

    lo_a, hi_a = n + W, len(log_s) - 12
    finalize("energy", "Энергопотребление (UCI Household)", "кВт·ч/день",
             net, xsc, ysc, Xte, yte, pers[n:], build_ctx, log_s, np.expm1,
             K=W, H=10, anchor_lo=lo_a, anchor_hi=hi_a)


# ── 3. Глюкоза / инсулин (OhioT1DM, пациент 559) ──────────────────────────
def prepare_glucose():
    data_dir = next((p for p in [BASE / "data_insulin", BASE.parent / "data_insulin"]
                     if p.exists()), None)
    xml = data_dir / "559-ws-training.xml"
    print(f"[glucose] читаю {xml.name}...")
    root = ET.parse(xml).getroot()
    vals = [float(v) for ev in root.find("glucose_level").findall("event")
            if pd.notna(v := pd.to_numeric(ev.get("value"), errors="coerce"))]
    values = np.array(vals, np.float32)

    W = 24
    mean_p, std_p = float(values.mean()), float(values.std() + 1e-6)
    z = ((values - mean_p) / std_p).astype(np.float32)

    def feats(zw):
        vw = zw * std_p + mean_p
        dz = np.diff(vw, prepend=vw[0])
        slope = (vw[-1] - vw[max(0, len(vw) - 4)]) / 3.0
        return [zw.mean(), zw.std() + 1e-6, dz.max(), dz.min(), slope]

    X, y, pers = [], [], []
    for i in range(len(values) - W - 1):
        X.append(np.concatenate([z[i:i + W], feats(z[i:i + W])]).astype(np.float32))
        y.append(z[i + W]); pers.append(z[i + W - 1])
    X, y, pers = np.array(X, np.float32), np.array(y, np.float32), np.array(pers, np.float32)
    Xtr, Xte, ytr, yte, xsc, ysc, n = split_scale(X, y)
    net = train_nsde(Xtr, ytr, hidden=128, activation="silu", lr=5e-4, obs_std=0.08, beta_max=0.04)

    def build_ctx(win, t, h):
        vw = win * std_p + mean_p
        dz = np.diff(vw, prepend=vw[:, :1], axis=1)
        slope = (vw[:, -1] - vw[:, -4]) / 3.0
        extra = np.stack([win.mean(1), win.std(1) + 1e-6,
                          dz.max(1), dz.min(1), slope], 1)
        return np.concatenate([win, extra], 1).astype(np.float32)

    denorm = lambda a: a * std_p + mean_p
    lo_a, hi_a = n + W, len(z) - 14
    finalize("glucose", "Глюкоза CGM (OhioT1DM, пациент 559)", "mg/dL",
             net, xsc, ysc, Xte, yte, pers[n:], build_ctx, z, denorm,
             K=W, H=12, anchor_lo=lo_a, anchor_hi=hi_a)


# ── 4. Ветрогенерация (GEFCom2014, зона 1) ────────────────────────────────
def prepare_wind():
    csv = next((p for p in [BASE / "Task15_W_Zone1.csv", BASE.parent / "Task15_W_Zone1.csv"]
                if p.exists()), None)
    print(f"[wind] читаю {csv.name}...")
    d = pd.read_csv(csv)
    d["TIMESTAMP"] = pd.to_datetime(d["TIMESTAMP"], format="%Y%m%d %H:%M")
    d = d.sort_values("TIMESTAMP").reset_index(drop=True)
    d["ws100"] = np.sqrt(d.U100 ** 2 + d.V100 ** 2)
    d["ws10"] = np.sqrt(d.U10 ** 2 + d.V10 ** 2)
    hh = d.TIMESTAMP.dt.hour
    d["hs"], d["hc"] = np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24)
    d = d.dropna(subset=["TARGETVAR"]).reset_index(drop=True)

    P = d.TARGETVAR.values.astype(np.float32)
    w1, w0 = d.ws100.values.astype(np.float32), d.ws10.values.astype(np.float32)
    hs, hc = d.hs.values.astype(np.float32), d.hc.values.astype(np.float32)
    W = 12
    X, y, pers = [], [], []
    for t in range(W, len(d) - 1):
        extra = [w1[t + 1], w0[t + 1], hs[t + 1], hc[t + 1]]
        X.append(np.concatenate([P[t - W:t], extra]).astype(np.float32))
        y.append(P[t + 1]); pers.append(P[t])
    X, y, pers = np.array(X, np.float32), np.array(y, np.float32), np.array(pers, np.float32)
    Xtr, Xte, ytr, yte, xsc, ysc, n = split_scale(X, y)
    net = train_nsde(Xtr, ytr, hidden=128, activation="silu", lr=5e-4, obs_std=0.08, beta_max=0.04)

    # raw_series для rollout = P, но индексация в X сдвинута на W (t начинался с W)
    def build_ctx(win, t, h):
        m = win.shape[0]
        ex = np.array([w1[t + h], w0[t + h], hs[t + h], hc[t + h]], np.float32)
        return np.concatenate([win, np.tile(ex, (m, 1))], 1).astype(np.float32)

    # anchor t здесь — абсолютный индекс в P; первый тестовый якорь после train-части
    lo_a, hi_a = W + n, len(P) - 14
    finalize("wind", "Ветрогенерация (GEFCom2014, зона 1)", "доля номинальной мощности",
             net, xsc, ysc, Xte, yte, pers[n:], build_ctx, P,
             lambda a: np.clip(a, 0, 1), K=W, H=12, anchor_lo=lo_a, anchor_hi=hi_a)


DOMAINS = {
    "stocks": prepare_stocks,
    "energy": prepare_energy,
    "glucose": prepare_glucose,
    "wind": prepare_wind,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(DOMAINS), default=None)
    args, _ = ap.parse_known_args()
    for d in ([args.only] if args.only else list(DOMAINS)):
        DOMAINS[d]()
    print("\nГотово. Дашборд: streamlit run app.py")
