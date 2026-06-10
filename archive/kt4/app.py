"""
Дашборд Neural SDE. Две страницы:
  1. Сравнение трёх доменов — почему выбрали глюкозу
  2. Прогноз глюкозы — рабочий инструмент с фильтрами
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Кэш ищем рядом с этим файлом — так работает и локально, и на Streamlit Cloud
# (там рабочая папка = корень репо, а app.py лежит в kt4/).
# __file__ есть при запуске через streamlit; в ноутбуке его нет — берём cwd.
try:
    BASE = Path(__file__).parent
except NameError:
    BASE = Path.cwd()
CACHE = BASE / "cache"

st.set_page_config(page_title="Neural SDE", layout="wide", page_icon="📈")


@st.cache_data
def load_npz(name):
    return dict(np.load(CACHE / name, allow_pickle=True))


@st.cache_data
def load_json(name):
    with open(CACHE / name) as f:
        return json.load(f)


def check_cache():
    missing = [f for f in ["glucose.npz", "finance.npz", "energy.npz"]
               if not (CACHE / f).exists()]
    if missing:
        st.error(f"Нет кэша: {', '.join(missing)}. Сначала запусти `python prepare.py`.")
        st.stop()


check_cache()


# ============================================================
# Sidebar


# ============================================================
st.sidebar.title("Neural SDE")
page = st.sidebar.radio("Раздел", [
    "1. Сравнение доменов",
    "2. Прогноз глюкозы",
])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Latent Neural SDE по Tzen & Raginsky (2019). "
    "Три домена пробовали — глюкоза победила по калибровке."
)


# ============================================================
# СТРАНИЦА 1 — сравнение доменов


# ============================================================
if page.startswith("1"):
    st.title("Где Neural SDE действительно полезен")
    st.markdown(
        "Мы прогнали один и тот же подход (нейросеть как дрейф SDE + Эйлер–Маруяма + ELBO) "
        "на трёх доменах. Цель страницы — показать честно, где он работает, а где нет."
    )

    fin = load_npz("finance.npz")
    fin_m = load_json("finance_metrics.json")
    eng = load_npz("energy.npz")
    eng_m = load_json("energy_metrics.json")
    glu_m = load_json("glucose_metrics.json")

    # --- блок 1: финансы ---
    st.markdown("### Финансы — AAPL, лог-доходности")
    c1, c2 = st.columns([3, 2])
    with c1:
        n_show = st.slider("Показать дней", 50, min(400, len(fin["Y_test"])), 150,
                           key="fin_n")
        df_fin = pd.DataFrame({
            "день": np.arange(n_show),
            "факт": fin["Y_test"][:n_show],
            "Neural SDE": fin["nsde_pred"][:n_show],
            "ARIMA": fin["arima_pred"][:n_show],
            "SARIMA": fin["sarima_pred"][:n_show],
        }).melt("день", var_name="модель", value_name="доходность")
        chart = alt.Chart(df_fin).mark_line().encode(
            x="день:Q", y="доходность:Q",
            color=alt.Color("модель:N", scale=alt.Scale(
                domain=["факт", "Neural SDE", "ARIMA", "SARIMA"],
                range=["black", "#d62728", "#1f77b4", "#2ca02c"])),
            opacity=alt.condition(alt.datum.модель == "факт",
                                  alt.value(1.0), alt.value(0.6)),
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    with c2:
        st.markdown("**RMSE (лог-доходность)**")
        st.dataframe(pd.DataFrame({
            "модель": ["Neural SDE", "ARIMA", "SARIMA"],
            "RMSE": [fin_m["nsde_rmse"], fin_m["arima_rmse"], fin_m["sarima_rmse"]],
        }).set_index("модель").style.format("{:.5f}"))
        st.caption(
            "RMSE между моделями отличается на 3-м знаке. "
            "Цены — почти мартингал, любая модель учится предсказывать ноль. "
            "Сюда Neural SDE не приносит ничего, чего бы не дал ARIMA."
        )

    st.markdown("---")

    # --- блок 2: энергия ---
    st.markdown("### Энергопотребление — UCI Household, дневные суммы")
    c1, c2 = st.columns([3, 2])
    with c1:
        n_e = st.slider("Показать дней", 30, min(200, len(eng["Y_test"])), 90,
                        key="eng_n")
        df_eng = pd.DataFrame({
            "день": np.arange(n_e),
            "факт": eng["Y_test"][:n_e],
            "Neural SDE": eng["nsde_pred"][:n_e],
            "SARIMA": eng["sarima_pred"][:n_e],
            "ci_lo": eng["ci_lo"][:n_e],
            "ci_hi": eng["ci_hi"][:n_e],
        })
        base = alt.Chart(df_eng).encode(x="день:Q")
        band = base.mark_area(opacity=0.2, color="#d62728").encode(
            y="ci_lo:Q", y2="ci_hi:Q")
        long = df_eng[["день", "факт", "Neural SDE", "SARIMA"]].melt(
            "день", var_name="модель", value_name="кВт·ч")
        lines = alt.Chart(long).mark_line().encode(
            x="день:Q", y="кВт·ч:Q",
            color=alt.Color("модель:N", scale=alt.Scale(
                domain=["факт", "Neural SDE", "SARIMA"],
                range=["black", "#d62728", "#1f77b4"])),
        )
        st.altair_chart((band + lines).properties(height=300),
                        use_container_width=True)
    with c2:
        st.markdown("**RMSE (кВт·ч)**")
        st.dataframe(pd.DataFrame({
            "модель": ["Neural SDE", "SARIMA"],
            "RMSE": [eng_m["nsde_rmse"], eng_m["sarima_rmse"]],
        }).set_index("модель").style.format("{:.2f}"))
        st.caption(
            "SDE даёт сопоставимое качество и доверительный интервал, "
            "но классическая SARIMA здесь — сильный бейзлайн. "
            "Сезонность хорошо описывается параметрической моделью."
        )

    st.markdown("---")

    # --- блок 3: глюкоза, и тут видно разницу ---
    st.markdown("### Глюкоза — OhioT1DM, CGM каждые 5 минут, прогноз на +30 мин")
    c1, c2 = st.columns([3, 2])
    with c1:
        lstm = [glu_m["lstm"]["cov80"], glu_m["lstm"]["cov90"], glu_m["lstm"]["cov95"]]
        nsde = [glu_m["nsde"]["cov80"], glu_m["nsde"]["cov90"], glu_m["nsde"]["cov95"]]
        nominal = [0.80, 0.90, 0.95]
        df_cal = pd.DataFrame({
            "номинал": nominal * 2,
            "эмпирическое покрытие": lstm + nsde,
            "модель": ["LSTM"] * 3 + ["Neural SDE"] * 3,
        })
        diag = alt.Chart(pd.DataFrame({"x": [0.7, 1.0], "y": [0.7, 1.0]})).mark_line(
            strokeDash=[4, 4], color="gray").encode(x="x", y="y")
        pts = alt.Chart(df_cal).mark_circle(size=180).encode(
            x=alt.X("номинал:Q", scale=alt.Scale(domain=[0.75, 1.0])),
            y=alt.Y("эмпирическое покрытие:Q", scale=alt.Scale(domain=[0.6, 1.0])),
            color=alt.Color("модель:N", scale=alt.Scale(
                domain=["LSTM", "Neural SDE"], range=["#1f77b4", "#d62728"])),
            tooltip=["модель", "номинал", "эмпирическое покрытие"],
        )
        st.altair_chart((diag + pts).properties(
            height=320, title="Калибровка: чем ближе к диагонали — тем правдивее интервалы"),
            use_container_width=True)
    with c2:
        st.markdown("**Метрики на +30 мин**")
        st.dataframe(pd.DataFrame({
            "метрика": ["MAE, mg/dL", "RMSE, mg/dL", "CRPS, mg/dL",
                        "Cov 80%", "Cov 90%", "Cov 95%"],
            "LSTM": [glu_m["lstm"]["mae"], glu_m["lstm"]["rmse"], glu_m["lstm"]["crps"],
                     glu_m["lstm"]["cov80"], glu_m["lstm"]["cov90"], glu_m["lstm"]["cov95"]],
            "Neural SDE": [glu_m["nsde"]["mae"], glu_m["nsde"]["rmse"], glu_m["nsde"]["crps"],
                           glu_m["nsde"]["cov80"], glu_m["nsde"]["cov90"], glu_m["nsde"]["cov95"]],
        }).set_index("метрика").style.format("{:.3f}"))
        st.caption(
            "По точечным метрикам (MAE/RMSE) модели близки, "
            "но Neural SDE даёт калиброванное предсказательное распределение. "
            "Для CGM это критично — мы хотим знать вероятность гипогликемии, "
            "а не только среднее."
        )

    st.markdown("---")
    st.success(
        "**Вывод.** Финансы — Neural SDE не помогает (шум доминирует). "
        "Энергия — работает, но SARIMA не хуже. "
        "Глюкоза — выигрывает там, где важна не точка, а распределение. "
        "Поэтому итоговая прикладная задача — прогноз CGM."
    )


# ============================================================
# СТРАНИЦА 2 — глюкоза, рабочий инструмент


# ============================================================
else:
    g = load_npz("glucose.npz")
    glu_m = load_json("glucose_metrics.json")

    X_test = g["X_test_raw"]
    Y_test = g["Y_test_raw"]
    ts_test = pd.to_datetime(g["ts_test"])
    lstm_mu = g["lstm_mu"]
    lstm_sigma = g["lstm_sigma"]
    nsde_samples = g["nsde_samples"]
    patient = int(g["patient"])

    K = X_test.shape[1]  # история, 24 шага
    H = Y_test.shape[1]  # горизонт, 6 шагов = 30 минут
    step_min = 5

    st.title(f"Прогноз глюкозы — пациент {patient}")
    st.caption("Latent Neural SDE: предсказание CGM на горизонт до 30 минут "
               "с калиброванным доверительным интервалом")

    # --- контролы ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Настройки**")

    band_pct = st.sidebar.select_slider(
        "Ширина доверительного интервала",
        options=[50, 70, 80, 90, 95, 99], value=90,
    )
    horizon_min = st.sidebar.select_slider(
        "Горизонт прогноза, мин",
        options=[5, 10, 15, 20, 25, 30], value=30,
    )
    horizon_step = horizon_min // step_min - 1  # индекс шага в Y

    hour_range = st.sidebar.slider(
        "Время суток (час начала прогноза)", 0, 23, (0, 23),
    )
    show_lstm = st.sidebar.checkbox("Показывать LSTM-бейзлайн", value=True)

    # --- фильтр окон по времени суток ---
    hours = pd.Series(ts_test).dt.hour.values
    mask = (hours >= hour_range[0]) & (hours <= hour_range[1])
    idx_filt = np.where(mask)[0]

    if len(idx_filt) == 0:
        st.warning("Нет окон в выбранном диапазоне часов. Расширь интервал.")
        st.stop()

    # --- сводные метрики (на выбранных окнах, выбранном горизонте) ---
    y_h = Y_test[idx_filt, horizon_step]
    lstm_mu_h = lstm_mu[idx_filt, horizon_step]
    lstm_s_h = lstm_sigma[idx_filt, horizon_step]
    ns_h = nsde_samples[idx_filt, horizon_step, :]
    ns_mean = ns_h.mean(axis=-1)

    q_lo = (1 - band_pct / 100) / 2
    q_hi = 1 - q_lo
    ns_lo = np.quantile(ns_h, q_lo, axis=-1)
    ns_hi = np.quantile(ns_h, q_hi, axis=-1)
    coverage = float(np.mean((y_h >= ns_lo) & (y_h <= ns_hi)))
    mae = float(np.mean(np.abs(y_h - ns_mean)))
    rmse = float(np.sqrt(np.mean((y_h - ns_mean) ** 2)))

    # доля попаданий в зоны гипо/гипер по нижнему квантилю
    hypo_risk = float(np.mean(ns_lo < 70))
    hyper_risk = float(np.mean(ns_hi > 180))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Окон в выборке", f"{len(idx_filt):,}")
    k2.metric(f"MAE, +{horizon_min} мин", f"{mae:.1f} mg/dL")
    k3.metric(f"RMSE, +{horizon_min} мин", f"{rmse:.1f} mg/dL")
    k4.metric(f"Покрытие {band_pct}%", f"{coverage*100:.1f}%",
              delta=f"{(coverage - band_pct/100)*100:+.1f} п.п.",
              delta_color="off")
    k5.metric("Риск гипо (<70)", f"{hypo_risk*100:.1f}%")

    st.markdown("---")

    # --- визуализация конкретного окна ---
    col_a, col_b = st.columns([1, 3])

    with col_a:
        st.markdown("**Окно из выборки**")
        if st.button("🎲 Случайное"):
            st.session_state.pick = int(np.random.choice(idx_filt))
        pick = st.session_state.get("pick", int(idx_filt[0]))
        if pick not in idx_filt:
            pick = int(idx_filt[0])

        pick = st.selectbox(
            "Или выбери вручную",
            options=idx_filt.tolist(),
            index=list(idx_filt).index(pick) if pick in idx_filt else 0,
            format_func=lambda i: f"#{i}  •  {pd.Timestamp(ts_test[i]).strftime('%Y-%m-%d %H:%M')}",
        )
        st.session_state.pick = pick

        st.markdown(f"**Время прогноза:** {pd.Timestamp(ts_test[pick]).strftime('%Y-%m-%d %H:%M')}")
        st.markdown(
            f"**Истинное значение через {horizon_min} мин:** "
            f"`{Y_test[pick, horizon_step]:.1f}` mg/dL"
        )
        st.markdown(
            f"**Прогноз NSDE:** `{ns_mean[list(idx_filt).index(pick)]:.1f}` "
            f"({ns_lo[list(idx_filt).index(pick)]:.0f} – "
            f"{ns_hi[list(idx_filt).index(pick)]:.0f})"
        )

    with col_b:
        # время в минутах относительно точки прогноза
        t_hist = np.arange(-K + 1, 1) * step_min
        t_fut = np.arange(1, H + 1) * step_min
        t_fut_use = t_fut[:horizon_step + 1]

        # история
        df_hist = pd.DataFrame({"мин": t_hist, "глюкоза": X_test[pick]})
        df_true = pd.DataFrame({"мин": t_fut, "глюкоза": Y_test[pick]})

        # прогноз NSDE (mean + квантили) на нужном горизонте
        S = nsde_samples[pick, :horizon_step + 1, :]  # (h, MC)
        nsde_mean_t = S.mean(axis=-1)
        nsde_lo_t = np.quantile(S, q_lo, axis=-1)
        nsde_hi_t = np.quantile(S, q_hi, axis=-1)
        df_nsde = pd.DataFrame({
            "мин": t_fut_use,
            "прогноз": nsde_mean_t,
            "lo": nsde_lo_t,
            "hi": nsde_hi_t,
        })

        # медицинские пороги
        threshold = pd.DataFrame({"уровень": [70, 180],
                                  "label": ["гипогликемия", "гипергликемия"]})

        hist_line = alt.Chart(df_hist).mark_line(color="#2ca580", strokeWidth=2).encode(
            x="мин:Q", y=alt.Y("глюкоза:Q", scale=alt.Scale(zero=False)))
        true_line = alt.Chart(df_true).mark_line(color="black", point=True,
                                                  strokeWidth=1.5).encode(
            x="мин:Q", y="глюкоза:Q")
        nsde_band = alt.Chart(df_nsde).mark_area(opacity=0.25, color="#d62728").encode(
            x="мин:Q", y="lo:Q", y2="hi:Q")
        nsde_line = alt.Chart(df_nsde).mark_line(color="#d62728", strokeDash=[4, 2],
                                                  strokeWidth=2).encode(
            x="мин:Q", y="прогноз:Q")
        rules = alt.Chart(threshold).mark_rule(strokeDash=[2, 2], opacity=0.5,
                                                color="gray").encode(y="уровень:Q")
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(opacity=0.4).encode(x="x:Q")

        layers = [hist_line, nsde_band, nsde_line, true_line, rules, zero]

        if show_lstm:
            from scipy.stats import norm
            z = norm.ppf(q_hi)
            lstm_mu_w = lstm_mu[pick, :horizon_step + 1]
            lstm_s_w = lstm_sigma[pick, :horizon_step + 1]
            df_lstm = pd.DataFrame({
                "мин": t_fut_use,
                "прогноз": lstm_mu_w,
                "lo": lstm_mu_w - z * lstm_s_w,
                "hi": lstm_mu_w + z * lstm_s_w,
            })
            lstm_band = alt.Chart(df_lstm).mark_area(opacity=0.15, color="#1f77b4").encode(
                x="мин:Q", y="lo:Q", y2="hi:Q")
            lstm_line = alt.Chart(df_lstm).mark_line(color="#1f77b4", strokeDash=[6, 3],
                                                      strokeWidth=1.5).encode(
                x="мин:Q", y="прогноз:Q")
            layers = [hist_line, lstm_band, nsde_band, lstm_line, nsde_line,
                      true_line, rules, zero]

        chart = alt.layer(*layers).properties(
            height=420,
            title=f"Окно #{pick}: история (зелёный), факт (чёрный), "
                  f"NSDE {band_pct}% (красный)" +
                  (", LSTM (синий)" if show_lstm else ""),
        ).resolve_scale(y="shared")
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # --- агрегированная аналитика по часу суток ---
    st.markdown("### Качество прогноза в течение суток")
    by_hour = []
    for h in range(24):
        m = (hours == h)
        if m.sum() < 10:
            continue
        y_h_h = Y_test[m, horizon_step]
        nsde_m = nsde_samples[m, horizon_step, :].mean(axis=-1)
        nsde_lo_h = np.quantile(nsde_samples[m, horizon_step, :], q_lo, axis=-1)
        nsde_hi_h = np.quantile(nsde_samples[m, horizon_step, :], q_hi, axis=-1)
        by_hour.append({
            "час": h,
            "MAE": float(np.mean(np.abs(y_h_h - nsde_m))),
            "покрытие": float(np.mean((y_h_h >= nsde_lo_h) & (y_h_h <= nsde_hi_h))),
            "окон": int(m.sum()),
        })
    df_h = pd.DataFrame(by_hour)
    if len(df_h):
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                alt.Chart(df_h).mark_bar(color="#d62728").encode(
                    x=alt.X("час:O", title="час суток"),
                    y=alt.Y("MAE:Q", title="MAE, mg/dL"),
                    tooltip=["час", "MAE", "окон"],
                ).properties(height=260, title=f"MAE по часам (горизонт +{horizon_min} мин)"),
                use_container_width=True)
        with c2:
            df_h2 = df_h.copy()
            df_h2["номинал"] = band_pct / 100
            cov_bars = alt.Chart(df_h2).mark_bar(color="#1f77b4").encode(
                x=alt.X("час:O"), y=alt.Y("покрытие:Q", title=f"покрытие {band_pct}%"),
                tooltip=["час", "покрытие", "окон"])
            cov_rule = alt.Chart(df_h2).mark_rule(color="black", strokeDash=[3, 3]).encode(
                y="номинал:Q")
            st.altair_chart((cov_bars + cov_rule).properties(
                height=260, title=f"Эмпирическое покрытие vs номинал {band_pct}%"),
                use_container_width=True)
        st.caption(
            "Ночью (когда пациент спит и питания нет) сигнал самый предсказуемый — "
            "MAE и доверительные интервалы здесь обычно меньше. "
            "Это важно: ночная гипогликемия — главный риск, и именно ночью модель надёжнее."
        )
