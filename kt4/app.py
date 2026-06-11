"""
Дашборд: «Наши попытки применить Neural SDE».

4 домена (акции, энергопотребление, глюкоза/инсулин, ветрогенерация) — одинаковая
картина вероятностного прогноза:
  • историческая траектория (факт прошлого),
  • предсказанная траектория Neural SDE + доверительный интервал (веер MC-сэмплов),
  • реальное будущее для сверки,
  • выученное распределение прогноза в точке горизонта (гистограмма сэмплов).
Данные — из cache_attempts/ (готовит prepare_attempts.py). Запуск: streamlit run app.py
"""
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

st.set_page_config(page_title="Наши попытки применить Neural SDE",
                   layout="wide", page_icon="🌀")

DOMAINS = {
    "Акции": "stocks",
    "Энергопотребление": "energy",
    "Глюкоза / инсулин": "glucose",
    "Ветрогенерация": "wind",
}
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
st.title("Наши попытки применить Neural SDE")
st.markdown(
    "Один подход — **нейросеть как дрейф SDE + Эйлер–Маруяма + KL-регуляризация** "
    "(Tzen & Raginsky, 2019) — на четырёх задачах. Модель прогоняется на несколько "
    "шагов вперёд много раз (Монте-Карло) и выдаёт не точку, а **распределение**: "
    "веер траекторий с доверительным интервалом."
)

if not CACHE.exists() or not list(CACHE.glob("*.npz")):
    st.error("Нет данных в cache_attempts/. Сначала запусти `python prepare_attempts.py`.")
    st.stop()

tabs = st.tabs(list(DOMAINS.keys()))
for tab, (name, domain) in zip(tabs, DOMAINS.items()):
    with tab:
        st.header(name)
        render(domain)

st.markdown("---")
st.caption(
    "Курсовой проект по статье Tzen & Raginsky (2019). Прогноз и распределение "
    "строятся вживую из обученных моделей (prepare_attempts.py), а не из картинок."
)
