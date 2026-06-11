# Интерактивный дашборд

Прикладной результат проекта — интерактивный дашборд на **Streamlit**. Он построен
на Latent Neural SDE (Tzen & Raginsky, 2019) и состоит из двух страниц:

1. **Сравнение трёх доменов** — финансы (AAPL), энергопотребление (UCI Household) и
   глюкоза (OhioT1DM). Показывает честно, где Neural SDE даёт выигрыш, а где
   классические модели (ARIMA / SARIMA) не хуже.
2. **Прогноз глюкозы** — рабочий инструмент: прогноз CGM на горизонт до 30 минут с
   калиброванным доверительным интервалом, фильтрами по времени суток и оценкой
   риска гипогликемии.

```{note}
Дашборд — это живое Streamlit-приложение, оно работает на отдельном сервере и не
входит в статическую сборку книги. Ниже оно встроено через iframe; если фрейм не
загрузился — открой его по прямой ссылке.
```

## Открыть дашборд

<a href="https://YOUR-APP.streamlit.app" target="_blank" rel="noopener"
   style="display:inline-block;padding:10px 18px;background:#d62728;color:#fff;
          border-radius:6px;text-decoration:none;font-weight:600;">
  🚀 Открыть дашборд в отдельной вкладке
</a>

## Встроенный просмотр

<iframe
  src="https://YOUR-APP.streamlit.app/?embed=true"
  width="100%"
  height="820"
  style="border:1px solid #ddd;border-radius:8px;"
  loading="lazy"
  title="Neural SDE Dashboard">
</iframe>

---

## Как запустить локально

```bash
cd kt4
python prepare.py        # один раз: обучает модели, складывает артефакты в cache/
streamlit run app.py     # запускает дашборд на http://localhost:8501
```

Ноутбуки [`prepare.ipynb`](prepare.ipynb) и [`app.ipynb`](app.ipynb) — те же файлы для
просмотра/редактирования логики; перед деплоем экспортируются в `.py`:

```bash
jupyter nbconvert --to script kt4/app.ipynb kt4/prepare.ipynb
```
