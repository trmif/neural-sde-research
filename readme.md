# Beyond Infinity: Neural SDEs in Practice

[Тимур Мифтахутдинов](https://github.com/trmif) | [Мария Сухова](https://github.com/moria_vohus) | [Арина Басак](https://github.com/rrqwt)

Что если вместо фиксированного числа слоёв нейросеть имела бы бесконечную глубину? Именно этот вопрос приводит к Neural SDE — стохастическому дифференциальному уравнению где drift и диффузия реализованы нейросетями. Мы взяли теорию Tzen & Raginsky (2019), вышли за пределы синтетических экспериментов и проверили — работает ли это на практике.

## Ссылки

- 📖 [О нашей работе](https://trmif.github.io/neural-sde-research/)
- 📊 [Дашборд](https://neural-sde.streamlit.app/)
- 📄 [Tzen & Raginsky (2019a)](https://arxiv.org/abs/1903.01608)
- 📄 [Tzen & Raginsky (2019b)](https://arxiv.org/abs/1905.09883)

## Структура репозитория

```
.
├── docs/ # Jupyter Book (текст проекта)
│ ├── assets/ # Картинки и диаграммы
│ ├── 01_intro.md # Введение и RQ
│ ├── 02_background.md # Предпосылки и история
│ ├── 03_neural_sde.md # Как работает Neural SDE
│ ├── 04_experiments.md # Эксперименты и выводы
├── src/ # Имплементация
│ ├── models/ # Neural SDE, бейзлайны
│ ├── experiments/ # Ноуты экспериментов
│ └── notebooks/ # Исследовательские ноутбуки
├── dashboard/ # Streamlit дашборд
├── \_config.yml # Конфиг Jupyter Book
├── \_toc.yml # Оглавление Jupyter Book
└── requirements.txt # Зависимости
```

## Воспроизведение

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск дашборда

```bash
streamlit run dashboard/app.py
```
