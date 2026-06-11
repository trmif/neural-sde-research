# Neural SDE

## 1. Deep Latent Gaussian Models

Отправной точкой служат Deep Latent Gaussian Models (DLGM, Rezende et al., 2014). Латентные переменные $X_0, \ldots, X_k$ и наблюдение $Y$ порождаются рекурсивно:

$$X_0 = Z_0$$

$$X_i = X_{i-1} + b_i(X_{i-1}) + \sigma_i Z_i, \quad i = 1, \ldots, k$$

$$Y \sim p(\cdot \mid X_k)$$

где $Z_0, \ldots, Z_k \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, I_d)$ — независимые стандартные гауссовские векторы, $b_i$ — нелинейные преобразования (нейросети), $\sigma_i$ — матрицы шума.

Вся случайность модели сосредоточена в примитивных объектах $Z_i$, а всё остальное — детерминированные преобразования. Совместное распределение $Y$ и $Z_0, \ldots, Z_k$ записывается как:

$$p_\theta(y, z_0, \ldots, z_k) = p(y \mid f_\theta(z_0, \ldots, z_k)) \prod_{i=0}^k \phi_d(z_i)$$

где $f_\theta$ — детерминированное отображение $(Z_0, \ldots, Z_k) \mapsto X_k$.

### Loss (ELBO)

Основной объект интереса — маргинальное правдоподобие $p_\theta(y)$, которое получается интегрированием по всем латентным переменным. Это интегрирование интрактабельно, поэтому вместо него минимизируют вариационную свободную энергию (Evidence Lower Bound, ELBO):

$$-\log p_\theta(y) \leq \inf_\beta\, \mathsf{F}_{\theta,\beta}(y)$$

$$\mathsf{F}_{\theta,\beta}(y) = \sum_{i=0}^k D\!\left(\mathcal{N}(\tilde{b}_i(y), C_i(y)) \,\|\, \mathcal{N}(0, I_d)\right) + \mathbf{E}\!\left[F_\theta\!\left(\tilde{b}_0(y) + C_0(y)^{1/2}Z_0, \ldots\right)\right]$$

где $F_\theta(z_0, \ldots, z_k) := -\log p(y \mid f_\theta(z_0, \ldots, z_k))$.

Лосс состоит из двух слагаемых:

- **KL-дивергенция** — штрафует за отклонение апостериорного распределения от априорного $\mathcal{N}(0, I)$. Для гауссиан считается аналитически.
- **Реконструкция** — измеряет насколько хорошо финальное состояние $X_1$ объясняет наблюдение $Y$ через максимизацию $\int_\Omega \log p(y \mid f_\theta(z_0, \ldots, z_k))\, \nu(dz_0, \ldots, dz_k \mid y)$

Градиенты считаются через репараметризационный трюк: вся случайность выносится в $Z_i \sim \mathcal{N}(0, I)$ которые не зависят от параметров, что позволяет протащить градиент сквозь ожидание.

## 2. Теоретический результат: любое распределение сэмплируемо

Ключевой теоретический результат Tzen & Raginsky (2019) — Neural SDE с обучаемым drift достаточно выразительна чтобы сэмплировать из **любого** целевого распределения $q$.

Для любой плотности вида $q(x) = f(x)\phi_d(x)$ существует оптимальный drift — **дрейф Фёллмера**:

$$b^*(x, t) = \nabla_x \log Q_{1-t}f(x), \quad Q_t f(x) := \mathbf{E}_{Z \sim \phi_d}[f(x + \sqrt{t}\,Z)]$$

При этом если запустить SDE с таким drift:

$$dX_t = \nabla_x \log Q_{1-t}f(X_t)\, dt + dW_t, \quad X_0 = 0$$

то $X_1 \sim q$ — точно

## 3. Предельный переход: DLGM → Neural SDE

Рассмотрим что происходит с DLGM когда число слоёв $k \to \infty$, шаг $\Delta t = 1/k \to 0$, дисперсия шума $\sigma_i \to 0$:

$$X_i = X_{i-1} + b_i(X_{i-1})\Delta t + \sigma_i\sqrt{\Delta t}\, Z_i \quad \xrightarrow{k \to \infty} \quad dX_t = b(X_t, t;\,\theta)\,dt + \sigma(X_t, t;\,\theta)\,dW_t$$

Дискретная марковская цепь становится непрерывным диффузионным процессом Ито. Наблюдение теперь порождается из финального состояния: $Y \sim p(\cdot \mid X_1)$.

Вместо $k$ отдельных нейросетей $b_i$ — одна сеть $b(x, t;\theta)$ которая принимает время $t$ как вход. Это **Neural SDE**.

## 4. Вариационный вывод для Neural SDE

### Пространство путей и мера Винера

В DLGM вся случайность выносилась в $Z_i \sim \mathcal{N}(0, I)$. В Neural SDE аналогом служит стандартный винеровский процесс $W = \{W_t\}_{t \in [0,1]}$ с мерой Винера $\mu$.

Совместное распределение в пространстве путей:

$$P_\theta(dy, dw) = p(y \mid [f_\theta(w)]_1)\,\mu(dw)\,dy$$

### Теорема Гирсанова и репараметризация

Вариационная формула на пространстве путей:

$$-\log p_\theta(y) = \inf_{\nu \in P(\mathbb{W})} \left\{D(\nu \| \mu) - \int_\mathbb{W} \log p(y \mid [f_\theta(w)]_1)\,\nu(dw)\right\}$$

Ключевое упрощение даёт теорема Гирсанова: любая мера $\nu$ абсолютно непрерывная относительно меры Винера $\mu$ соответствует сдвигу Винера на детерминированный drift:

$$Z_t = W_t + \int_0^t u_s\,ds \sim \nu, \quad D(\nu \| \mu) = \mathbf{E}_\mu\!\left[\frac{1}{2}\int_0^1 \|u_t\|^2\,dt\right]$$

Параметризуя drift нейросетью $u_t = \tilde{b}(y, t;\beta)$, получаем **mean-field вариационную границу**:

$$\mathsf{F}_{\theta,\beta}(y) = \frac{1}{2}\int_0^1 \|\tilde{b}(y, t;\beta)\|^2\,dt + \mathbf{E}\!\left[F_\theta\!\left(W + \int_0^\cdot \tilde{b}(y, s;\beta)\,ds\right)\right]$$

## 5. Как считать градиенты

Вычисление градиентов $\nabla_\theta \mathsf{F}_{\theta,\beta}$ и $\nabla_\beta \mathsf{F}_{\theta,\beta}$ требует дифференцирования через решение SDE — это нетривиально.

### Проблема adjoint метода

В Neural ODE градиенты считаются через adjoint sensitivity method: прогоняем ODE вперёд, затем решаем сопряжённое ODE назад — эффективно с любым black-box ODE солвером.

В Neural SDE этот подход не работает напрямую: сопряжённое уравнение становится forward-backward SDE (FBSDE) — системой где backward уравнение зависит от всего пути $W_t$ и нарушает причинность. Эффективных black-box солверов для FBSDE не существует.

Предлагается два решения этой задачи

### Подход 1 | Euler-Manuyama approximation

Дискретизируем SDE схемой Эйлера и применяем стандартный backprop:

$$\hat{X}_{t_{i+1}} = \hat{X}_{t_i} + h_{i+1}\left(b(\hat{X}_{t_i}, t_i;\theta) + \tilde{b}(y, t_i;\beta)\right) + \sqrt{h_{i+1}}\,\sigma(\hat{X}_{t_i}, t_i;\theta)\,Z_{i+1}$$

Случайные $Z_{i+1}$ сэмплируются заранее и становятся константами для автодифференцирования — получаем обычный граф вычислений.

** Euler Backprop для Neural SDE математически эквивалентен Stochastic Backpropagation (Rezende et al., 2014) для DLGM — непрерывный и дискретный случаи сходятся! **

### Подход 2 | Pathwise Differentiation

Используем black-box SDE солвер и дифференцируем через него. По теории стохастических потоков (Kunita, 1984) производные $\partial X_t / \partial \beta$ и $\partial X_t / \partial \theta$ сами решают SDE и могут быть получены одним forward проходом:

$$d\frac{\partial X_t}{\partial \beta_i} = \left(\frac{\partial b_s}{\partial x}\frac{\partial X_s}{\partial \beta_i} + \frac{\partial \tilde{b}_s}{\partial \beta_i}\right)dt + \sum_\ell \frac{\partial \sigma_{s,\ell}}{\partial x}\frac{\partial X_s}{\partial \beta_i}\,dW_s^\ell$$

## 6. Связь со score-based моделями

При фиксированном $\sigma = I_d$ оптимальный drift Neural SDE — дрейф Фёллмера — совпадает со score function:

$$b^*(x, t) = \nabla_x \log p_t(x)$$

Это в точности reverse-time SDE из Score-based Generative Models (Song et al., 2020):

$$dx = \left[-\frac{1}{2}\beta(t)x - \beta(t)\nabla_x \log p_t(x)\right]dt + \sqrt{\beta(t)}\,d\bar{W}$$

Разница в том что в score-based моделях forward pass фиксирован, а score function учится явно через score matching. В Neural SDE и forward и backward процессы обучаются совместно через ELBO.
