# ML Gini Index

Репозиторий содержит в файле utils.py набор функций для расчёта модельного коэффициента Джини, оценки его асимптотической дисперсии, построения доверительного интервала и сравнения двух нормированных коэффициентов Джини с помощью асимптотического критерия Вальда.
В файле experiment.ipynb содержится численный эксперимент с использованием данных функций, описанный в ВКР
## Основная идея

Пусть есть неотрицательная целевая величина `y` и модельная оценка её условного математического ожидания `mu_hat`.

Модельный коэффициент Джини измеряет, насколько хорошо модель ранжирует наблюдения по сравнению с идеальным ранжированием по самой целевой величине `y`.

Эмпирически он задаётся как отношение двух площадей:

```math
\widehat G(y, \hat\mu) = \frac{A}{B},
```

где:

- `A` — площадь между диагональю и эмпирической кривой Лоренца, построенной после сортировки наблюдений по `mu_hat`;
- `B` — площадь между диагональю и эмпирической кривой Лоренца, построенной после сортировки наблюдений по `y`.

Если модель ранжирует наблюдения почти так же, как истинные значения `y`, относительный Джини будет близок к `1`.

---

## Функции

### `relative_gini_value`

```python
relative_gini_value(y, mu_hat)
```

Вычисляет эмпирический модельный коэффициент Джини.

#### Аргументы

- `y` — массив истинных значений целевой величины.
- `mu_hat` — массив модельных оценок / предсказаний / score-функции.

#### Возвращает

- `float` — значение модельного коэффициента Джини.

#### Пример

```python
gini = relative_gini_value(y, mu_hat)
print(gini)
```

---

### `relative_gini_variance`

```python
relative_gini_variance(y, mu_hat)
```

Вычисляет состоятельную оценку асимптотической дисперсии модельного коэффициента Джини.

Если

```math
\sqrt n(\widehat G_n - G) \Rightarrow N(0, \sigma^2),
```

то функция возвращает оценку именно для `sigma^2`.

Дисперсия самой оценки `relative_gini_value(y, mu_hat)` асимптотически равна:

```math
\frac{\widehat\sigma^2}{n}.
```

#### Аргументы

- `y` — массив истинных значений.
- `mu_hat` — массив модельных оценок.

#### Возвращает

- `float` — оценка асимптотической дисперсии `sigma2_hat`.

#### Пример

```python
sigma2_hat = relative_gini_variance(y, mu_hat)
standard_error = (sigma2_hat / len(y)) ** 0.5
```

---

### `relative_gini_confint`

```python
relative_gini_confint(y, mu_hat, alpha=0.05)
```

Строит асимптотический доверительный интервал для модельного коэффициента Джини.

Интервал имеет вид:

```math
\widehat G_n \pm z_{1-\alpha/2}\frac{\widehat\sigma}{\sqrt n}.
```

#### Аргументы

- `y` — массив истинных значений.
- `mu_hat` — массив модельных оценок.
- `alpha` — уровень значимости. По умолчанию `0.05`, что соответствует 95% доверительному интервалу.

#### Возвращает

- `(lower, upper)` — нижняя и верхняя границы доверительного интервала.

#### Пример

```python
lower, upper = relative_gini_confint(y, mu_hat, alpha=0.05)

print(lower, upper)
```

---

### `gini_difference_wald_test`

```python
gini_difference_wald_test(y, score_a, score_b, score_c, alpha=0.05)
```

Проводит асимптотический критерий Вальда для проверки равенства двух модельных коэффициентов Джини.

Проверяется гипотеза:

```math
H_0:
\frac{Gini_A}{Gini_C}
-
\frac{Gini_B}{Gini_C}
= 0.
```

То есть функция проверяет, различаются ли две модели по нормированному коэффициенту Джини.

Обычно:

- `score_a` — предсказания первой модели;
- `score_b` — предсказания второй модели;
- `score_c` — нормирующий score, часто `y`.

#### Аргументы

- `y` — массив истинных значений.
- `score_a` — score первой модели.
- `score_b` — score второй модели.
- `score_c` — score для нормировки.
- `alpha` — уровень значимости теста.

#### Возвращает

Словарь с основными результатами:

- `reject` — отвергается ли гипотеза `H0`;
- `p_value` — p-value теста;
- `t_stat` — значение статистики Вальда;
- `critical_value` — критическое значение;
- `delta_hat` — оценка разности нормированных Джини;
- `tau2_hat` — оценка асимптотической дисперсии разности;
- `gini_A`, `gini_B`, `gini_C` — обычные коэффициенты Джини;
- `normalized_gini_A`, `normalized_gini_B` — нормированные коэффициенты Джини.

#### Пример

```python
result = gini_difference_wald_test(
    y=y,
    score_a=mu_hat_model_1,
    score_b=mu_hat_model_2,
    score_c=y,
    alpha=0.05,
)

print("Relative Gini A:", result["normalized_gini_A"])
print("Relative Gini B:", result["normalized_gini_B"])
print("Delta:", result["delta_hat"])
print("p-value:", result["p_value"])
print("Reject H0:", result["reject"])
```

---

## Пример полного использования

```python
import numpy as np

rng = np.random.default_rng(42)

n = 1000

mu_hat = rng.lognormal(mean=2.0, sigma=0.5, size=n)
y = mu_hat * rng.gamma(shape=2.0, scale=0.5, size=n)

gini = relative_gini_value(y, mu_hat)
sigma2_hat = relative_gini_variance(y, mu_hat)
lower, upper = relative_gini_confint(y, mu_hat, alpha=0.05)

print("Relative Gini:", gini)
print("Asymptotic variance estimate:", sigma2_hat)
print("95% confidence interval:", (lower, upper))
```

---

## Интерпретация

- `relative_gini_value` близок к `1`: модель хорошо ранжирует наблюдения.
- `relative_gini_value` близок к `0`: модель ранжирует слабо.
- `relative_gini_value` меньше `0`: модель ранжирует хуже случайного или в обратном направлении.
- Малое `p_value` в `gini_difference_wald_test` означает статистически значимое различие между двумя нормированными коэффициентами Джини.

---

## Требования

Минимальные зависимости:

```bash
numpy
```

Если используются дополнительные симуляции, графики или эксперименты, могут понадобиться:

```bash
scipy
matplotlib
pandas
```
