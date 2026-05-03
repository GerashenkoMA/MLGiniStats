import numpy as np
from math import sqrt
from statistics import NormalDist


def relative_gini(y, mu_hat, *, check_nonnegative=True, on_zero_denominator="raise"):
    """
    Эмпирический относительный коэффициент Джини из определения 6.

    Parameters
    ----------
    y : array-like, shape (n,)
        Истинные значения целевой величины Y. В определении предполагается y >= 0.
    mu_hat : array-like, shape (n,)
        Оценки / score / предсказания \hat{mu}(x_i).
    check_nonnegative : bool, default=True
        Если True, проверяет, что все y_i >= 0.
    on_zero_denominator : {"raise", "nan"}, default="raise"
        Что делать, если знаменатель B равен нулю.

    Returns
    -------
    float
        Эмпирический относительный коэффициент Джини G_hat(y, mu_hat).

    Notes
    -----
    G_hat = A / B,

    где

        A = 1/2 - sum_i ((C_i + C_{i-1}) / 2) * (alpha_i - alpha_{i-1}),

        B = 1/2 - sum_i ((L_i + L_{i-1}) / 2) * (alpha_i - alpha_{i-1}).

    Здесь alpha_i = i / n.
    """

    y = np.asarray(y, dtype=float)
    mu_hat = np.asarray(mu_hat, dtype=float)

    if y.ndim != 1 or mu_hat.ndim != 1:
        raise ValueError("y and mu_hat must be one-dimensional arrays.")

    if len(y) != len(mu_hat):
        raise ValueError("y and mu_hat must have the same length.")

    n = len(y)
    if n == 0:
        raise ValueError("y and mu_hat must be non-empty.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")

    if not np.all(np.isfinite(mu_hat)):
        raise ValueError("mu_hat contains NaN or infinite values.")

    if check_nonnegative and np.any(y < 0):
        raise ValueError("Definition assumes y >= 0, but y contains negative values.")

    total_y = np.sum(y)
    if total_y <= 0:
        raise ValueError("sum(y) must be positive.")

    def lorenz_area(sorted_y):
        """
        Площадь под эмпирической кривой Лоренца методом трапеций.
        """
        cumulative_share = np.cumsum(sorted_y) / total_y

        # Добавляем значение в alpha_0 = 0: C_0 = 0 или L_0 = 0
        curve = np.concatenate(([0.0], cumulative_share))

        # alpha_i - alpha_{i-1} = 1 / n
        area = np.sum((curve[1:] + curve[:-1]) / 2) / n
        return area

    # Числитель A: сортировка y по возрастанию mu_hat
    order_by_mu = np.argsort(mu_hat, kind="mergesort")
    y_sorted_by_mu = y[order_by_mu]
    area_C = lorenz_area(y_sorted_by_mu)
    A = 0.5 - area_C

    # Знаменатель B: сортировка y по возрастанию y
    y_sorted = np.sort(y, kind="mergesort")
    area_L = lorenz_area(y_sorted)
    B = 0.5 - area_L

    if np.isclose(B, 0.0):
        if on_zero_denominator == "raise":
            raise ZeroDivisionError(
                "Denominator B is zero or too close to zero. "
                "Relative Gini is not well-defined."
            )
        if on_zero_denominator == "nan":
            return np.nan
        raise ValueError("on_zero_denominator must be either 'raise' or 'nan'.")

    return A / B


import numpy as np
from math import sqrt
from statistics import NormalDist


def _validate_y_score(y, score, *, check_nonnegative=True):
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)

    if y.ndim != 1 or score.ndim != 1:
        raise ValueError("y and score must be one-dimensional arrays.")

    if len(y) != len(score):
        raise ValueError("y and score must have the same length.")

    if len(y) == 0:
        raise ValueError("y and score must be non-empty.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")

    if not np.all(np.isfinite(score)):
        raise ValueError("score contains NaN or infinite values.")

    if check_nonnegative and np.any(y < 0):
        raise ValueError("The relative Gini setup assumes y >= 0.")

    if np.sum(y) <= 0:
        raise ValueError("sum(y) must be positive.")

    return y, score


def _empirical_F_and_FL(y, score):
    """
    Для каждого i считает

        F_hat(score_i)  = 1/n * sum_j 1{score_j <= score_i},
        FL_hat(score_i) = sum_j y_j 1{score_j <= score_i} / sum_j y_j.

    Используется <=, как в определении F_L(s).
    """
    n = len(y)
    total_y = np.sum(y)

    order = np.argsort(score, kind="mergesort")
    score_sorted = score[order]
    y_sorted = y[order]

    cum_y_sorted = np.cumsum(y_sorted)

    # pos_i = количество элементов score_j <= score_i
    pos = np.searchsorted(score_sorted, score, side="right")

    F_hat = pos / n
    FL_hat = cum_y_sorted[pos - 1] / total_y

    return F_hat, FL_hat


def _h1_hat(y, score):
    """
    Эмпирический аналог проекции Хеффдинга:

        h1_hat_i = 1/2 * (y_bar * FL_hat(score_i)
                          + y_i * (1 - F_hat(score_i))).
    """
    y_bar = np.mean(y)
    F_hat, FL_hat = _empirical_F_and_FL(y, score)

    return 0.5 * (y_bar * FL_hat + y * (1.0 - F_hat))


def _ordinary_gini_from_h1(y, h1):
    """
    Обычный эмпирический Gini для заданной score-функции:

        Gini_hat = (y_bar - 2 h1_bar) / y_bar.
    """
    y_bar = np.mean(y)
    h1_bar = np.mean(h1)
    return (y_bar - 2.0 * h1_bar) / y_bar


def relative_gini_value(y, mu_hat, *, check_nonnegative=True):
    """
    Эмпирический относительный коэффициент Джини:

        G_hat = Gini_hat_A / Gini_hat_B,

    где A соответствует score = mu_hat,
    а B соответствует score = y.
    """
    y, mu_hat = _validate_y_score(
        y,
        mu_hat,
        check_nonnegative=check_nonnegative,
    )

    h1A = _h1_hat(y, mu_hat)
    h1B = _h1_hat(y, y)

    gini_A = _ordinary_gini_from_h1(y, h1A)
    gini_B = _ordinary_gini_from_h1(y, h1B)

    if np.isclose(gini_B, 0.0):
        raise ZeroDivisionError(
            "Gini_B is zero or too close to zero. "
            "Relative Gini is not well-defined."
        )

    return gini_A / gini_B


def relative_gini_variance(
    y,
    mu_hat,
    *,
    check_nonnegative=True,
    return_components=False,
    clip_negative=True,
    negative_tol=1e-12,
):
    """
    Оценка асимптотической дисперсии относительного Джини из теоремы 2.

    Возвращает sigma2_hat, где

        sqrt(n) * (G_hat - G) -> N(0, sigma^2).

    Поэтому стандартная ошибка самого G_hat равна

        sqrt(sigma2_hat / n).

    Parameters
    ----------
    y : array-like, shape (n,)
        Истинные значения Y.
    mu_hat : array-like, shape (n,)
        Оценки / score / предсказания \\hat{mu}.
    check_nonnegative : bool, default=True
        Проверять ли y >= 0.
    return_components : bool, default=False
        Если True, возвращает также словарь с промежуточными величинами.
    clip_negative : bool, default=True
        Если из-за численной ошибки sigma2_hat получилось слегка отрицательным,
        заменить его на 0.
    negative_tol : float, default=1e-12
        Допуск для слегка отрицательных значений sigma2_hat.

    Returns
    -------
    float or tuple[float, dict]
        sigma2_hat или (sigma2_hat, components).
    """
    y, mu_hat = _validate_y_score(
        y,
        mu_hat,
        check_nonnegative=check_nonnegative,
    )

    n = len(y)
    y_bar = np.mean(y)

    # h1A для score S_A = mu_hat
    # h1B для score S_B = y
    h1A_i = _h1_hat(y, mu_hat)
    h1B_i = _h1_hat(y, y)

    h1A = np.mean(h1A_i)
    h1B = np.mean(h1B_i)

    # Моментные оценки ковариаций из теоремы 2
    Sigma_hA = np.mean(h1A_i ** 2) - h1A ** 2
    Sigma_hB = np.mean(h1B_i ** 2) - h1B ** 2
    Sigma_y = np.mean(y ** 2) - y_bar ** 2

    Sigma_hA_y = np.mean(h1A_i * y) - h1A * y_bar
    Sigma_hB_y = np.mean(h1B_i * y) - h1B * y_bar
    Sigma_hA_hB = np.mean(h1A_i * h1B_i) - h1A * h1B

    # Обычные эмпирические Gini_A и Gini_B
    gini_A = (y_bar - 2.0 * h1A) / y_bar
    gini_B = (y_bar - 2.0 * h1B) / y_bar

    if np.isclose(gini_B, 0.0):
        raise ZeroDivisionError(
            "Gini_B is zero or too close to zero. "
            "Variance of relative Gini is not well-defined."
        )

    # Оценки sigma_A^2, sigma_B^2 и sigma_AB
    sigma2_A = (4.0 / y_bar ** 2) * (
        4.0 * Sigma_hA
        + (h1A ** 2 / y_bar ** 2) * Sigma_y
        - 4.0 * (h1A / y_bar) * Sigma_hA_y
    )

    sigma2_B = (4.0 / y_bar ** 2) * (
        4.0 * Sigma_hB
        + (h1B ** 2 / y_bar ** 2) * Sigma_y
        - 4.0 * (h1B / y_bar) * Sigma_hB_y
    )

    sigma_AB = (4.0 / y_bar ** 2) * (
        4.0 * Sigma_hA_hB
        - 2.0 * (h1B / y_bar) * Sigma_hA_y
        - 2.0 * (h1A / y_bar) * Sigma_hB_y
        + (h1A * h1B / y_bar ** 2) * Sigma_y
    )

    # Дельта-метод для отношения Gini_A / Gini_B
    sigma2_rel = (
        sigma2_A / gini_B ** 2
        - 2.0 * gini_A * sigma_AB / gini_B ** 3
        + (gini_A ** 2) * sigma2_B / gini_B ** 4
    )

    if sigma2_rel < 0:
        if clip_negative and sigma2_rel >= -negative_tol:
            sigma2_rel = 0.0
        else:
            raise ArithmeticError(
                f"sigma2_hat is negative: {sigma2_rel}. "
                "This may indicate numerical instability, small sample issues, "
                "or violation of assumptions."
            )

    if not return_components:
        return sigma2_rel

    components = {
        "n": n,
        "y_bar": y_bar,
        "h1A": h1A,
        "h1B": h1B,
        "gini_A": gini_A,
        "gini_B": gini_B,
        "relative_gini": gini_A / gini_B,
        "Sigma_hA": Sigma_hA,
        "Sigma_hB": Sigma_hB,
        "Sigma_y": Sigma_y,
        "Sigma_hA_y": Sigma_hA_y,
        "Sigma_hB_y": Sigma_hB_y,
        "Sigma_hA_hB": Sigma_hA_hB,
        "sigma2_A": sigma2_A,
        "sigma2_B": sigma2_B,
        "sigma_AB": sigma_AB,
        "sigma2_relative": sigma2_rel,
    }

    return sigma2_rel, components


def relative_gini_confint(
    y,
    mu_hat,
    *,
    alpha=0.05,
    check_nonnegative=True,
    return_components=False,
):
    """
    Асимптотический доверительный интервал уровня 1 - alpha
    для относительного коэффициента Джини из теоремы 4:

        G_hat +- z_{1-alpha/2} * sigma_hat / sqrt(n).

    Parameters
    ----------
    y : array-like, shape (n,)
        Истинные значения Y.
    mu_hat : array-like, shape (n,)
        Оценки / score / предсказания \\hat{mu}.
    alpha : float, default=0.05
        Уровень значимости. Для 95% интервала alpha = 0.05.
    check_nonnegative : bool, default=True
        Проверять ли y >= 0.
    return_components : bool, default=False
        Если True, возвращает также промежуточные величины.

    Returns
    -------
    tuple or dict
        Если return_components=False:

            (lower, upper)

        Если return_components=True:

            {
                "relative_gini": ...,
                "sigma2_hat": ...,
                "sigma_hat": ...,
                "standard_error": ...,
                "z": ...,
                "alpha": ...,
                "confidence_level": ...,
                "lower": ...,
                "upper": ...,
                "components": ...
            }
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    y, mu_hat = _validate_y_score(
        y,
        mu_hat,
        check_nonnegative=check_nonnegative,
    )

    n = len(y)

    sigma2_hat, components = relative_gini_variance(
        y,
        mu_hat,
        check_nonnegative=check_nonnegative,
        return_components=True,
    )

    g_hat = components["relative_gini"]

    sigma_hat = sqrt(sigma2_hat)
    standard_error = sigma_hat / sqrt(n)

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    lower = g_hat - z * standard_error
    upper = g_hat + z * standard_error

    if not return_components:
        return lower, upper

    return {
        "relative_gini": g_hat,
        "sigma2_hat": sigma2_hat,
        "sigma_hat": sigma_hat,
        "standard_error": standard_error,
        "z": z,
        "alpha": alpha,
        "confidence_level": 1.0 - alpha,
        "lower": lower,
        "upper": upper,
        "components": components,
    }


def _validate_inputs(y, *scores, check_nonnegative=True):
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    if len(y) == 0:
        raise ValueError("y must be non-empty.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")

    if check_nonnegative and np.any(y < 0):
        raise ValueError("The Gini setup assumes y >= 0.")

    if np.mean(y) <= 0:
        raise ValueError("mean(y) must be positive.")

    scores_out = []

    for score in scores:
        score = np.asarray(score, dtype=float)

        if score.ndim != 1:
            raise ValueError("All scores must be one-dimensional arrays.")

        if len(score) != len(y):
            raise ValueError("All scores must have the same length as y.")

        if not np.all(np.isfinite(score)):
            raise ValueError("Some score contains NaN or infinite values.")

        scores_out.append(score)

    return (y, *scores_out)


def _empirical_F_and_FL(y, score):
    """
    Для каждого i считает:

        F_hat(score_i)  = 1/n * sum_j 1{score_j <= score_i},
        FL_hat(score_i) = sum_j y_j 1{score_j <= score_i} / sum_j y_j.

    Используем <=, как в определении из файла.
    """
    n = len(y)
    total_y = np.sum(y)

    order = np.argsort(score, kind="mergesort")
    score_sorted = score[order]
    y_sorted = y[order]

    cum_y_sorted = np.cumsum(y_sorted)

    # pos[i] = количество score_j <= score_i
    pos = np.searchsorted(score_sorted, score, side="right")

    F_hat = pos / n
    FL_hat = cum_y_sorted[pos - 1] / total_y

    return F_hat, FL_hat


def _h1_hat(y, score):
    """
    Эмпирический аналог проекции Хеффдинга:

        h1_hat_i = 1/2 * (y_bar * FL_hat(score_i)
                          + y_i * (1 - F_hat(score_i))).
    """
    y_bar = np.mean(y)
    F_hat, FL_hat = _empirical_F_and_FL(y, score)

    h1 = 0.5 * (
        y_bar * FL_hat
        + y * (1.0 - F_hat)
    )

    return h1


def _ordinary_gini_from_h1(y, h1):
    """
    Обычный эмпирический коэффициент Джини:

        Gini_hat = (y_bar - 2 * h1_bar) / y_bar.
    """
    y_bar = np.mean(y)
    h1_bar = np.mean(h1)

    return (y_bar - 2.0 * h1_bar) / y_bar


def _sigma_jk_hat(y, h1_j, h1_k):
    """
    Оценка ковариационного элемента sigma_JK.

    Для J = K это даёт sigma_J^2.
    Для J != K это даёт sigma_JK.

    Формула соответствует теореме 5/6:

        sigma_JK =
        4 / y_bar^2 *
        (
            4 Sigma_hJ,hK
            - 2 hK/y_bar Sigma_hJ,y
            - 2 hJ/y_bar Sigma_hK,y
            + hJ hK / y_bar^2 Sigma_y
        ).
    """
    y_bar = np.mean(y)

    h1_j_bar = np.mean(h1_j)
    h1_k_bar = np.mean(h1_k)

    sigma_hj_hk = np.mean(h1_j * h1_k) - h1_j_bar * h1_k_bar
    sigma_hj_y = np.mean(h1_j * y) - h1_j_bar * y_bar
    sigma_hk_y = np.mean(h1_k * y) - h1_k_bar * y_bar
    sigma_y = np.mean(y ** 2) - y_bar ** 2

    sigma_jk = (4.0 / y_bar ** 2) * (
        4.0 * sigma_hj_hk
        - 2.0 * (h1_k_bar / y_bar) * sigma_hj_y
        - 2.0 * (h1_j_bar / y_bar) * sigma_hk_y
        + (h1_j_bar * h1_k_bar / y_bar ** 2) * sigma_y
    )

    return sigma_jk


def gini_difference_wald_test(
    y,
    score_a,
    score_b,
    score_c,
    *,
    alpha=0.05,
    check_nonnegative=True,
    clip_negative_tau=True,
    negative_tol=1e-12,
    return_components=True,
):
    """
    Асимптотический критерий Вальда из теоремы 7.

    Проверяется гипотеза:

        H0: Delta_ABC = 0

    против двусторонней альтернативы:

        H1: Delta_ABC != 0,

    где

        Delta_ABC = Gini_A / Gini_C - Gini_B / Gini_C.

    Parameters
    ----------
    y : array-like, shape (n,)
        Истинные значения Y.

    score_a : array-like, shape (n,)
        Score-функция S_A, например предсказания первой модели.

    score_b : array-like, shape (n,)
        Score-функция S_B, например предсказания второй модели.

    score_c : array-like, shape (n,)
        Score-функция S_C, относительно которой нормируются Gini_A и Gini_B.
        Частый выбор: score_c = y.

    alpha : float, default=0.05
        Уровень значимости теста.

    check_nonnegative : bool, default=True
        Проверять ли y >= 0.

    clip_negative_tau : bool, default=True
        Если tau2_hat слегка отрицательна из-за численной ошибки,
        заменить её на 0.

    negative_tol : float, default=1e-12
        Допуск для слегка отрицательной tau2_hat.

    return_components : bool, default=True
        Если True, возвращает подробный словарь.
        Если False, возвращает только reject.

    Returns
    -------
    dict or bool
        Если return_components=True, возвращает словарь:

            {
                "reject": ...,
                "p_value": ...,
                "t_stat": ...,
                "critical_value": ...,
                "delta_hat": ...,
                "tau2_hat": ...,
                "tau_hat": ...,
                "gini_A": ...,
                "gini_B": ...,
                "gini_C": ...,
                ...
            }

        Если return_components=False, возвращает только True/False.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    y, score_a, score_b, score_c = _validate_inputs(
        y,
        score_a,
        score_b,
        score_c,
        check_nonnegative=check_nonnegative,
    )

    n = len(y)

    # Эмпирические h1 для трёх score-функций
    h1A = _h1_hat(y, score_a)
    h1B = _h1_hat(y, score_b)
    h1C = _h1_hat(y, score_c)

    # Обычные эмпирические коэффициенты Джини
    gini_A = _ordinary_gini_from_h1(y, h1A)
    gini_B = _ordinary_gini_from_h1(y, h1B)
    gini_C = _ordinary_gini_from_h1(y, h1C)

    if np.isclose(gini_C, 0.0):
        raise ZeroDivisionError(
            "Gini_C is zero or too close to zero. "
            "The normalized difference is not well-defined."
        )

    # Delta_hat = Gini_A / Gini_C - Gini_B / Gini_C
    delta_hat = (gini_A - gini_B) / gini_C

    # Оценки элементов ковариационной матрицы
    sigma2_A = _sigma_jk_hat(y, h1A, h1A)
    sigma2_B = _sigma_jk_hat(y, h1B, h1B)
    sigma2_C = _sigma_jk_hat(y, h1C, h1C)

    sigma_AB = _sigma_jk_hat(y, h1A, h1B)
    sigma_AC = _sigma_jk_hat(y, h1A, h1C)
    sigma_BC = _sigma_jk_hat(y, h1B, h1C)

    # Оценка tau^2_ABC из теоремы 6,
    # которая используется в критерии Вальда теоремы 7
    tau2_hat = (
        (sigma2_A + sigma2_B - 2.0 * sigma_AB) / gini_C ** 2
        - 2.0
        * (gini_A - gini_B)
        * (sigma_AC - sigma_BC)
        / gini_C ** 3
        + ((gini_A - gini_B) ** 2)
        * sigma2_C
        / gini_C ** 4
    )

    if tau2_hat < 0:
        if clip_negative_tau and tau2_hat >= -negative_tol:
            tau2_hat = 0.0
        else:
            raise ArithmeticError(
                f"tau2_hat is negative: {tau2_hat}. "
                "This may indicate numerical instability, small sample issues, "
                "or violation of assumptions."
            )

    if np.isclose(tau2_hat, 0.0):
        raise ZeroDivisionError(
            "tau2_hat is zero or too close to zero. "
            "Wald statistic is not well-defined."
        )

    tau_hat = sqrt(tau2_hat)

    # Статистика Вальда
    t_stat = sqrt(n) * delta_hat / tau_hat

    normal = NormalDist()
    critical_value = normal.inv_cdf(1.0 - alpha / 2.0)

    # Двустороннее p-value
    p_value = 2.0 * (1.0 - normal.cdf(abs(t_stat)))

    reject = abs(t_stat) > critical_value

    if not return_components:
        return reject

    return {
        "reject": reject,
        "p_value": p_value,
        "t_stat": t_stat,
        "critical_value": critical_value,
        "alpha": alpha,
        "delta_hat": delta_hat,
        "tau2_hat": tau2_hat,
        "tau_hat": tau_hat,  
        "n": n,
        "gini_A": gini_A,
        "gini_B": gini_B,
        "gini_C": gini_C,
        "normalized_gini_A": gini_A / gini_C,
        "normalized_gini_B": gini_B / gini_C,
        "sigma2_A": sigma2_A,
        "sigma2_B": sigma2_B,
        "sigma2_C": sigma2_C,
        "sigma_AB": sigma_AB,
        "sigma_AC": sigma_AC,
        "sigma_BC": sigma_BC,
    }
