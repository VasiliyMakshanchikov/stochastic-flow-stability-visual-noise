import numpy as np
from scipy import stats


def compute_poisson_parameters(event_times: np.ndarray, check_stationarity: bool = True) -> dict:
    """
    Оценивает параметры Пуассоновского процесса по временам событий.
    """
    if len(event_times) < 2:
        raise ValueError("Need at least 2 events to estimate parameters")

    event_times = np.sort(event_times)
    interarrival_times = np.diff(event_times)
    
    # MLE оценка
    mean_interarrival = np.mean(interarrival_times)
    lam = 1.0 / mean_interarrival
    
    total_time = event_times[-1] - event_times[0]
    n_events = len(event_times)
    
    # Дополнительные проверки
    results = {
        "lambda": lam,
        "lambda_std": lam / np.sqrt(n_events),  # стандартная ошибка
        "mean_interarrival": mean_interarrival,
        "n_events": n_events,
        "total_time": total_time,
        "interarrival_std": np.std(interarrival_times),
        "cv": np.std(interarrival_times) / mean_interarrival,  # коэфф. вариации
    }
    
    if check_stationarity:
        # Тест на стационарность: разбиваем на интервалы
        n_intervals = min(10, n_events // 10)
        if n_intervals >= 2:
            interval_size = total_time / n_intervals
            events_per_interval = []
            for i in range(n_intervals):
                t_start = event_times[0] + i * interval_size
                t_end = t_start + interval_size
                count = np.sum((event_times >= t_start) & (event_times < t_end))
                events_per_interval.append(count)
            
            # Проверка на пуассоновость через дисперсию
            var_counts = np.var(events_per_interval)
            mean_counts = np.mean(events_per_interval)
            results["dispersion_index"] = var_counts / mean_counts
            results["stationarity_pvalue"] = 1 - stats.chi2.cdf(
                (n_intervals - 1) * var_counts / mean_counts, 
                n_intervals - 1
            ) if mean_counts > 0 else 1.0
    
    return results

def test_poisson_assumptions(interarrival_times: np.ndarray, title: str = ""):
    """
    Тестирование свойств пуассоновского потока
    """
    print(f"\n=== Проверка свойств для {title} ===")
    
    # 1. Экспоненциальность интервалов
    if len(interarrival_times) > 5:
        # Тест Колмогорова-Смирнова
        ks_stat, ks_pvalue = stats.kstest(
            interarrival_times, 
            'expon', 
            args=(0, np.mean(interarrival_times))
        )
        print(f"KS test for exponential: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f}")
        
        if ks_pvalue > 0.05:
            print("  ✓ Интервалы распределены экспоненциально (p > 0.05)")
        else:
            print("  ✗ Интервалы НЕ распределены экспоненциально (p ≤ 0.05)")
    
    # 2. Коэффициент вариации
    cv = np.std(interarrival_times) / np.mean(interarrival_times)
    print(f"Coefficient of variation: {cv:.4f} (ожидается ~1 для пуассона)")
    
    # 3. Отсутствие корреляции
    if len(interarrival_times) > 10:
        autocorr = np.corrcoef(interarrival_times[:-1], interarrival_times[1:])[0,1]
        print(f"Autocorrelation: {autocorr:.4f} (ожидается ~0)")
    
    return ks_pvalue if 'ks_pvalue' in locals() else None


def compare_poisson_parameters(events_true: np.ndarray,
                               events_noisy: np.ndarray) -> dict:
    """
    Сравнивает параметры Пуассоновского процесса
    для чистых и зашумленных данных.

    Parameters
    ----------
    events_true : np.ndarray
        Истинные времена событий

    events_noisy : np.ndarray
        Времена событий после добавления шума

    Returns
    -------
    dict
        Словарь с результатами сравнения:
        - lambda_true
        - lambda_noisy
        - delta_lambda
    """

    params_true = compute_poisson_parameters(events_true)
    params_noisy = compute_poisson_parameters(events_noisy)

    lambda_true = params_true["lambda"]
    lambda_noisy = params_noisy["lambda"]

    delta_lambda = abs(lambda_noisy - lambda_true)

    return {
        "lambda_true": lambda_true,
        "lambda_noisy": lambda_noisy,
        "delta_lambda": delta_lambda
    }

def generate_poisson_events(lam: float, T: float) -> np.ndarray:
    """
    Генерирует события Пуассоновского процесса.

    Parameters
    ----------
    lam : float
        Интенсивность потока (λ)

    T : float
        Общее время наблюдения

    Returns
    -------
    np.ndarray
        Массив времён событий
    """

    times = []
    t = 0

    while t < T:
        dt = np.random.exponential(1 / lam)
        t += dt
        if t < T:
            times.append(t)

    return np.array(times)

def add_noise_to_events(events: np.ndarray,
                        miss_prob: float = 0.1,
                        false_rate: float = 0.1,
                        T: float = 100.0) -> np.ndarray:
    """
    Добавляет шум к событиям:
    - пропуски (false negatives)
    - ложные события (false positives)
    """

    # Пропуски
    kept = [e for e in events if np.random.rand() > miss_prob]

    # Ложные события
    n_false = int(len(events) * false_rate)
    false_events = np.random.uniform(0, T, n_false)

    noisy = np.sort(np.concatenate([kept, false_events]))

    return noisy