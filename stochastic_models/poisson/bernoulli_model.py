import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def compute_bernoulli_parameters(event_frames: np.ndarray, total_frames: int, check_stationarity: bool = True) -> dict:
    """
    Оценивает параметры процесса Бернулли по номерам кадров событий.
    
    Процесс Бернулли - это дискретный аналог пуассоновского процесса.
    В каждом кадре событие происходит с вероятностью p.
    Интервалы между событиями имеют геометрическое распределение.
    
    Parameters
    ----------
    event_frames : np.ndarray
        Массив номеров кадров, в которых произошли события.
        Должен быть отсортирован по возрастанию.
    total_frames : int
        Общее количество кадров в видео.
    check_stationarity : bool
        Проверять ли стационарность (постоянство p во времени).
    
    Returns
    -------
    dict
        Словарь с параметрами
    """
    if len(event_frames) < 2:
        raise ValueError("Need at least 2 events to estimate parameters")
    
    event_frames = np.sort(event_frames)
    
    # MLE оценка вероятности события в кадре
    p = len(event_frames) / total_frames
    
    # Интервалы между событиями в кадрах
    intervals_frames = np.diff(event_frames)
    mean_interval_frames = np.mean(intervals_frames)
    
    # Стандартная ошибка (для биномиального распределения)
    p_std = np.sqrt(p * (1 - p) / total_frames)
    
    # Базовые результаты
    results = {
        "p": p,
        "p_std": p_std,
        "p_ci_lower": p - 1.96 * p_std,  # 95% доверительный интервал
        "p_ci_upper": p + 1.96 * p_std,
        "mean_interval_frames": mean_interval_frames,
        "n_events": len(event_frames),
        "total_frames": total_frames,
        "interval_std_frames": np.std(intervals_frames),
        "cv": np.std(intervals_frames) / mean_interval_frames,
        "theoretical_mean_interval": 1 / p if p > 0 else np.inf,
    }
    
    # Проверка стационарности
    if check_stationarity and len(event_frames) >= 20:
        n_intervals = min(10, len(event_frames) // 5)
        if n_intervals >= 2:
            frames_per_interval = total_frames // n_intervals
            events_per_interval = []
            
            for i in range(n_intervals):
                start_frame = i * frames_per_interval
                end_frame = (i + 1) * frames_per_interval
                count = np.sum((event_frames >= start_frame) & (event_frames < end_frame))
                events_per_interval.append(count)
            
            expected_per_interval = len(event_frames) / n_intervals
            chi2_stat = np.sum((np.array(events_per_interval) - expected_per_interval) ** 2 / expected_per_interval)
            df = n_intervals - 1
            results["stationarity_pvalue"] = 1 - stats.chi2.cdf(chi2_stat, df)
            results["events_per_interval"] = events_per_interval
    
    return results


def test_bernoulli_assumptions(event_frames: np.ndarray, total_frames: int, title: str = ""):
    """
    Тестирование свойств процесса Бернулли.
    """
    print(f"\n=== Проверка свойств процесса Бернулли для {title} ===")
    
    if len(event_frames) < 2:
        print("  Недостаточно событий для проверки")
        return
    
    intervals = np.diff(event_frames)
    p = len(event_frames) / total_frames
    
    # 1. Проверка на геометрическое распределение
    if len(intervals) > 5:
        observed_counts = np.bincount(intervals)
        max_interval = max(intervals)
        
        theoretical_probs = stats.geom.pmf(np.arange(1, max_interval + 1), p)
        theoretical_probs = theoretical_probs / np.sum(theoretical_probs)
        
        expected_counts = theoretical_probs * len(intervals)
        
        # Объединяем категории с ожиданием < 5
        min_expected = 5
        merged_observed = []
        merged_expected = []
        current_obs = 0
        current_exp = 0
        
        for i in range(len(theoretical_probs)):
            current_obs += observed_counts[i+1] if i+1 < len(observed_counts) else 0
            current_exp += expected_counts[i]
            
            if current_exp >= min_expected or i == len(theoretical_probs) - 1:
                if current_exp > 0:
                    merged_observed.append(current_obs)
                    merged_expected.append(current_exp)
                current_obs = 0
                current_exp = 0
        
        if len(merged_observed) > 1:
            chi2_stat, chi2_pvalue = stats.chisquare(merged_observed, merged_expected)
            print(f"  Хи-квадрат тест на геометрическое распределение:")
            print(f"    statistic={chi2_stat:.4f}, p-value={chi2_pvalue:.4f}")
            
            if chi2_pvalue > 0.05:
                print("    ✓ Интервалы соответствуют геометрическому распределению")
            else:
                print("    ✗ Интервалы НЕ соответствуют геометрическому распределению")
    
    # 2. Коэффициент вариации
    cv = np.std(intervals) / np.mean(intervals)
    print(f"\n  Коэффициент вариации интервалов: {cv:.4f}")
    
    if 0.8 <= cv <= 1.2:
        print("    ✓ CV близок к 1 (соответствует геометрическому распределению)")
    elif cv > 1.2:
        print("    ⚠️ CV > 1.2 (сверхдисперсия, события кластеризуются)")
    else:
        print("    ⚠️ CV < 0.8 (недодисперсия, события слишком регулярны)")
    
    # 3. Проверка независимости
    if len(intervals) > 10:
        autocorr = np.corrcoef(intervals[:-1], intervals[1:])[0, 1] if len(intervals) > 1 else 0
        print(f"\n  Автокорреляция интервалов: {autocorr:.4f}")
        if abs(autocorr) < 0.2:
            print("    ✓ Слабая корреляция (интервалы независимы)")
        else:
            print("    ⚠️ Есть значительная корреляция")
    
    # 4. Индекс дисперсии
    variance = np.var(intervals)
    mean_interval = np.mean(intervals)
    dispersion_index = variance / mean_interval if mean_interval > 0 else 0
    
    print(f"\n  Индекс дисперсии интервалов: {dispersion_index:.4f}")
    print(f"    (1 = геометрическое, >1 = кластеризация, <1 = регулярность)")


def visualize_bernoulli_process(event_frames: np.ndarray, total_frames: int, title: str = "Процесс Бернулли"):
    """
    Визуализация процесса Бернулли.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    p = len(event_frames) / total_frames
    intervals = np.diff(event_frames)
    
    # 1. Бинарный временной ряд (первые 200 кадров) - используем plot вместо stem
    ax1 = axes[0, 0]
    n_show = min(200, total_frames)
    binary_series = np.zeros(n_show)
    event_mask = event_frames[event_frames < n_show]
    binary_series[event_mask] = 1
    
    # Рисуем точки и линии отдельно (избегаем stem полностью)
    ax1.plot(np.arange(n_show), binary_series, 'bo', markersize=3, alpha=0.7)
    # Рисуем вертикальные линии только для событий
    for i in event_mask:
        ax1.vlines(i, 0, 1, colors='blue', linewidth=0.5, alpha=0.5)
    
    ax1.set_xlabel('Номер кадра')
    ax1.set_ylabel('Событие (0/1)')
    ax1.set_title(f'{title} - бинарный ряд (первые {n_show} кадров)')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # 2. Гистограмма интервалов
    ax2 = axes[0, 1]
    if len(intervals) > 0:
        max_interval = min(max(intervals), 50)
        intervals_filtered = intervals[intervals <= max_interval]
        
        ax2.hist(intervals_filtered, bins=np.arange(0.5, max_interval + 1.5, 1), 
                density=True, alpha=0.7, color='blue', edgecolor='black', label='Эмпирические')
        
        # Теоретическое геометрическое распределение
        x = np.arange(1, max_interval + 1)
        geom_pmf = stats.geom.pmf(x, p)
        ax2.plot(x, geom_pmf, 'r-', linewidth=2, label=f'Геометрическое (p={p:.3f})')
        
        ax2.set_xlabel('Интервал между событиями (кадры)')
        ax2.set_ylabel('Вероятность')
        ax2.set_title('Распределение интервалов')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Кумулятивная функция
    ax3 = axes[1, 0]
    cumulative = np.arange(1, len(event_frames) + 1)
    ax3.plot(event_frames, cumulative, 'b-', linewidth=2, label='Реальные события')
    
    # Теоретическая линия
    theoretical = p * event_frames
    ax3.plot(event_frames, theoretical, 'r--', alpha=0.7, label=f'Теоретическая (p={p:.3f})')
    
    ax3.set_xlabel('Номер кадра')
    ax3.set_ylabel('Накопленное число событий')
    ax3.set_title('Кумулятивная функция событий')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q plot для геометрического распределения
    ax4 = axes[1, 1]
    if len(intervals) > 5:
        n = len(intervals)
        theoretical_quantiles = stats.geom.ppf(np.arange(0.5, n + 0.5) / n, p)
        empirical_quantiles = np.sort(intervals)
        
        ax4.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.6)
        max_val = max(max(theoretical_quantiles), max(empirical_quantiles))
        ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        ax4.set_xlabel('Теоретические квантили (геометрическое)')
        ax4.set_ylabel('Эмпирические квантили')
        ax4.set_title('Q-Q plot (геометрическое распределение)')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} (p={p:.4f}, всего кадров={total_frames})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()