import numpy as np
from scipy.stats import entropy
from typing import List, Tuple, Optional, Dict, Any
import warnings


def build_transition_matrix(states: np.ndarray, max_state: Optional[int] = None) -> np.ndarray:
    """
    Строит матрицу переходов марковской цепи из последовательности состояний.

    Параметры
    ----------
    states : np.ndarray
        Одномерный массив целочисленных состояний (количество людей) для каждого момента времени.
    max_state : Optional[int], default=None
        Максимальное значение состояния, которое будет использоваться для размерности матрицы.
        Если None, то max_state = max(states). Все состояния выше max_state будут игнорироваться.

    Возвращает
    -------
    P : np.ndarray
        Квадратная матрица переходов размера (max_state+1) x (max_state+1).
        Строки нормированы, сумма по строке = 1. Если строка не содержит переходов,
        заполняется нулями (но предупреждение).
    """
    states = np.asarray(states).flatten()
    if max_state is None:
        max_state = int(np.max(states))
    
    P = np.zeros((max_state + 1, max_state + 1), dtype=np.float64)
    for t in range(len(states) - 1):
        i = int(states[t])
        j = int(states[t+1])
        if i > max_state or j > max_state:
            continue
        P[i, j] += 1

    # Нормировка с проверкой на нулевые строки
    row_sums = P.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    if np.any(zero_rows):
        warnings.warn(f"Состояния без переходов: {np.where(zero_rows)[0]}. Их строки останутся нулевыми.")
    row_sums[row_sums == 0] = 1  # избегаем деления на 0
    P = P / row_sums
    return P


def stationary_distribution(P: np.ndarray, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
    """
    Вычисляет стационарное распределение марковской цепи методом итераций (power method).

    Параметры
    ----------
    P : np.ndarray
        Квадратная стохастическая матрица переходов.
    tol : float
        Критерий остановки по изменению распределения (норма L1).
    max_iter : int
        Максимальное число итераций.

    Возвращает
    -------
    pi : np.ndarray
        Стационарное распределение (вектор-строка) суммы 1.

    Примечания
    ----------
    Для эргодических цепей сходимость гарантирована. Если цепь не эргодична,
    результат может быть не единственным. Функция выдаёт предупреждение,
    если максимальное число итераций достигнуто.
    """
    n = P.shape[0]
    pi = np.ones(n) / n  # равномерное начальное
    for _ in range(max_iter):
        pi_next = pi @ P
        delta = np.linalg.norm(pi_next - pi, ord=1)
        if delta < tol:
            return pi_next
        pi = pi_next
    warnings.warn(f"Стационарное распределение не сошлось за {max_iter} итераций. tol={delta:.2e}")
    return pi


def event_sequence(states: np.ndarray, threshold: int = 3) -> np.ndarray:
    """
    Преобразует последовательность состояний в бинарную последовательность событий:
    1, если абсолютное изменение между t и t+1 >= threshold, иначе 0.

    Параметры
    ----------
    states : np.ndarray
        Последовательность состояний (количество людей) длины T.
    threshold : int
        Минимальное изменение для регистрации события.

    Возвращает
    -------
    events : np.ndarray
        Бинарный массив длины T-1.
    """
    states = np.asarray(states).flatten()
    changes = np.abs(np.diff(states))
    events = (changes >= threshold).astype(int)
    return events


def build_binary_transition_matrix(events: np.ndarray) -> np.ndarray:
    """
    Строит матрицу переходов 2x2 для бинарной марковской цепи (событие/не событие).

    Параметры
    ----------
    events : np.ndarray
        Бинарная последовательность (0 или 1).

    Возвращает
    -------
    P_bin : np.ndarray
        Матрица размера 2x2: P[0][0] = P(0->0), P[0][1] = P(0->1) и т.д.
    """
    events = np.asarray(events).flatten()
    P = np.zeros((2, 2), dtype=np.float64)
    for t in range(len(events)-1):
        i = int(events[t])
        j = int(events[t+1])
        P[i, j] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    # Если строка нулевая, оставляем равномерное (или можно 0)
    row_sums[row_sums == 0] = 1
    P = P / row_sums
    return P


def extract_states_from_events_list(events_list: List[Dict[str, Any]], total_frames: Optional[int] = None) -> np.ndarray:
    """
    Извлекает полную временную последовательность состояний (people_count) из списка событий,
    полученного функцией get_frame_timestamps.

    Параметры
    ----------
    events_list : list of dict
        Список, где каждый словарь содержит ключи 'frame', 'people_count'.
        Предполагается, что события идут в порядке возрастания кадров.
    total_frames : Optional[int]
        Общее количество кадров (если None, берётся максимальный frame из списка).
        Если указано, массив будет дополнен значениями предыдущего состояния для отсутствующих кадров.

    Возвращает
    -------
    states : np.ndarray
        Массив states[t] = количество людей в кадре с индексом t (t от 0 до T-1).
        Кадры без изменений сохраняют предыдущее значение.
    """
    if not events_list:
        raise ValueError("events_list не должен быть пустым")
    
    # Сортируем по frame (на всякий случай)
    sorted_events = sorted(events_list, key=lambda x: x['frame'])
    
    if total_frames is None:
        total_frames = max(ev['frame'] for ev in sorted_events)
    
    states = np.zeros(total_frames, dtype=int)
    current_count = sorted_events[0]['people_count']  # начальное состояние
    
    last_frame = 1  # кадры нумеруются с 1
    for ev in sorted_events:
        frame = ev['frame']
        # Заполняем кадры от last_frame до frame-1 текущим значением
        for t in range(last_frame - 1, frame - 1):
            states[t] = current_count
        # Обновляем текущее количество в момент frame
        current_count = ev['people_count']
        states[frame - 1] = current_count
        last_frame = frame + 1
    
    # Заполняем оставшиеся кадры после последнего события
    for t in range(last_frame - 1, total_frames):
        states[t] = current_count
    
    return states


def compute_transition_metrics(P_true: np.ndarray, P_noisy: np.ndarray, metric: str = 'mae') -> float:
    """
    Вычисляет метрику различия между двумя матрицами переходов.

    Параметры
    ----------
    P_true, P_noisy : np.ndarray
        Две матрицы переходов (одинаковой размерности).
    metric : str
        Тип метрики: 'mae' (средняя абсолютная ошибка), 'mse' (среднеквадратичная),
        'kl' (среднее KL-расхождение по строкам, сглаженное).

    Возвращает
    -------
    value : float
        Численная метрика.
    """
    if metric == 'mae':
        return np.mean(np.abs(P_true - P_noisy))
    elif metric == 'mse':
        return np.mean((P_true - P_noisy)**2)
    elif metric == 'kl':
        # Среднее KL-расхождение по строкам, где строки ненулевые
        kl_sum = 0.0
        n_rows = P_true.shape[0]
        for i in range(n_rows):
            p = P_true[i, :]
            q = P_noisy[i, :]
            # Избегаем логарифма от нуля: добавляем небольшой эпсилон
            eps = 1e-12
            p = p + eps
            q = q + eps
            p = p / np.sum(p)
            q = q / np.sum(q)
            kl_sum += entropy(p, q)
        return kl_sum / n_rows
    else:
        raise ValueError(f"Неизвестная метрика: {metric}")


def simulate_markov_chain(P: np.ndarray, initial_state: int, n_steps: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Генерирует последовательность состояний по заданной матрице переходов.

    Параметры
    ----------
    P : np.ndarray
        Матрица переходов (размера n x n).
    initial_state : int
        Начальное состояние (0 <= initial_state < n).
    n_steps : int
        Количество шагов (длина генерируемой последовательности = n_steps+1).
    seed : Optional[int]
        Для воспроизводимости.

    Возвращает
    -------
    states : np.ndarray
        Массив состояний длины n_steps+1, начиная с initial_state.
    """
    if seed is not None:
        np.random.seed(seed)
    n_states = P.shape[0]
    states = np.zeros(n_steps + 1, dtype=int)
    states[0] = initial_state
    current = initial_state
    for t in range(1, n_steps + 1):
        current = np.random.choice(n_states, p=P[current, :])
        states[t] = current
    return states