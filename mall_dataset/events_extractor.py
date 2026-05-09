import scipy.io
import numpy as np
from typing import List, Tuple

def load_mall_ground_truth(mat_path: str) -> np.ndarray:
    """
    Загрузка ground truth из mall_gt.mat
    
    Returns:
        list_counts: список количества людей на каждом кадре
    """
    data = scipy.io.loadmat(mat_path)
    
    # Извлекаем массив frame
    frames = data['frame']
    
    # Подсчитываем количество людей в каждом кадре
    counts_per_frame = []
    
    for f in frames[0]:
        try:
            points = f[0][0][0]
            # print(points, '\n')
            # если это массив точек
            if isinstance(points, np.ndarray):
                if points.ndim == 2:
                    counts_per_frame.append(points.shape[0])
                elif points.ndim == 1:
                    counts_per_frame.append(len(points))
                else:
                    counts_per_frame.append(0)

            else:
                # если это float / None / что-то странное
                counts_per_frame.append(0)

        except Exception:
            # fallback на случай неожиданных структур
            counts_per_frame.append(0)

    return np.array(counts_per_frame)


def extract_event_timestamps(counts_per_frame: np.ndarray) -> List[Tuple[int, int, str]]:
    """
    Извлечение временных меток событий (изменения числа людей)
    
    Returns:
        список событий: (frame_index, count, event_type)
        event_type: 'enter' (увеличение), 'exit' (уменьшение)
    """
    events = []
    
    for i in range(1, len(counts_per_frame)):
        prev_count = counts_per_frame[i-1]
        curr_count = counts_per_frame[i]
        
        if curr_count > prev_count:
            events.append((i, curr_count, 'increase', curr_count - prev_count))
        elif curr_count < prev_count:
            events.append((i, curr_count, 'decrease', prev_count - curr_count))
    
    return events

def extract_significant_events(counts_per_frame: np.ndarray, threshold: int = 3) -> List[Tuple[int, int, str]]:
    """
    Извлечение только значительных изменений (больше threshold)
    """
    events = []
    prev_count = counts_per_frame[0]
    
    for i in range(1, len(counts_per_frame)):
        curr_count = counts_per_frame[i]
        diff = curr_count - prev_count
        
        if abs(diff) >= threshold:
            event_type = 'increase' if diff > 0 else 'decrease'
            events.append((i, curr_count, event_type, abs(diff)))
            prev_count = curr_count  # сброс после значительного события
    
    return events


def get_frame_timestamps(mat_path: str, fps: float = 1, significant_events: bool = False, threshold: int = 3) -> List[float]:
    """
    Конвертация номеров кадров во временные метки (секунды)
    
    Args:
        fps: частота кадров видео (по умолчанию 25 fps для mall dataset)
    """
    counts = load_mall_ground_truth(mat_path)
    if not significant_events:
        events = extract_event_timestamps(counts)
    else:
        events = extract_significant_events(counts, threshold=3)
    
    timestamps = []
    for frame_idx, count, event_type, delta in events:
        timestamp = frame_idx / fps
        timestamps.append({
            'frame': frame_idx,
            'timestamp_sec': timestamp,
            'timestamp_str': f"{int(timestamp//60)}:{int(timestamp%60):02d}.{int((timestamp%1)*100):02d}",
            'people_count': count,
            'event': event_type,
            'change': delta
        })
    
    return timestamps