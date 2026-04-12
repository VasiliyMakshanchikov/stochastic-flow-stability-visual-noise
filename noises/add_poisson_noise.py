import numpy as np


def add_poisson_noise(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Добавляет пуассоновский (shot) шум к изображению.

    Parameters
    ----------
    image : np.ndarray
        Входное изображение (BGR или grayscale), dtype=uint8.

    scale : float, optional
        Масштаб интенсивности (λ).
        - scale > 1 → меньше шума (больше "фотонов")
        - scale < 1 → больше шума

    Returns
    -------
    np.ndarray
        Изображение с добавленным пуассоновским шумом (dtype=uint8).

    Notes
    -----
    - Пуассоновский шум зависит от интенсивности сигнала:
      Var(X) = E[X]
    - Используется для моделирования:
        * фотонного шума камер
        * дискретных потоков событий
    - Особенно релевантен для задач CV + стохастических потоков
    """

    # Нормализация
    img = image.astype(np.float32) / 255.0

    # Избегаем нулей (иначе Poisson всегда 0)
    img = np.clip(img, 1e-6, 1.0)

    # Масштабирование (имитация числа "событий")
    scaled = img * scale * 255.0

    # Применение пуассоновского шума
    noisy = np.random.poisson(scaled) / (scale * 255.0)

    # Обратное преобразование
    noisy = np.clip(noisy, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)

    return noisy