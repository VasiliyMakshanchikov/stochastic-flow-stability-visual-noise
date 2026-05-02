import numpy as np

def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    var: float = 0.01,
) -> np.ndarray:
    """
    Добавляет аддитивный гауссов шум к изображению.

    Parameters
    ----------
    image : np.ndarray
        Входное изображение (BGR или grayscale), dtype=uint8.
    mean : float, optional
        Среднее значение гауссова шума.
    var : float, optional
        Дисперсия гауссова шума.

    Returns
    -------
    np.ndarray
        Изображение с добавленным гауссовым шумом (dtype=uint8).
    """
    img = image.astype(np.float32) / 255.0
    sigma = np.sqrt(var)
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)
    
    return noisy

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

def add_salt_pepper_noise(
    image: np.ndarray,
    salt_prob: float = 0.01,
    pepper_prob: float = 0.01,
) -> np.ndarray:
    """
    Добавляет импульсный шум (соль и перец) к изображению.

    Parameters
    ----------
    image : np.ndarray
        Входное изображение (BGR или grayscale), dtype=uint8.
    salt_prob : float, optional
        Вероятность "соли" (белые пиксели).
    pepper_prob : float, optional
        Вероятность "перца" (чёрные пиксели).

    Returns
    -------
    np.ndarray
        Изображение с добавленным шумом "соль и перец" (dtype=uint8).
    """
    img = image.astype(np.float32) / 255.0
    noisy = img.copy()
    total_pixels = img.shape[0] * img.shape[1]

    # Соль (белые пиксели)
    num_salt = int(total_pixels * salt_prob)
    if num_salt > 0:
        coords = (
            np.random.randint(0, img.shape[0], num_salt),
            np.random.randint(0, img.shape[1], num_salt),
        )
        noisy[coords] = 1.0

    # Перец (чёрные пиксели)
    num_pepper = int(total_pixels * pepper_prob)
    if num_pepper > 0:
        coords = (
            np.random.randint(0, img.shape[0], num_pepper),
            np.random.randint(0, img.shape[1], num_pepper),
        )
        noisy[coords] = 0.0

    noisy = np.clip(noisy, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)
    
    return noisy


def add_speckle_noise(
    image: np.ndarray,
    scale: float = 0.2,
) -> np.ndarray:
    """
    Добавляет мультипликативный спекл-шум к изображению.

    Parameters
    ----------
    image : np.ndarray
        Входное изображение (BGR или grayscale), dtype=uint8.
    scale : float, optional
        Коэффициент масштабирования шума.

    Returns
    -------
    np.ndarray
        Изображение с добавленным спекл-шумом (dtype=uint8).
    """
    img = image.astype(np.float32) / 255.0
    noise = np.random.randn(*img.shape)
    noisy = img + img * noise * scale
    noisy = np.clip(noisy, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)
    
    return noisy


def add_noise(
    image: np.ndarray,
    noise_type: str = "gaussian",
    mean: float = 0.0,
    var: float = 0.01,
    salt_prob: float = 0.01,
    pepper_prob: float = 0.01,
    speckle_scale: float = 0.2,
    poisson_scale: float = 1.0,
) -> np.ndarray:
    """
        Parameters
    ----------
    image : np.ndarray
        Входное изображение (BGR или grayscale), dtype=uint8.

    noise_type : str, optional
        Тип шума:
        - 'gaussian' — аддитивный гауссов шум
        - 'salt_pepper' — импульсный шум (соль и перец)
        - 'speckle' — мультипликативный шум
        - 'poisson' — пуассоновский шум (фотонный шум)

    mean : float, optional
        Среднее значение гауссова шума (используется только для 'gaussian').
        Значение по умолчанию: 0.0

    var : float, optional
        Дисперсия гауссова шума (используется только для 'gaussian').
        Значение по умолчанию: 0.01
        Рекомендуемый диапазон: 0.001-0.05

    salt_prob : float, optional
        Вероятность "соли" (белые пиксели) для salt & pepper шума.
        Значение по умолчанию: 0.01
        Рекомендуемый диапазон: 0.001-0.05

    pepper_prob : float, optional
        Вероятность "перца" (чёрные пиксели) для salt & pepper шума.
        Значение по умолчанию: 0.01
        Рекомендуемый диапазон: 0.001-0.05

    speckle_scale : float, optional
        Коэффициент масштабирования для speckle шума.
        Значение по умолчанию: 0.2
        Рекомендуемый диапазон: 0.05-0.5

    poisson_scale : float, optional
        Масштабирующий коэффициент для пуассоновского шума.
        Значение по умолчанию: 1.0
        Рекомендуемый диапазон: 0.1-2.0
        (меньшее значение = более сильный шум)

    Returns
    -------
    np.ndarray
        Изображение с добавленным шумом (dtype=uint8).
    """
    if noise_type == "gaussian":
        return add_gaussian_noise(image, mean, var)
    elif noise_type == "salt_pepper":
        return add_salt_pepper_noise(image, salt_prob, pepper_prob)
    elif noise_type == "speckle":
        return add_speckle_noise(image, speckle_scale)
    elif noise_type == "poisson":
        return add_poisson_noise(image, poisson_scale)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}. "
                         f"Supported types: 'gaussian', 'salt_pepper', "
                         f"'speckle', 'poisson'")