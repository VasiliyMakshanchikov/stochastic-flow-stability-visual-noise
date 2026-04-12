from add_gaussian_noise import add_gaussian_noise
from add_salt_pepper_noise import add_salt_pepper_noise
from add_speckle_noise import add_speckle_noise
from add_poisson_noise import add_poisson_noise

import numpy as np

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