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