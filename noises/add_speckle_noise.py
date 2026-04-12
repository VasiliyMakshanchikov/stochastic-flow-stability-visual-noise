import numpy as np

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