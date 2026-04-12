import cv2
import numpy as np

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