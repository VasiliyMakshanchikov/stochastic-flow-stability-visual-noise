import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import cv2
import numpy as np
import multiprocessing as mp
import time
from pathlib import Path
from tqdm import tqdm  # используется только для эффекта, можно убрать
import tempfile
import shutil

# Пути к конфигурации и весам модели
config_path = "/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
weights_path = "/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/GroundingDINO/weights/groundingdino_swint_ogc.pth"

text_prompt = "people"
box_threshold = 0.35
text_threshold = 0.25

# Директории
FRAME_DIR = "/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/mall_dataset/frames"
OUTPUT_BASE = Path("/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/inferense_datasets/groundingdino_results_parr")
OUTPUT_BASE.mkdir(exist_ok=True)


from wrapper import add_noise

# Глобальная переменная для модели (будет инициализирована в каждом процессе)
_model = None

def init_worker(cfg_path, wgt_path):
    """Инициализация модели в каждом рабочем процессе."""
    global _model
    from groundingdino.util.inference import load_model, load_image, predict
    _model = load_model(cfg_path, wgt_path)
    _model = _model.to('cpu')   # или 'cuda' при наличии GPU

def count_people_in_image(image_path):
    """Подсчёт людей на изображении с использованием глобальной модели."""
    from groundingdino.util.inference import load_image, predict
    global _model
    try:
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cpu"
        )
        return len(boxes)
    except Exception as e:
        print(f"Ошибка {image_path}: {e}")
        return 0

def process_single_experiment(frame_dir, output_path, noise_params, skip_existing=True):
    """
    Обрабатывает один эксперимент:
    - если noise_params is None -> baseline (без шума)
    - иначе применяет шум к каждому кадру, сохраняя временные копии
    """
    output_path = Path(output_path)
    if skip_existing and output_path.exists():
        print(f"Пропуск (уже есть): {output_path}")
        return np.load(output_path)

    # Получаем список файлов кадров
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    counts = []

    # Создаём уникальную временную папку для этого эксперимента
    temp_dir = None
    if noise_params is not None:
        temp_dir = Path(tempfile.mkdtemp(prefix="exp_noise_"))
        # print(f"Создана временная папка: {temp_dir}")

    print(f"Обработка эксперимента: {output_path.name} (кадров: {len(frame_files)})")
    for frame_file in frame_files:
        img_path = os.path.join(frame_dir, frame_file)
        if noise_params is not None:
            img = cv2.imread(img_path)
            if img is None:
                counts.append(0)
                continue
            noisy_img = add_noise(img, **noise_params)
            temp_path = temp_dir / frame_file
            cv2.imwrite(str(temp_path), noisy_img)
            cnt = count_people_in_image(str(temp_path))
        else:
            cnt = count_people_in_image(img_path)
        counts.append(cnt)

    # Удаляем временную папку
    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)
        # print(f"Удалена временная папка: {temp_dir}")

    counts = np.array(counts, dtype=int)
    np.save(output_path, counts)
    # print(f"Сохранено: {output_path}")
    return counts

def run_experiments_parallel(num_workers=None):
    """
    Запуск всех экспериментов параллельно.
    num_workers: количество параллельных процессов (по умолчанию mp.cpu_count())
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    print(f"Запуск с {num_workers} параллельными процессами")

    # Список задач: каждая задача = (frame_dir, output_path, noise_params, skip_existing)
    tasks = []

    # Baseline (без шума)
    orig_path = OUTPUT_BASE / "original.npy"
    tasks.append((FRAME_DIR, orig_path, None, True))

    # Конфигурации экспериментов
    experiments_config = [
        {
            "noise_type": "gaussian",
            "param_name": "var",
            "values": np.linspace(0.001, 0.05, 10),
            "fixed_params": {"mean": 0.0}
        },
        {
            "noise_type": "salt_pepper",
            "param_name": "prob",
            "values": np.linspace(0.001, 0.05, 10),
            "fixed_params": {}
        },
        {
            "noise_type": "speckle",
            "param_name": "scale",
            "values": np.linspace(0.05, 0.8, 12),
            "fixed_params": {}
        },
        {
            "noise_type": "poisson",
            "param_name": "scale",
            "values": np.linspace(0.2, 2.0, 10),
            "fixed_params": {}
        }
    ]

    for cfg in experiments_config:
        noise_type = cfg["noise_type"]
        param_name = cfg["param_name"]
        values = cfg["values"]
        fixed = cfg["fixed_params"]

        for val in values:
            # Формируем параметры шума и имя выходного файла
            if noise_type == "gaussian":
                params = {
                    "noise_type": "gaussian",
                    "mean": fixed.get("mean", 0.0),
                    "var": val
                }
                filename = f"{noise_type}_{param_name}_{val:.6f}.npy"
            elif noise_type == "salt_pepper":
                params = {
                    "noise_type": "salt_pepper",
                    "salt_prob": val,
                    "pepper_prob": val
                }
                filename = f"{noise_type}_{param_name}_{val:.6f}.npy"
            elif noise_type == "speckle":
                params = {
                    "noise_type": "speckle",
                    "speckle_scale": val
                }
                filename = f"{noise_type}_{param_name}_{val:.6f}.npy"
            elif noise_type == "poisson":
                params = {
                    "noise_type": "poisson",
                    "poisson_scale": val
                }
                filename = f"{noise_type}_{param_name}_{val:.6f}.npy"
            else:
                continue

            out_path = OUTPUT_BASE / filename
            tasks.append((FRAME_DIR, out_path, params, True))

    print(f"Всего задач: {len(tasks)}")

    # Запуск пула процессов с инициализацией модели в каждом
    with mp.Pool(processes=num_workers,
                 initializer=init_worker,
                 initargs=(config_path, weights_path)) as pool:
        # starmap позволяет передать несколько аргументов в функцию
        pool.starmap(process_single_experiment, tasks)

    print("\nВсе эксперименты завершены!")

if __name__ == "__main__":
    start = time.perf_counter()
    run_experiments_parallel(num_workers=4)  
    end = time.perf_counter()
    print(f"Общее время выполнения: {end - start:.2f} сек")