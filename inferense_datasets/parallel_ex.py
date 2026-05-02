import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from wrapper import add_noise 
import numpy as np
import multiprocessing as mp
import cv2
import time
from pathlib    import Path
from groundingdino.util.inference import load_model, load_image, predict


# Глобальные переменные для каждого процесса (будут инициализированы заново)
_model = None
_text_prompt = "people"
_box_threshold = 0.35
_text_threshold = 0.25
_config_path = "/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
_weights_path = "/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/GroundingDINO/weights/groundingdino_swint_ogc.pth"

def get_model():
    global _model
    if _model is None:
        _model = load_model(_config_path, _weights_path)
        _model = _model.to('cpu')
    return _model

def count_people_in_image(image_path):
    try:
        model = get_model()
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=_text_prompt,
            box_threshold=_box_threshold,
            text_threshold=_text_threshold,
            device="cpu"
        )
        return len(boxes)
    except Exception as e:
        print(f"Ошибка {image_path}: {e}")
        return 0

def process_single_experiment(frame_dir, output_path, noise_params, skip_existing):
    if skip_existing and output_path.exists():
        print(f"Пропуск (уже есть): {output_path}")
        return None

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    counts = []
    temp_dir = None
    if noise_params is not None:
        temp_dir = Path(f"temp_{os.getpid()}")
        temp_dir.mkdir(exist_ok=True)

    for i, frame_file in enumerate(frame_files):
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

        # Лог каждые 100 кадров (не спамит)
        if (i+1) % 100 == 0:
            print(f"[{output_path.name}] processed {i+1}/{len(frame_files)} frames")

    if temp_dir is not None:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    counts = np.array(counts, dtype=int)
    np.save(output_path, counts)
    print(f"Сохранено: {output_path}")
    return counts

def run_experiments_parallel(num_workers=None):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    print(f"Используется {num_workers} параллельных процессов")

    tasks = []

    # Baseline
    orig_path = OUTPUT_BASE / "original.npy"
    tasks.append((FRAME_DIR, orig_path, None, True))

    # Остальные эксперименты (как ранее)
    experiments_config = [
        {"noise_type": "gaussian", "param_name": "var", "values": np.linspace(0.001, 0.05, 10), "fixed_params": {"mean": 0.0}},
        {"noise_type": "salt_pepper", "param_name": "prob", "values": np.linspace(0.001, 0.05, 10), "fixed_params": {}},
        {"noise_type": "speckle", "param_name": "scale", "values": np.linspace(0.05, 0.8, 12), "fixed_params": {}},
        {"noise_type": "poisson", "param_name": "scale", "values": np.linspace(0.2, 2.0, 10), "fixed_params": {}}
    ]

    for cfg in experiments_config:
        noise_type = cfg["noise_type"]
        param_name = cfg["param_name"]
        values = cfg["values"]
        fixed = cfg["fixed_params"]
        for val in values:
            if noise_type == "gaussian":
                params = {"noise_type": "gaussian", "mean": fixed.get("mean", 0.0), "var": val}
                filename = f"{noise_type}_{param_name}_{val:.6f}.npy"
            elif noise_type == "salt_pepper":
                params = {"noise_type": "salt_pepper", "salt_prob": val, "pepper_prob": val}
                filename = f"{noise_type}_{param_name}_{val:.6f}.npy"
            elif noise_type == "speckle":
                params = {"noise_type": "speckle", "speckle_scale": val}
                filename = f"{noise_type}_{param_name}_{val:.6f}.npy"
            elif noise_type == "poisson":
                params = {"noise_type": "poisson", "poisson_scale": val}
                filename = f"{noise_type}_{param_name}_{val:.6f}.npy"
            else:
                continue
            out_path = OUTPUT_BASE / filename
            tasks.append((FRAME_DIR, out_path, params, True))

    # Запуск с контекстом spawn
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        # imap_unordered для получения результатов без сохранения порядка
        results = pool.starmap(process_single_experiment, tasks)

    print("Все эксперименты завершены!")

if __name__ == "__main__":
    # Важно: на macOS необходимо защищать вызов мультипроцессинга
    FRAME_DIR = "/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/mall_dataset/pre_frames"
    OUTPUT_BASE = Path("/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/inferense_datasets/groundingdino_results_parr")
    OUTPUT_BASE.mkdir(exist_ok=True)
    total_start = time.perf_counter()
    run_experiments_parallel(num_workers=3)
    total_end = time.perf_counter()
    print(f"Общее время: {total_end - total_start:.2f} сек")