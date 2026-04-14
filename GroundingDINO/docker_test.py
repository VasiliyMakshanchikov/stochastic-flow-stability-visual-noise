from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import cv2
import numpy as np

# 1. ЗАГРУЖАЕМ МОДЕЛЬ
config_path = "/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
weights_path = "/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/GroundingDINO/weights/groundingdino_swint_ogc.pth"

model = load_model(config_path, weights_path)
model = model.to('cpu')

# 2. ЗАГРУЖАЕМ ИЗОБРАЖЕНИЕ
print("2. Загрузка изображения...")
image_path = "/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/mall_dataset/frames/seq_000001.jpg"
# image_path = "//Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/noises/image.jpg"
image_source, image = load_image(image_path)

# СОЗДАЁМ РЕДАКТИРУЕМУЮ КОПИЮ
print("3. Поиск объектов...")
image_copy = image_source.copy()

# 3. ЗАДАЁМ ТЕКСТОВЫЙ ЗАПРОС
text_prompt = "people"

# 4. ЗАПУСКАЕМ ПОИСК
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=text_prompt,
    box_threshold=0.35,
    text_threshold=0.25,
    device="cpu"
)

print(f"4. Найдено {len(boxes)} объектов")

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("output_annotated.jpg", annotated_frame)
print("5. Также сохранён output_annotated.jpg со встроенной аннотацией")