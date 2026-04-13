from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import cv2

# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
# # model = model.to('cuda:0')
# print(torch.cuda.is_available())
# print('DONE!')

# 1. ЗАГРУЖАЕМ МОДЕЛЬ (путь к конфигу и весам)
config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
weights_path = "weights/groundingdino_swint_ogc.pth"

model = load_model(config_path, weights_path)
model = model.to('cpu')

# 2. ЗАГРУЖАЕМ ИЗОБРАЖЕНИЕ
print("2. Загрузка изображения...")

image_path = "/Users/v.makshanchikov/Documents/Python Proj/ВКР Магистратура/noises/image.jpg"  # укажите путь к вашему фото
image_source, image = load_image(image_path)

# СОЗДАЁМ РЕДАКТИРУЕМУЮ КОПИЮ
print("3. Поиск объектов...")
image_copy = image_source.copy()

# 3. ЗАДАЁМ ТЕКСТОВЫЙ ЗАПРОС (что ищем)
text_prompt = "a man"  # можно на русском, но лучше на английском

# 4. ЗАПУСКАЕМ ПОИСК
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=text_prompt,
    box_threshold=0.35,   # порог уверенности (0.3-0.5 обычно норм)
    text_threshold=0.25,
    device="cpu"
)

print(f"4. Найдено {len(boxes)} объектов")

# РИСУЕМ НА КОПИИ
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box.tolist()
    confidence = logits[i].item()
    cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    label = f"{text_prompt}: {confidence:.2f}"
    cv2.putText(image_copy, label, (int(x1), int(y1)-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Сохраняем результат
cv2.imwrite("output.jpg", image_copy)
print("5. Результат сохранён в output.jpg")