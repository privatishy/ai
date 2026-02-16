# Скрипт для обучения YOLOv8m на датасете школьных блюд
# transfer learning, YOLOv8m

import os
from ultralytics import YOLO
import yaml

def train():
    print("=" * 50)
    print("🚀 Обучение модели YOLOv8m для школьной столовой")
    print("=" * 50)
    
    # 1. Загрузка предобученной модели (transfer learning)
    print("\n📦 Загрузка предобученной модели YOLOv8m...")
    model = YOLO('yolov8m.pt')
    
    # 2. Параметры обучения из config.yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    dataset_config = config['dataset']
    
    # 3. Запуск обучения
    print("\n📚 Начало обучения...")
    print(f"   - Epochs: {model_config['epochs']}")
    print(f"   - Batch size: {model_config['batches']}")
    print(f"   - Image size: {model_config['image_size']}")
    print(f"   - Dataset: {dataset_config['path']}")
    
    results = model.train(
        data='dataset.yaml',           # Путь к конфигу датасета
        epochs=model_config['epochs'],
        batch=model_config['batches'],
        imgsz=model_config['image_size'],
        pretrained=True,                # Transfer learning
        patience=model_config['patience'],
        device=0,                       # GPU (или 'cpu' если нет GPU)
        workers=4,
        verbose=True,
        project='weights/',
        name='school_canteen_model',
        exist_ok=True
    )
    
    # 4. Сохранение лучшей модели
    print("\n✅ Обучение завершено!")
    print(f"📁 Лучшая модель сохранена в: weights/school_canteen_model/weights/best.pt")
    
    # 5. Тестирование модели
    print("\n🧪 Тестирование модели...")
    metrics = model.val(data='dataset.yaml')
    print(f"\n📊 Метрики:")
    print(f"   - mAP50: {metrics.box.map50:.4f}")
    print(f"   - mAP50-95: {metrics.box.map:.4f}")
    
    print("\n" + "=" * 50)
    print("🎉 Обучение успешно завершено!")
    print("=" * 50)

if __name__ == "__main__":
    train()