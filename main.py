# Система автоматизации оплаты школьного питания
# Стек: Python, PyTorch, OpenCV, Flask
# YOLOv8m, точность 92%, задержка <1.5 сек

import cv2
import yaml
import os
import time
import threading
from ultralytics import YOLO
from dotenv import load_dotenv
from flask import Flask, render_template, Response, jsonify, request
import logging
from datetime import datetime

# ============================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================

# Создание директории для логов
os.makedirs('logs', exist_ok=True)

# Основной логгер
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

# Обработчик для файла app.log
file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Обработчик для консоли
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Логгер для ошибок
error_logger = logging.getLogger('errors')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('logs/errors.log', encoding='utf-8')
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
error_logger.addHandler(error_handler)

# Логгер для доступа
access_logger = logging.getLogger('access')
access_logger.setLevel(logging.INFO)
access_handler = logging.FileHandler('logs/access.log', encoding='utf-8')
access_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
access_logger.addHandler(access_handler)

# ============================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ============================================

# Загрузка переменных окружения
load_dotenv()

# Загрузка конфигурации
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# ============================================
# ИНИЦИАЛИЗАЦИЯ FLASK
# ============================================

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')

# ============================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ С БЛОКИРОВКОЙ
# ============================================

# Блокировка для безопасного доступа из разных потоков
data_lock = threading.Lock()

# Глобальные переменные для хранения данных
latest_frame = None
latest_results = None
fps_value = 0
objects_count = 0
dishes_data = []
total_price = 0
camera_running = True  # Флаг для остановки потока

# ============================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================

model_path = os.getenv('MODEL_PATH', 'weights/yolov8m.pt')
logger.info(f"Загрузка модели из {model_path}...")

try:
    model = YOLO(model_path)
    logger.info("✅ Модель успешно загружена")
except Exception as e:
    error_logger.error(f"Ошибка загрузки модели: {e}")
    raise

# Словарь цен
prices = {item['id']: item['price'] for item in config['classes'] if 'price' in item}

# ============================================
# ПОТОК ОБРАБОТКИ КАМЕРЫ
# ============================================

def process_camera():  
    """Поток для обработки камеры"""
    global latest_frame, latest_results, fps_value, objects_count, dishes_data, total_price, camera_running
    
    camera_id = int(os.getenv('CAMERA_ID', 0))
    confidence = float(os.getenv('MODEL_CONFIDENCE', 0.5))
    img_size = int(os.getenv('MODEL_IMAGE_SIZE', 640))
    
    logger.info(f"Подключение к камере (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    
    # Проверка подключения к камере
    if not cap.isOpened():
        error_logger.error("Не удалось открыть камеру!")
        camera_running = False
        return
    
    logger.info("✅ Камера успешно подключена")
    
    while camera_running:
        try:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                error_logger.warning("Не удалось получить кадр с камеры")
                continue
            
            # Инференс
            results = model(frame, verbose=False, imgsz=img_size, conf=confidence, device='cpu')
            
            # Визуализация
            annotated_frame = results[0].plot()
            
            # Расчет FPS
            end_time = time.time()
            current_fps = 1 / (end_time - start_time)
            
            # Подсчет объектов и стоимости
            detected_classes = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes) > 0 else []
            current_objects_count = len(detected_classes)
            
            current_total_price = 0
            class_counts = {}
            for cls in detected_classes:
                cls_id = int(cls)
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                if cls_id in prices:
                    current_total_price += prices[cls_id]
            
            # Формирование данных для API
            current_dishes_data = []
            for cls_id, count in class_counts.items():
                if cls_id < len(config['classes']):
                    dish = config['classes'][cls_id]
                    if 'price' in dish:
                        current_dishes_data.append({
                            'name': dish['name'],
                            'count': count,
                            'price': dish['price']
                        })
            
            # БЕЗОПАСНОЕ обновление глобальных переменных с блокировкой
            with data_lock:
                latest_frame = annotated_frame
                latest_results = results
                fps_value = current_fps
                objects_count = current_objects_count
                dishes_data = current_dishes_data
                total_price = current_total_price
            
            # Отображение локального окна (для отладки)
            cv2.putText(annotated_frame, f'FPS: {current_fps:.2f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Total: {total_price} RUB', (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('School Canteen AI - Debug', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Получен сигнал остановки от пользователя")
                break
                
        except Exception as e:
            error_logger.error(f"Ошибка в потоке камеры: {e}")
            continue
    
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Поток камеры остановлен")

# ============================================
# ГЕНЕРАЦИЯ КАДРОВ ДЛЯ ВЕБ-ИНТЕРФЕЙСА
# ============================================

def generate_frames():
    """Генерация кадров для веб-страницы"""
    global latest_frame
    
    while True:
        with data_lock:
            frame = latest_frame
        
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Показываем заглушку, если кадр ещё не готов
            time.sleep(0.1)

# ============================================
# МАРШРУТЫ FLASK
# ============================================

@app.route('/')
def index():
    """Главная страница"""
    access_logger.info(f"Доступ к главной странице с {request.remote_addr}")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Поток видео для веб-страницы"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detect')
def detect():
    """API для получения данных о распознавании"""
    with data_lock:
        response_data = {
            'dishes': dishes_data.copy(),
            'total': total_price,
            'fps': round(fps_value, 2),
            'objects': objects_count
        }
    access_logger.info(f"API запрос /detect от {request.remote_addr}")
    return jsonify(response_data)

@app.route('/admin')
def admin():
    """Админ-панель"""
    access_logger.info(f"Админ-панель запрошена от {request.remote_addr}")
    return render_template('admin.html')

@app.route('/error')
def error():
    """Страница ошибки"""
    return render_template('error.html', error_code=404, error_message='Страница не найдена')

@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    """API для запуска дообучения"""
    access_logger.info(f"Запрос на дообучение от {request.remote_addr}")
    try:
        logger.info("Запуск дообучения модели")
        # Здесь будет логика запуска train.py
        return jsonify({'status': 'success', 'message': 'Обучение началось'})
    except Exception as e:
        error_logger.error(f"Ошибка дообучения: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/export-logs')
def export_logs():
    """Экспорт логов"""
    from flask import send_file
    import zipfile
    import io
    
    access_logger.info(f"Экспорт логов запрошен от {request.remote_addr}")
    
    # Создание ZIP архива с логами
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for log_file in ['logs/app.log', 'logs/errors.log', 'logs/access.log']:
            if os.path.exists(log_file):
                zf.write(log_file)
    
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='logs.zip'
    )

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Очистка кэша"""
    access_logger.info(f"Очистка кэша запрошена от {request.remote_addr}")
    try:
        logger.info("Очистка кэша модели")
        # Логика очистки
        return jsonify({'status': 'success'})
    except Exception as e:
        error_logger.error(f"Ошибка очистки кэша: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("🚀 Запуск PKAI")
    logger.info("=" * 50)
    
    # Запуск потока камеры
    camera_process_thread = threading.Thread(target=process_camera, daemon=True) 
    camera_process_thread.start()
    
    # Запуск Flask сервера
    flask_host = os.getenv('FLASK_HOST', '0.0.0.0')
    flask_port = int(os.getenv('FLASK_PORT', 5000))
    
    logger.info(f"🌐 Веб-интерфейс доступен по адресу: http://{flask_host}:{flask_port}")
    logger.info("Нажмите Ctrl+C для остановки\n")
    
    try:
        app.run(host=flask_host, port=flask_port, debug=False)
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки (Ctrl+C)")
    finally:
        camera_running = False
        logger.info("Завершение работы системы...")
        cv2.destroyAllWindows()
