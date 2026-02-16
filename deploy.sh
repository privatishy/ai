#!/bin/bash
# Скрипт автоматического развёртывания на сервер

echo "🚀 Начало развёртывания..."

# 1. Обновление кода
git pull origin main

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Миграции БД
python migrate.py

# 4. Остановка старого процесса
pkill -f "python main.py"

# 5. Запуск нового (через Gunicorn для продакшна)
gunicorn -w 4 -b 0.0.0.0:5000 main:app &

echo "✅ Развёртывание завершено!"