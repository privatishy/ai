// Основной JavaScript для системы распознавания

class CanteenAI {
  constructor() {
    this.apiUrl = "/api/detect";
    this.updateInterval = 2000; // 2 секунды
    this.init();
  }

  init() {
    console.log("🚀 CanteenAI initialized");
    this.startAutoUpdate();
    this.setupEventListeners();
  }

  // Автоматическое обновление данных
  startAutoUpdate() {
    setInterval(() => {
      this.fetchData();
    }, this.updateInterval);
  }

  // Получение данных с сервера
  async fetchData() {
    try {
      const response = await fetch(this.apiUrl);
      const data = await response.json();
      this.updateUI(data);
    } catch (error) {
      console.error("❌ Error fetching data:", error);
      this.showError("Ошибка подключения к серверу");
    }
  }

  // Обновление интерфейса
  updateUI(data) {
    // Обновление списка блюд
    this.updateDishesList(data.dishes);

    // Обновление общей стоимости
    this.updateTotalPrice(data.total);

    // Обновление статистики
    this.updateStats(data.fps, data.objects);
  }

  updateDishesList(dishes) {
    const container = document.getElementById("dishes-list");
    if (!container) return;

    if (dishes && dishes.length > 0) {
      container.innerHTML = dishes
        .map(
          (dish) => `
                <div class="dish-item">
                    <span class="dish-name">${this.getDishName(dish.name)}</span>
                    <span class="dish-count">${dish.count} шт.</span>
                    <span class="dish-price">${dish.price * dish.count} ₽</span>
                </div>
            `,
        )
        .join("");
    } else {
      container.innerHTML = '<p class="no-dishes">Поднос не обнаружен</p>';
    }
  }

  updateTotalPrice(total) {
    const element = document.getElementById("total-price");
    if (element) {
      element.textContent = `${total} ₽`;
      // Анимация при изменении цены
      element.style.transform = "scale(1.1)";
      setTimeout(() => {
        element.style.transform = "scale(1)";
      }, 200);
    }
  }

  updateStats(fps, objects) {
    const fpsElement = document.getElementById("fps");
    const objectsElement = document.getElementById("objects");

    if (fpsElement) fpsElement.textContent = fps || 0;
    if (objectsElement) objectsElement.textContent = objects || 0;

    // Предупреждение о низком FPS
    if (fps < 10) {
      fpsElement.style.color = "#ff6b6b";
    } else {
      fpsElement.style.color = "#667eea";
    }
  }

  // Получение русского названия блюда
  getDishName(englishName) {
    const names = {
      tray: "Поднос",
      borsch: "Борщ",
      pasta: "Макароны",
      cutlet: "Котлета",
      salad: "Салат",
      tea: "Чай",
      compote: "Компот",
      bread: "Хлеб",
      soup: "Суп",
      garnish: "Гарнир",
      dessert: "Десерт",
      drink: "Напиток",
    };
    return names[englishName] || englishName;
  }

  // Показ ошибки
  showError(message) {
    console.error(message);
    // Можно добавить toast-уведомление
  }

  // Настройка обработчиков событий
  setupEventListeners() {
    // Обработка клавиш
    document.addEventListener("keydown", (e) => {
      if (e.key === "r" || e.key === "R") {
        this.fetchData(); // Принудительное обновление
      }
    });
  }
}

// Функции для админ-панели
function retrainModel() {
  if (confirm("Запустить дообучение модели?")) {
    fetch("/api/retrain", { method: "POST" })
      .then((response) => response.json())
      .then((data) => alert("Обучение началось!"))
      .catch((error) => alert("Ошибка: " + error));
  }
}

function exportLogs() {
  window.location.href = "/api/export-logs";
}

function clearCache() {
  if (confirm("Очистить кэш модели?")) {
    fetch("/api/clear-cache", { method: "POST" })
      .then(() => alert("Кэш очищен"))
      .catch((error) => alert("Ошибка: " + error));
  }
}

function restartSystem() {
  if (confirm("Перезапустить систему?")) {
    fetch("/api/restart", { method: "POST" })
      .then(() => alert("Система перезапускается..."))
      .catch((error) => alert("Ошибка: " + error));
  }
}

// Инициализация при загрузке страницы
document.addEventListener("DOMContentLoaded", () => {
  window.canteenAI = new CanteenAI();
});
