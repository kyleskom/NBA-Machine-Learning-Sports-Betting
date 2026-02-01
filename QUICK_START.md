# 🏀 NBA Player Props - Краткая инструкция

## Что это?
Система для предсказания over/under линий очков игроков NBA с помощью машинного обучения.

## ⚡ Быстрый старт (5 минут до первых предсказаний)

### Вариант 1: Автоматическая установка
```bash
python quickstart.py
```
Скрипт автоматически всё настроит (займёт 15-20 минут).

### Вариант 2: Ручная установка

1. **Инициализация**
```bash
python src/Process-Data/Get_Player_Stats.py --init
python src/Process-Data/Get_Player_Props.py --init
```

2. **Загрузка данных** (10-15 мин)
```bash
python src/Process-Data/Get_Player_Stats.py --full-update --fetch-games 60
python src/Process-Data/Get_Player_Props.py --fetch
```

3. **Создание датасета**
```bash
python src/Process-Data/Create_Player_Props_Dataset.py --days-back 180
```

4. **Обучение модели** (5-10 мин)
```bash
python src/Train-Models/XGBoost_Model_PlayerProps.py --trials 50
```

5. **Получение предсказаний** ⭐
```bash
python predict_player_props.py
```

## 📊 Пример вывода

```
PLAYER                     LINE   PRED    AVG  TREND    REC   CONF   EDGE
================================================================================
LeBron James               25.5   27.3   26.2   +1.1   OVER  68.3%  +12.4
Stephen Curry              28.5   26.8   27.5   -0.7  UNDER  61.2%   +8.3
Kevin Durant               27.5   28.1   27.8   +0.3   OVER  58.7%   +5.1
```

**Объяснение:**
- **LINE** - Линия букмекера
- **PRED** - Наше предсказание
- **AVG** - Средние очки за последние 10 игр
- **TREND** - Тренд (+рост, -падение)
- **REC** - Рекомендация (OVER/UNDER/PASS)
- **CONF** - Уверенность модели (%)
- **EDGE** - Математическое преимущество

## 🎯 Как использовать

### Ежедневные предсказания
```bash
# Все предсказания на сегодня
python predict_player_props.py

# Только хорошие ставки (edge > 5%)
python predict_player_props.py --min-edge 5

# Анализ конкретного игрока
python predict_player_props.py --player "LeBron James"
```

### Ежедневное обновление данных
```bash
# Обновить статистику (3 последних дня)
python src/Process-Data/Get_Player_Stats.py --fetch-games 3

# Обновить букмекерские линии
python src/Process-Data/Get_Player_Props.py --fetch

# Получить предсказания
python predict_player_props.py --min-edge 5
```

## 💡 Важные советы

✅ **ДЕЛАЙТЕ:**
- Ищите edge > 5% (лучше > 10%)
- Проверяйте injury reports
- Обновляйте данные ежедневно
- Переобучайте модель раз в месяц

❌ **НЕ ДЕЛАЙТЕ:**
- Не ставьте на всё подряд
- Не игнорируйте последние новости (травмы, rest days)
- Не используйте старые данные (> 3 дней)
- Не забывайте про bankroll management

## 🔧 Решение проблем

**"No props for today"**
→ Запустите: `python src/Process-Data/Get_Player_Props.py --fetch`

**"Model not found"**
→ Обучите модель: `python src/Train-Models/XGBoost_Model_PlayerProps.py --trials 20`

**"Датасет пуст"**
→ Загрузите данные: `python src/Process-Data/Get_Player_Stats.py --fetch-games 60`

**HTTP ошибки**
→ Проверьте интернет и доступность NBA API

## 📁 Важные файлы

```
predict_player_props.py          ← Главный скрипт для предсказаний
PLAYER_PROPS_README.md          ← Полная документация
quickstart.py                    ← Автоматическая установка

Data/
├── PlayerStats.sqlite          ← Статистика игроков
├── PlayerProps.sqlite          ← Букмекерские линии
└── player_props_dataset.sqlite ← Обучающий датасет

Models/PlayerProps_Models/       ← Обученные модели
```

## 🌐 Бесплатные источники данных

Система использует 100% бесплатные API:

1. **NBA Stats API** - официальная статистика NBA
   - Бесплатно, без ключа
   - Rate limit: ~1 запрос в 1-2 секунды

2. **PrizePicks API** - DFS платформа
   - Бесплатно, без ключа
   - Player props линии

3. **Underdog Fantasy API** - DFS платформа
   - Бесплатно, без ключа
   - Player props линии

4. **The Odds API** (опционально)
   - Нужен бесплатный ключ
   - 500 запросов/месяц бесплатно
   - https://the-odds-api.com/

## 📈 Ожидаемые результаты

**Точность модели:** 58-65%
- Это ХОРОШИЙ результат!
- Букмекеры очень точны (обычно ~52-53%)
- Даже 55% = значительное преимущество

**Expected Value (EV):**
- EV > 5% = хорошая ставка
- EV > 10% = отличная ставка
- EV > 15% = проверьте данные (может быть ошибка)

## ⚠️ Disclaimer

- Это образовательный проект
- Не финансовый совет
- Ставки на спорт = риск
- Играйте ответственно
- The house always has an edge

## 📚 Дополнительно

Полная документация: `PLAYER_PROPS_README.md`

Вопросы? Проблемы? Проверьте README или логи ошибок.

---

**Удачи! 🍀**

*Remember: Past performance doesn't guarantee future results*
