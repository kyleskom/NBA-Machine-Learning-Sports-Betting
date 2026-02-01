# NBA Player Props Prediction System

Система для предсказания over/under линий очков игроков NBA с использованием машинного обучения.

## 🎯 Что делает эта система

Система анализирует:
- Историческую статистику игроков
- Актуальные линии букмекеров  
- Форму игрока (последние 5-10 игр)
- Тренды (рост/падение результатов)
- Дни отдыха, соперника, домашние/выездные игры

И предсказывает:
- Вероятность OVER/UNDER
- Expected Value (EV) для каждой ставки
- Рекомендации с уровнем уверенности

## 📋 Требования

```bash
pip install pandas numpy xgboost scikit-learn colorama requests joblib
```

Или из существующего requirements.txt:
```bash
pip install -r requirements.txt
```

## 🚀 Быстрый старт

### Шаг 1: Инициализация базы данных

```bash
# Создать базы данных для статистики игроков
python src/Process-Data/Get_Player_Stats.py --init

# Создать базу для букмекерских линий
python src/Process-Data/Get_Player_Props.py --init
```

### Шаг 2: Загрузка исторических данных

```bash
# Получить информацию об игроках и статистику за последние 60 дней
python src/Process-Data/Get_Player_Stats.py --full-update --fetch-games 60

# Это займёт 10-15 минут из-за rate limiting API
```

### Шаг 3: Получение букмекерских линий

```bash
# Получить актуальные player props линии
python src/Process-Data/Get_Player_Props.py --fetch

# Посмотреть что получили
python src/Process-Data/Get_Player_Props.py --show-today
```

**Примечание:** Бесплатные источники:
- **PrizePicks** - работает без API ключа
- **Underdog Fantasy** - работает без API ключа  
- **The Odds API** - нужен бесплатный ключ (500 запросов/мес)

Для The Odds API:
1. Зарегистрируйтесь на https://the-odds-api.com/
2. Получите бесплатный API ключ
3. Вставьте его в `src/Process-Data/Get_Player_Props.py` (строка 75)

### Шаг 4: Создание датасета для обучения

```bash
# Объединить статистику игроков с историческими линиями
python src/Process-Data/Create_Player_Props_Dataset.py --days-back 180

# Посмотреть анализ датасета
python src/Process-Data/Create_Player_Props_Dataset.py --days-back 180 --analyze
```

### Шаг 5: Обучение модели

```bash
# Обучить XGBoost модель (займёт 5-10 минут)
python src/Train-Models/XGBoost_Model_PlayerProps.py --trials 50 --splits 5

# Для быстрого теста (меньше точности):
python src/Train-Models/XGBoost_Model_PlayerProps.py --trials 10 --splits 3
```

Ожидаемые результаты:
- Accuracy: ~58-65%
- AUC-ROC: ~0.60-0.67
- Это нормально! Линии букмекеров очень хорошие, даже 58% = edge

### Шаг 6: Получение предсказаний

```bash
# Предсказания на сегодня
python predict_player_props.py

# Только ставки с положительным EV
python predict_player_props.py --min-edge 5

# Анализ конкретного игрока
python predict_player_props.py --player "LeBron James"
```

## 📊 Пример вывода

```
================================================================================
PLAYER                     LINE   PRED    AVG  TREND    REC   CONF   EDGE
================================================================================
LeBron James               25.5   27.3   26.2   +1.1   OVER  68.3%  +12.4
Stephen Curry              28.5   26.8   27.5   -0.7  UNDER  61.2%   +8.3
Kevin Durant               27.5   28.1   27.8   +0.3   OVER  58.7%   +5.1
Giannis Antetokounmpo      30.5   29.4   30.1   -0.7   PASS  52.1%   +1.2
================================================================================

Всего: 15 | Over: 6 | Under: 4 | Pass: 5

=== ТОП-5 СТАВОК ===
1. LeBron James: OVER 25.5 (edge: +12.4, confidence: 68.3%)
2. Stephen Curry: UNDER 28.5 (edge: +8.3, confidence: 61.2%)
3. Kevin Durant: OVER 27.5 (edge: +5.1, confidence: 58.7%)
```

## 🔄 Ежедневное обновление

Создайте скрипт для автоматического обновления:

```bash
#!/bin/bash
# daily_update.sh

echo "=== Обновление данных игроков ==="
python src/Process-Data/Get_Player_Stats.py --fetch-games 3

echo "=== Обновление букмекерских линий ==="
python src/Process-Data/Get_Player_Props.py --fetch

echo "=== Получение предсказаний ==="
python predict_player_props.py --min-edge 5
```

Запускайте каждое утро:
```bash
chmod +x daily_update.sh
./daily_update.sh
```

## 📁 Структура данных

```
Data/
├── PlayerStats.sqlite       # Статистика игроков
│   ├── players             # Информация об игроках
│   ├── game_stats          # Статистика по играм
│   └── rolling_averages    # Скользящие средние
│
├── PlayerProps.sqlite       # Букмекерские линии
│   └── player_props        # Линии от разных букмекеров
│
└── player_props_dataset.sqlite  # Датасет для обучения
    └── player_props_training    # Обучающие данные
```

## 🎓 Объяснение фич

Модель использует следующие признаки:

**Статистика игрока:**
- `avg_points_L5` - Средние очки за последние 5 игр
- `avg_points_L10` - Средние очки за последние 10 игр
- `avg_minutes` - Среднее время на площадке
- `points_std` - Стабильность (стандартное отклонение)
- `consistency` - Метрика консистентности (1 / std)
- `trend` - Тренд (последние 3 vs предыдущие 3 игры)

**Эффективность:**
- `avg_fg_pct` - Процент попаданий с игры
- `avg_fg3_pct` - Процент трёхочковых
- `avg_ft_pct` - Процент штрафных

**Контекст:**
- `home_games_pct` - Процент домашних игр
- `days_rest` - Дни отдыха с последней игры
- `line` - Линия букмекера
- `over_odds` / `under_odds` - Коэффициенты

## 💡 Советы по использованию

1. **Обновляйте данные регулярно**
   ```bash
   # Каждые 2-3 дня
   python src/Process-Data/Get_Player_Stats.py --fetch-games 3
   ```

2. **Проверяйте травмы**
   - Модель не учитывает last-minute травмы
   - Проверяйте injury reports перед ставками

3. **Ищите edge > 5%**
   - Edge < 5% = слишком мало преимущества
   - Edge > 10% = отличная ставка

4. **Смотрите на consistency**
   - Высокая consistency = более надёжное предсказание
   - Низкая consistency = рискованная ставка

5. **Обновляйте модель ежемесячно**
   ```bash
   # Переобучить с новыми данными
   python src/Process-Data/Create_Player_Props_Dataset.py --days-back 180
   python src/Train-Models/XGBoost_Model_PlayerProps.py --trials 50
   ```

## 🔍 Анализ результатов

Посмотреть точность модели по разным категориям:

```python
import sqlite3
import pandas as pd

# Загрузить результаты
con = sqlite3.connect('Data/player_props_dataset.sqlite')
df = pd.read_sql_query("SELECT * FROM player_props_training", con)

# Точность по линиям
df['line_bucket'] = (df['line'] // 5) * 5
accuracy = df.groupby('line_bucket')['target'].mean()
print(accuracy)

# Топ игроки по консистентности
top_consistent = df.groupby('player_name').agg({
    'points_std': 'mean',
    'target': 'count'
}).sort_values('points_std').head(20)
print(top_consistent)
```

## ⚠️ Важные замечания

1. **Это не financial advice**
   - Система для образовательных целей
   - Ставки на спорт = риск

2. **Модель не идеальна**
   - Букмекеры очень хороши в установке линий
   - 58-65% точности = хороший результат
   - Нужен bankroll management

3. **Rate limiting**
   - NBA Stats API имеет лимиты
   - Используйте задержки между запросами
   - Не спамьте API

4. **Данные могут быть неполными**
   - Не все игры могут быть в базе
   - Проверяйте актуальность перед использованием

## 🐛 Troubleshooting

**"No props for today"**
```bash
# Проверьте букмекерские источники
python src/Process-Data/Get_Player_Props.py --fetch --show-today
```

**"Датасет пуст"**
```bash
# Убедитесь что есть исторические данные
python src/Process-Data/Get_Player_Stats.py --fetch-games 60
python src/Process-Data/Create_Player_Props_Dataset.py --days-back 180
```

**"Model not found"**
```bash
# Обучите модель
python src/Train-Models/XGBoost_Model_PlayerProps.py --trials 20
```

**HTTP ошибки при запросах**
```bash
# Проверьте интернет соединение
# Проверьте что NBA API доступен
curl -I https://stats.nba.com/stats/leaguedashplayerstats
```

## 📈 Улучшения системы

Идеи для развития:

1. **Добавить больше фич:**
   - Defensive rating соперника
   - Pace игры команды
   - Back-to-back игры
   - Историю против конкретного соперника

2. **Другие типы props:**
   - Rebounds (подборы)
   - Assists (передачи)
   - Points + Rebounds + Assists
   - Three-pointers made

3. **Ансамблирование:**
   - Комбинировать XGBoost + Neural Network
   - Использовать stacking

4. **Live betting:**
   - Учитывать live статистику
   - In-game props

## 📞 Поддержка

Если что-то не работает:

1. Проверьте что все зависимости установлены
2. Убедитесь что базы данных инициализированы
3. Проверьте что есть данные за последние 30+ дней
4. Посмотрите логи ошибок

## 📄 Лицензия

MIT License - используйте на свой риск.

---

**Удачи! 🍀**

Remember: The house always has an edge. Play responsibly.
