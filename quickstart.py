#!/usr/bin/env python3
"""
Скрипт быстрого старта системы player props prediction.
Автоматически проходит все этапы установки и обучения.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent


def run_command(description, command_list, check=True):
    print(f"\n{'='*70}")
    print(f"⏳ {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            command_list,
            check=check,
            text=True,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - ГОТОВО")
            return True
        else:
            print(f"❌ {description} - ОШИБКА (код: {result.returncode})")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка: {e}")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False


def check_dependencies():
    """Проверить установку зависимостей"""
    print("\n🔍 Проверка зависимостей...")
    
    required = ['pandas', 'numpy', 'xgboost', 'sklearn', 'colorama', 'requests']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Отсутствующие пакеты: {', '.join(missing)}")
        print("Установите их командой:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✅ Все зависимости установлены")
    return True


def main():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║      NBA Player Props Prediction - Быстрый старт               ║
    ║                                                                ║
    ║  Этот скрипт автоматически настроит систему для предсказания  ║
    ║  over/under линий очков игроков NBA                            ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    start_time = datetime.now()
    
    # Проверка зависимостей
    if not check_dependencies():
        print("\n❌ Сначала установите зависимости")
        return
    
    steps_completed = 0
    total_steps = 6
    
    # Шаг 1: Инициализация баз данных
    if run_command(
        f"Шаг 1/{total_steps}: Инициализация баз данных",
        ["python", "src/Process-Data/Get_Player_Stats.py", "--init"]
    ):
        steps_completed += 1
    
    if run_command(
        f"Шаг 1/{total_steps}: Инициализация базы props",
        ["python", "src/Process-Data/Get_Player_Props.py", "--init"]
    ):
        steps_completed += 1
    
    print("\n⚠️  ВНИМАНИЕ: Следующий шаг займёт 10-15 минут")
    print("API NBA имеет rate limiting, нужно делать паузы между запросами")
    
    response = input("\nПродолжить? (y/n): ")
    if response.lower() != 'y':
        print("Установка прервана пользователем")
        return
    
    if run_command(
        f"Шаг 2/{total_steps}: Загрузка статистики игроков (60 дней)",
        ["python", "src/Process-Data/Get_Player_Stats.py", "--full-update", "--fetch-games", "60"]
    ):
        steps_completed += 1
    
    if run_command(
        f"Шаг 3/{total_steps}: Загрузка букмекерских линий",
        ["python", "src/Process-Data/Get_Player_Props.py", "--fetch"]
    ):
        steps_completed += 1
    
    if run_command(
        f"Шаг 4/{total_steps}: Создание обучающего датасета",
        ["python", "src/Process-Data/Create_Player_Props_Dataset.py", "--days-back", "180", "--analyze"]
    ):
        steps_completed += 1
    
    print("\n⚠️  ВНИМАНИЕ: Обучение модели займёт 5-10 минут")
    
    response = input("\nОбучить модель сейчас? (y/n): ")
    if response.lower() == 'y':
        if run_command(
            f"Шаг 5/{total_steps}: Обучение XGBoost модели (50 trials)",
            ["python", "src/Train-Models/XGBoost_Model_PlayerProps.py", "--trials", "50", "--splits", "5"]
        ):
            steps_completed += 1
    else:
        print("Обучение пропущено. Запустите позже:")
        print("  python src/Train-Models/XGBoost_Model_PlayerProps.py --trials 50")
    
    if steps_completed >= 4:
        print("\n🎯 Попробуем получить предсказания...")
        run_command(
            f"Шаг 6/{total_steps}: Тестовый прогон предсказаний",
            ["python", "predict_player_props.py", "--min-edge", "0"],
            check=False
        )
    
    # Итоги
    elapsed = datetime.now() - start_time
    
    print(f"\n{'='*70}")
    print(f"📊 ИТОГИ УСТАНОВКИ")
    print(f"{'='*70}")
    print(f"Завершено шагов: {steps_completed}/{total_steps}")
    print(f"Затрачено времени: {elapsed}")
    
    if steps_completed == total_steps:
        print(f"\n✅ Установка успешно завершена!")
        print("\nЧто дальше:")
        print("  1. Получите предсказания:")
        print("     python predict_player_props.py")
        print("\n  2. Настройте ежедневное обновление:")
        print("     - Создайте cron job для обновления данных")
        print("     - См. PLAYER_PROPS_README.md для деталей")
        print("\n  3. Изучите систему:")
        print("     less PLAYER_PROPS_README.md")
    else:
        print(f"\n⚠️  Установка не полностью завершена")
        print("Некоторые шаги были пропущены или завершились с ошибкой")
        print("Проверьте логи выше для деталей")
    
    print(f"\n{'='*70}")
    print("Документация: PLAYER_PROPS_README.md")
    print("Удачи! 🍀")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Установка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Критическая ошибка: {e}")
        sys.exit(1)
