"""
Основной скрипт для запуска всего пайплайна лабораторной работы
Гибридный поиск: BM25 + Re-ranking

Использование:
    python run_pipeline.py --all           # Запустить весь пайплайн
    python run_pipeline.py --load          # Только загрузка данных
    python run_pipeline.py --index         # Только индексация
    python run_pipeline.py --search        # Только поиск BM25
    python run_pipeline.py --rerank        # Только переранжирование
"""

import argparse
import subprocess
import sys
import time


def run_script(script_name: str, description: str):
    """Запускает Python скрипт."""
    print("\n" + "=" * 70)
    print(f"ЭТАП: {description}")
    print("=" * 70)
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ОШИБКА: Скрипт {script_name} завершился с кодом {result.returncode}")
        sys.exit(1)
    
    print(f"Время выполнения: {elapsed:.1f} сек")
    return result


def main():
    parser = argparse.ArgumentParser(description="Пайплайн гибридного поиска")
    parser.add_argument("--all", action="store_true", help="Запустить весь пайплайн")
    parser.add_argument("--load", action="store_true", help="Загрузка данных Mr. TyDi")
    parser.add_argument("--index", action="store_true", help="Индексация в ElasticSearch")
    parser.add_argument("--search", action="store_true", help="Поиск BM25 и метрики")
    parser.add_argument("--rerank", action="store_true", help="Переранжирование")
    
    args = parser.parse_args()
    
    # Если ничего не выбрано - запускаем всё
    if not any([args.load, args.index, args.search, args.rerank]):
        args.all = True
    
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №2")
    print("Гибридный поиск и ранжирование (Retrieve & Re-Rank)")
    print("Датасет: Mr. TyDi (русский язык)")
    print("=" * 70)
    
    total_start = time.time()
    
    if args.all or args.load:
        run_script("01_load_data.py", "Загрузка данных Mr. TyDi")
    
    if args.all or args.index:
        run_script("02_index_elasticsearch.py", "Индексация в ElasticSearch")
    
    if args.all or args.search:
        run_script("03_search_bm25.py", "Поиск BM25 и вычисление метрик")
    
    if args.all or args.rerank:
        run_script("04_rerank.py", "Переранжирование с нейросетями")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("ПАЙПЛАЙН ЗАВЕРШЁН")
    print(f"Общее время: {total_elapsed:.1f} сек ({total_elapsed / 60:.1f} мин)")
    print("=" * 70)
    print("\nРезультаты сохранены в директории 'results/':")
    print("  - bm25_run.txt           : Результаты BM25 в формате TREC")
    print("  - bm25_results.json      : Результаты BM25 в JSON")
    print("  - bm25_metrics.json      : Метрики BM25")
    print("  - cross_encoder_run.txt  : Результаты Cross-Encoder")
    print("  - biencoder_run.txt      : Результаты Bi-Encoder")
    print("  - all_metrics.json       : Сравнение всех метрик")


if __name__ == "__main__":
    main()
