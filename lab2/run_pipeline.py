import argparse
import subprocess
import sys
import time


def run_script(script_name: str, description: str):
    """Запускает Python скрипт."""
    print(f"\nЭтап: {description}")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Ошибка: скрипт {script_name} завершился с кодом {result.returncode}")
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
    
    if not any([args.load, args.index, args.search, args.rerank]):
        args.all = True
    
    print("Лабораторная работа №2")
    print("Гибридный поиск и ранжирование (Retrieve & Re-Rank)")
    print("Датасет: Mr. TyDi (русский язык)")
    
    total_start = time.time()
    
    if args.all or args.load:
        run_script("load_data_smart.py", "Загрузка данных Mr. TyDi (сбалансированный корпус 100k)")
    
    if args.all or args.index:
        run_script("index_elasticsearch.py", "Индексация в ElasticSearch")
    
    if args.all or args.search:
        run_script("search_bm25.py", "Поиск BM25 и вычисление метрик")
    
    if args.all or args.rerank:
        run_script("rerank.py", "Переранжирование с нейросетями")
    
    total_elapsed = time.time() - total_start
    
    print("\nПайплайн завершён")


if __name__ == "__main__":
    main()
