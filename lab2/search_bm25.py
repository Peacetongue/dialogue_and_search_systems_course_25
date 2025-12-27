import json
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

from elasticsearch import Elasticsearch
from tqdm import tqdm
from trectools import TrecRun, TrecQrel, TrecEval

warnings.filterwarnings("ignore", category=FutureWarning)


def load_queries(queries_path: str) -> Dict[str, str]:
    """Загружает запросы."""
    with open(queries_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Загружает qrels."""
    with open(qrels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_bm25(
    es: Elasticsearch,
    query: str,
    index_name: str = "mrtydi_russian",
    top_k: int = 50
) -> List[Tuple[str, float]]:
    """
    Поиск документов по запросу с использованием BM25.
    Возвращает список (doc_id, score).
    """
    # Улучшенный поиск: boost для заголовков, multi_match
    result = es.search(
        index=index_name,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "text"],  # Заголовок в 3 раза важнее
                    "type": "best_fields",
                    "operator": "or"
                }
            },
            "size": top_k
        }
    )
    
    return [
        (hit["_id"], hit["_score"])
        for hit in result["hits"]["hits"]
    ]


def search_all_queries(
    es: Elasticsearch,
    queries: Dict[str, str],
    index_name: str = "mrtydi_russian",
    top_k: int = 50
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Выполняет поиск для всех запросов.
    """
    results = {}
    
    for query_id, query_text in tqdm(queries.items(), desc="Поиск BM25"):
        results[query_id] = search_bm25(es, query_text, index_name, top_k)
    
    return results


def compute_metrics_trectools(
    search_results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    run_file_path: str,
    qrels_file_path: str = "data/qrels.txt",
    k: int = 5
) -> Dict[str, float]:
    """
    Вычисляет метрики поиска с использованием trectools.
    Precision@k, Recall@k, MAP@k, MRR@k
    """
    # Загружаем run и qrels из файлов
    run = TrecRun(run_file_path)
    qrels_obj = TrecQrel(qrels_file_path)
    
    # Создаем объект для вычисления метрик
    evaluator = TrecEval(run, qrels_obj)
    
    # Вычисляем метрики
    p_at_k = evaluator.get_precision(depth=k)
    r_at_k = evaluator.get_recall(depth=k)
    map_at_k = evaluator.get_map(depth=k)
    mrr = evaluator.get_reciprocal_rank()
    
    return {
        f"precision@{k}": p_at_k,
        f"recall@{k}": r_at_k,
        f"map@{k}": map_at_k,
        f"mrr@{k}": mrr
    }


def save_run_file(
    search_results: Dict[str, List[Tuple[str, float]]],
    output_path: str,
    run_id: str = "bm25"
):
    """
    Сохраняет результаты поиска в формате TREC.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for query_id, doc_scores in search_results.items():
            for rank, (doc_id, score) in enumerate(doc_scores, 1):
                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_id}\n")
    print(f"Результаты сохранены: {output_path}")


if __name__ == "__main__":
    # Подключение к ElasticSearch
    es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
    es_port = os.getenv("ELASTICSEARCH_PORT", "9200")
    es_url = f"http://{es_host}:{es_port}"
    
    print(f"Подключение к ElasticSearch: {es_url}")
    es = Elasticsearch([es_url])
    
    # Загружаем данные
    queries = load_queries("data/queries.json")
    qrels = load_qrels("data/qrels.json")
    
    print(f"Загружено {len(queries)} запросов и {len(qrels)} qrels")
    
    # Поиск BM25 (топ-50 для последующего ранжирования)
    index_name = "mrtydi_russian"
    search_results = search_all_queries(es, queries, index_name, top_k=50)
    
    # Сохраняем результаты
    os.makedirs("results", exist_ok=True)
    save_run_file(search_results, "results/bm25_run.txt", "bm25")
    
    # Также сохраняем в JSON для последующего использования
    with open("results/bm25_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {qid: [(d, s) for d, s in docs] for qid, docs in search_results.items()},
            f,
            ensure_ascii=False,
            indent=2
        )
    
    # Вычисляем метрики
    print("\nМетрики BM25 (trectools):")
    metrics = compute_metrics_trectools(
        search_results, qrels, 
        "results/bm25_run.txt", 
        "data/qrels.txt", 
        k=5
    )
    for name, value in sorted(metrics.items()):
        print(f"{name}: {value:.4f}")
    
    # Сохраняем метрики
    with open("results/bm25_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("\nМетрики сохранены в results/bm25_metrics.json")
