"""
Поиск BM25 и вычисление метрик
- Поиск для всех запросов
- Вычисление Precision@5, Recall@5, MAP@5, MRR@5
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from elasticsearch import Elasticsearch
from tqdm import tqdm
import pytrec_eval


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
    result = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "title_text": {
                        "query": query,
                        "operator": "or"
                    }
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


def compute_metrics(
    search_results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 5
) -> Dict[str, float]:
    """
    Вычисляет метрики поиска с использованием pytrec_eval.
    Precision@k, Recall@k, MAP@k, MRR@k
    """
    # Преобразуем результаты в формат для pytrec_eval
    run = {}
    for query_id, doc_scores in search_results.items():
        run[query_id] = {
            doc_id: score
            for doc_id, score in doc_scores[:k]
        }
    
    # Преобразуем qrels (оставляем только релевантные документы)
    qrels_binary = {}
    for query_id, docs in qrels.items():
        qrels_binary[query_id] = {
            doc_id: rel for doc_id, rel in docs.items() if rel > 0
        }
    
    # Оставляем только запросы, которые есть в обоих наборах
    common_queries = set(run.keys()) & set(qrels_binary.keys())
    run = {qid: run[qid] for qid in common_queries if qid in run}
    qrels_binary = {qid: qrels_binary[qid] for qid in common_queries if qid in qrels_binary}
    
    # Метрики
    metrics_to_compute = {
        f"P_{k}",
        f"recall_{k}",
        f"map_cut_{k}",
        f"recip_rank"
    }
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_binary, metrics_to_compute)
    results = evaluator.evaluate(run)
    
    # Усредняем метрики по всем запросам
    metrics = defaultdict(float)
    for query_id, query_metrics in results.items():
        for metric_name, value in query_metrics.items():
            metrics[metric_name] += value
    
    num_queries = len(results)
    for metric_name in metrics:
        metrics[metric_name] /= num_queries
    
    return dict(metrics)


def compute_metrics_manual(
    search_results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 5
) -> Dict[str, float]:
    """
    Ручное вычисление метрик для проверки.
    """
    precisions = []
    recalls = []
    aps = []  # Average Precision
    rrs = []  # Reciprocal Rank
    
    for query_id, doc_scores in search_results.items():
        if query_id not in qrels:
            continue
        
        relevant_docs = set(
            doc_id for doc_id, rel in qrels[query_id].items() if rel > 0
        )
        
        if not relevant_docs:
            continue
        
        # Получаем топ-k результатов
        top_k_docs = [doc_id for doc_id, _ in doc_scores[:k]]
        
        # Precision@k
        relevant_in_top_k = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)
        precision = relevant_in_top_k / k
        precisions.append(precision)
        
        # Recall@k
        recall = relevant_in_top_k / len(relevant_docs) if relevant_docs else 0
        recalls.append(recall)
        
        # MAP@k (Average Precision)
        ap = 0.0
        num_relevant = 0
        for i, doc_id in enumerate(top_k_docs):
            if doc_id in relevant_docs:
                num_relevant += 1
                ap += num_relevant / (i + 1)
        ap = ap / min(len(relevant_docs), k) if relevant_docs else 0
        aps.append(ap)
        
        # MRR (Reciprocal Rank)
        rr = 0.0
        for i, doc_id in enumerate(top_k_docs):
            if doc_id in relevant_docs:
                rr = 1.0 / (i + 1)
                break
        rrs.append(rr)
    
    return {
        f"precision@{k}": sum(precisions) / len(precisions) if precisions else 0,
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0,
        f"map@{k}": sum(aps) / len(aps) if aps else 0,
        f"mrr@{k}": sum(rrs) / len(rrs) if rrs else 0
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
    es = Elasticsearch(["http://localhost:9200"])
    
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
    print("\n" + "=" * 60)
    print("МЕТРИКИ BM25 (pytrec_eval)")
    print("=" * 60)
    
    metrics_pytrec = compute_metrics(search_results, qrels, k=5)
    for name, value in sorted(metrics_pytrec.items()):
        print(f"{name}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("МЕТРИКИ BM25 (ручной расчёт)")
    print("=" * 60)
    
    metrics_manual = compute_metrics_manual(search_results, qrels, k=5)
    for name, value in sorted(metrics_manual.items()):
        print(f"{name}: {value:.4f}")
    
    # Сохраняем метрики
    with open("results/bm25_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "pytrec_eval": metrics_pytrec,
            "manual": metrics_manual
        }, f, indent=2)
    print("\nМетрики сохранены в results/bm25_metrics.json")
