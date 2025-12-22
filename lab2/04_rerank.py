"""
Переранжирование результатов с использованием Cross-Encoder
- Использует модель из SentenceTransformers
- Переранжирует топ-50 документов из BM25
- Вычисляет метрики до и после ранжирования
"""

import json
import os
from typing import Dict, List, Tuple

import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm


def load_corpus(corpus_path: str) -> Dict[str, dict]:
    """Загружает корпус документов."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_queries(queries_path: str) -> Dict[str, str]:
    """Загружает запросы."""
    with open(queries_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Загружает qrels."""
    with open(qrels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_bm25_results(results_path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Загружает результаты BM25."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {qid: [(d, s) for d, s in docs] for qid, docs in data.items()}


def rerank_with_cross_encoder(
    queries: Dict[str, str],
    bm25_results: Dict[str, List[Tuple[str, float]]],
    corpus: Dict[str, dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k_rerank: int = 50,
    batch_size: int = 32
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Переранжирует результаты BM25 с помощью Cross-Encoder.
    
    Args:
        queries: Словарь запросов {query_id: query_text}
        bm25_results: Результаты BM25 {query_id: [(doc_id, score), ...]}
        corpus: Корпус документов
        model_name: Название модели Cross-Encoder
        top_k_rerank: Количество документов для переранжирования
        batch_size: Размер батча
    
    Returns:
        Переранжированные результаты
    """
    print(f"\nЗагрузка модели Cross-Encoder: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Используемое устройство: {device}")
    
    model = CrossEncoder(model_name, max_length=512, device=device)
    
    reranked_results = {}
    
    for query_id, doc_scores in tqdm(bm25_results.items(), desc="Переранжирование"):
        query_text = queries.get(query_id, "")
        
        if not query_text:
            reranked_results[query_id] = doc_scores
            continue
        
        # Берём топ-k документов для переранжирования
        candidates = doc_scores[:top_k_rerank]
        
        # Формируем пары (query, document)
        pairs = []
        doc_ids = []
        for doc_id, _ in candidates:
            if doc_id in corpus:
                doc_text = corpus[doc_id].get("title", "") + " " + corpus[doc_id]["text"]
                # Ограничиваем длину документа
                doc_text = doc_text[:1000]
                pairs.append([query_text, doc_text])
                doc_ids.append(doc_id)
        
        if not pairs:
            reranked_results[query_id] = doc_scores
            continue
        
        # Получаем скоры от Cross-Encoder
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        
        # Сортируем по новым скорам
        doc_score_pairs = list(zip(doc_ids, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_results[query_id] = [(doc_id, float(score)) for doc_id, score in doc_score_pairs]
    
    return reranked_results


def rerank_with_sentence_transformer(
    queries: Dict[str, str],
    bm25_results: Dict[str, List[Tuple[str, float]]],
    corpus: Dict[str, dict],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    top_k_rerank: int = 50
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Переранжирует результаты с помощью Bi-Encoder (Sentence Transformer).
    Вычисляет косинусное сходство между эмбеддингами запроса и документов.
    """
    from sentence_transformers import SentenceTransformer, util
    
    print(f"\nЗагрузка модели Sentence Transformer: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Используемое устройство: {device}")
    
    model = SentenceTransformer(model_name, device=device)
    
    reranked_results = {}
    
    for query_id, doc_scores in tqdm(bm25_results.items(), desc="Переранжирование (Bi-Encoder)"):
        query_text = queries.get(query_id, "")
        
        if not query_text:
            reranked_results[query_id] = doc_scores
            continue
        
        candidates = doc_scores[:top_k_rerank]
        
        # Собираем тексты документов
        doc_texts = []
        doc_ids = []
        for doc_id, _ in candidates:
            if doc_id in corpus:
                doc_text = corpus[doc_id].get("title", "") + " " + corpus[doc_id]["text"]
                doc_texts.append(doc_text[:1000])
                doc_ids.append(doc_id)
        
        if not doc_texts:
            reranked_results[query_id] = doc_scores
            continue
        
        # Вычисляем эмбеддинги
        query_embedding = model.encode(query_text, convert_to_tensor=True)
        doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)
        
        # Косинусное сходство
        scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
        
        # Сортируем
        doc_score_pairs = list(zip(doc_ids, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_results[query_id] = [(doc_id, float(score)) for doc_id, score in doc_score_pairs]
    
    return reranked_results


def compute_metrics_manual(
    search_results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 5
) -> Dict[str, float]:
    """Вычисляет метрики вручную."""
    precisions = []
    recalls = []
    aps = []
    rrs = []
    
    for query_id, doc_scores in search_results.items():
        if query_id not in qrels:
            continue
        
        relevant_docs = set(
            doc_id for doc_id, rel in qrels[query_id].items() if rel > 0
        )
        
        if not relevant_docs:
            continue
        
        top_k_docs = [doc_id for doc_id, _ in doc_scores[:k]]
        
        relevant_in_top_k = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)
        precision = relevant_in_top_k / k
        precisions.append(precision)
        
        recall = relevant_in_top_k / len(relevant_docs) if relevant_docs else 0
        recalls.append(recall)
        
        ap = 0.0
        num_relevant = 0
        for i, doc_id in enumerate(top_k_docs):
            if doc_id in relevant_docs:
                num_relevant += 1
                ap += num_relevant / (i + 1)
        ap = ap / min(len(relevant_docs), k) if relevant_docs else 0
        aps.append(ap)
        
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
    run_id: str
):
    """Сохраняет результаты в формате TREC."""
    with open(output_path, "w", encoding="utf-8") as f:
        for query_id, doc_scores in search_results.items():
            for rank, (doc_id, score) in enumerate(doc_scores, 1):
                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_id}\n")
    print(f"Результаты сохранены: {output_path}")


def print_comparison(bm25_metrics: dict, reranked_metrics: dict, method_name: str):
    """Выводит сравнение метрик."""
    print("\n" + "=" * 70)
    print(f"СРАВНЕНИЕ: BM25 vs {method_name}")
    print("=" * 70)
    print(f"{'Метрика':<20} {'BM25':>12} {method_name:>12} {'Δ':>12} {'%':>10}")
    print("-" * 70)
    
    for metric in sorted(bm25_metrics.keys()):
        bm25_val = bm25_metrics[metric]
        rerank_val = reranked_metrics.get(metric, 0)
        delta = rerank_val - bm25_val
        pct = (delta / bm25_val * 100) if bm25_val != 0 else 0
        sign = "+" if delta > 0 else ""
        print(f"{metric:<20} {bm25_val:>12.4f} {rerank_val:>12.4f} {sign}{delta:>11.4f} {sign}{pct:>9.1f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    # Загружаем данные
    print("Загрузка данных...")
    corpus = load_corpus("data/corpus.json")
    queries = load_queries("data/queries.json")
    qrels = load_qrels("data/qrels.json")
    bm25_results = load_bm25_results("results/bm25_results.json")
    
    print(f"Корпус: {len(corpus)} документов")
    print(f"Запросы: {len(queries)}")
    print(f"BM25 результаты: {len(bm25_results)} запросов")
    
    # Метрики BM25
    bm25_metrics = compute_metrics_manual(bm25_results, qrels, k=5)
    print("\n--- Метрики BM25 ---")
    for name, value in sorted(bm25_metrics.items()):
        print(f"{name}: {value:.4f}")
    
    os.makedirs("results", exist_ok=True)
    
    # === Переранжирование с Cross-Encoder ===
    print("\n" + "=" * 70)
    print("ПЕРЕРАНЖИРОВАНИЕ С CROSS-ENCODER")
    print("=" * 70)
    
    # Используем мультиязычную модель для русского языка
    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    reranked_cross = rerank_with_cross_encoder(
        queries, bm25_results, corpus,
        model_name=cross_encoder_model,
        top_k_rerank=50
    )
    
    # Сохраняем результаты
    save_run_file(reranked_cross, "results/cross_encoder_run.txt", "cross_encoder")
    with open("results/cross_encoder_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {qid: [(d, s) for d, s in docs] for qid, docs in reranked_cross.items()},
            f, ensure_ascii=False, indent=2
        )
    
    # Метрики после Cross-Encoder
    cross_metrics = compute_metrics_manual(reranked_cross, qrels, k=5)
    print_comparison(bm25_metrics, cross_metrics, "Cross-Encoder")
    
    # === Переранжирование с Sentence Transformer (Bi-Encoder) ===
    print("\n" + "=" * 70)
    print("ПЕРЕРАНЖИРОВАНИЕ С SENTENCE TRANSFORMER (BI-ENCODER)")
    print("=" * 70)
    
    # Мультиязычная модель
    biencoder_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    reranked_biencoder = rerank_with_sentence_transformer(
        queries, bm25_results, corpus,
        model_name=biencoder_model,
        top_k_rerank=50
    )
    
    # Сохраняем результаты
    save_run_file(reranked_biencoder, "results/biencoder_run.txt", "biencoder")
    with open("results/biencoder_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {qid: [(d, s) for d, s in docs] for qid, docs in reranked_biencoder.items()},
            f, ensure_ascii=False, indent=2
        )
    
    # Метрики после Bi-Encoder
    biencoder_metrics = compute_metrics_manual(reranked_biencoder, qrels, k=5)
    print_comparison(bm25_metrics, biencoder_metrics, "Bi-Encoder")
    
    # === Итоговая таблица ===
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)
    print(f"{'Метрика':<20} {'BM25':>15} {'Cross-Encoder':>15} {'Bi-Encoder':>15}")
    print("-" * 80)
    
    for metric in sorted(bm25_metrics.keys()):
        print(f"{metric:<20} {bm25_metrics[metric]:>15.4f} {cross_metrics.get(metric, 0):>15.4f} {biencoder_metrics.get(metric, 0):>15.4f}")
    
    print("=" * 80)
    
    # Сохраняем все метрики
    all_metrics = {
        "bm25": bm25_metrics,
        "cross_encoder": cross_metrics,
        "biencoder": biencoder_metrics
    }
    with open("results/all_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nВсе метрики сохранены в results/all_metrics.json")
