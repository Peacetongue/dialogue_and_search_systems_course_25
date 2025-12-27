import json
import os
import random
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict


def load_mrtydi_queries_and_qrels(save_dir: str = "data"):
    """
    Загружает запросы и qrels из Mr. TyDi для русского языка.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nЗагрузка запросов Mr. TyDi (русский)...")
    
    # Загружаем dev split (содержит запросы с релевантными документами)
    dataset = load_dataset("castorini/mr-tydi", "russian", trust_remote_code=True)
    
    queries = {}
    qrels = {}  # query_id -> {doc_id: relevance}
    relevant_doc_ids = set()
    
    # Обрабатываем dev split
    if "dev" in dataset:
        dev_data = dataset["dev"]
        print(f"Dev split: {len(dev_data)} записей")
        
        for item in tqdm(dev_data, desc="Обработка dev"):
            query_id = item["query_id"]
            query_text = item["query"]
            
            queries[query_id] = query_text
            
            if query_id not in qrels:
                qrels[query_id] = {}
            
            # Положительные документы
            for doc_id in item["positive_passages"]:
                if isinstance(doc_id, dict):
                    doc_id = doc_id.get("docid", "")
                doc_id = str(doc_id)
                qrels[query_id][doc_id] = 1
                relevant_doc_ids.add(doc_id)
            
            # Отрицательные документы (если есть)
            if "negative_passages" in item:
                for doc_id in item["negative_passages"]:
                    if isinstance(doc_id, dict):
                        doc_id = doc_id.get("docid", "")
                    doc_id = str(doc_id)
                    if doc_id not in qrels[query_id]:
                        qrels[query_id][doc_id] = 0
                        relevant_doc_ids.add(doc_id)  # Добавляем и негативные
    
    # Обрабатываем test split если есть
    if "test" in dataset:
        test_data = dataset["test"]
        print(f"Test split: {len(test_data)} записей")
        
        for item in tqdm(test_data, desc="Обработка test"):
            query_id = item["query_id"]
            query_text = item["query"]
            
            if query_id not in queries:
                queries[query_id] = query_text
            
            if query_id not in qrels:
                qrels[query_id] = {}
            
            for doc_id in item["positive_passages"]:
                if isinstance(doc_id, dict):
                    doc_id = doc_id.get("docid", "")
                doc_id = str(doc_id)
                qrels[query_id][doc_id] = 1
                relevant_doc_ids.add(doc_id)
    
    print(f"\nНайдено уникальных релевантных документов: {len(relevant_doc_ids)}")
    
    # Сохраняем запросы
    queries_path = os.path.join(save_dir, "queries.json")
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    
    # Сохраняем qrels
    qrels_path = os.path.join(save_dir, "qrels.json")
    with open(qrels_path, "w", encoding="utf-8") as f:
        json.dump(qrels, f, ensure_ascii=False, indent=2)
    
    # Также сохраняем в формате TREC
    qrels_trec_path = os.path.join(save_dir, "qrels.txt")
    with open(qrels_trec_path, "w", encoding="utf-8") as f:
        for query_id, docs in qrels.items():
            for doc_id, rel in docs.items():
                f.write(f"{query_id}\t0\t{doc_id}\t{rel}\n")
    
    print(f"Запросы сохранены: {queries_path} ({len(queries)} запросов)")
    print(f"Qrels сохранены: {qrels_path}")
    print(f"Qrels (TREC): {qrels_trec_path}")
    
    return queries, qrels, relevant_doc_ids


def load_smart_corpus(
    relevant_doc_ids: set, 
    save_dir: str = "data",
    target_total: int = 100000
):
    """
    Загружает сбалансированный корпус:
    1. ВСЕ релевантные документы из qrels
    2. Случайные нерелевантные до достижения target_total (по умолчанию 100k)
    
    Соотношение: примерно 50/50 релевантные/нерелевантные
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_irrelevant = target_total - len(relevant_doc_ids)
    print(f"Loading corpus: {len(relevant_doc_ids):,} relevant + {num_irrelevant:,} irrelevant = {target_total:,} total")
    
    # Загружаем весь датасет (потоковая загрузка)
    corpus_dataset = load_dataset(
        "castorini/mr-tydi-corpus", 
        "russian", 
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    relevant_docs = {}
    irrelevant_candidates = []
    
    for doc in tqdm(corpus_dataset, desc="Processing corpus"):
        doc_id = str(doc["docid"])
        doc_data = {
            "docid": doc_id,
            "title": doc.get("title", "").strip(),
            "text": doc["text"]
        }
        
        # 1. Если это релевантный документ - обязательно берём
        if doc_id in relevant_doc_ids:
            relevant_docs[doc_id] = doc_data
            
            # Прогресс
            if len(relevant_docs) % 100 == 0:
                print(f"  Найдено релевантных: {len(relevant_docs)}/{len(relevant_doc_ids)}", end="\r")
        
        # 2. Собираем нерелевантные кандидаты (с семплированием для экономии памяти)
        elif len(irrelevant_candidates) < num_irrelevant * 2:  # Берём с запасом
            # Семплируем с вероятностью (чтобы не хранить все документы в памяти)
            if random.random() < 0.3:  # Берём ~30% нерелевантных
                irrelevant_candidates.append(doc_data)
        
        # Останавливаемся когда нашли все релевантные и достаточно кандидатов
        if len(relevant_docs) >= len(relevant_doc_ids) and len(irrelevant_candidates) >= num_irrelevant:
            break
    
    print(f"\nНайдено релевантных документов: {len(relevant_docs):,}/{len(relevant_doc_ids):,}")
    print(f"Собрано кандидатов нерелевантных: {len(irrelevant_candidates):,}")
    
    # Если не хватает релевантных - корректируем
    if len(relevant_docs) < len(relevant_doc_ids):
        missing = len(relevant_doc_ids) - len(relevant_docs)
        print(f"WARNING: {missing} relevant docs not found in corpus")
        num_irrelevant = target_total - len(relevant_docs)
        print(f"Using {num_irrelevant:,} irrelevant docs instead")
    
    # Выбираем случайные нерелевантные
    if len(irrelevant_candidates) > num_irrelevant:
        selected_irrelevant = random.sample(irrelevant_candidates, num_irrelevant)
    else:
        print(f"Warning: only {len(irrelevant_candidates):,} irrelevant candidates, need {num_irrelevant:,}")
        selected_irrelevant = irrelevant_candidates
    
    # Объединяем корпус
    documents = relevant_docs.copy()
    for doc in selected_irrelevant:
        documents[doc["docid"]] = doc
    
    print(f"Relevant: {len(relevant_docs):,}, Irrelevant: {len(selected_irrelevant):,}, Total: {len(documents):,}")
    
    # Сохраняем в JSON
    corpus_path = os.path.join(save_dir, "corpus.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"Corpus saved: {corpus_path}")
    return documents


def get_statistics(save_dir: str = "data"):
    """Выводит статистику по загруженным данным."""
    
    with open(os.path.join(save_dir, "corpus.json"), "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    with open(os.path.join(save_dir, "queries.json"), "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    with open(os.path.join(save_dir, "qrels.json"), "r", encoding="utf-8") as f:
        qrels = json.load(f)
    
    total_relevant = sum(
        sum(1 for rel in docs.values() if rel == 1)
        for docs in qrels.values()
    )
    
    # Проверяем покрытие
    all_rel_docs = set()
    for docs in qrels.values():
        all_rel_docs.update(doc_id for doc_id, rel in docs.items() if rel > 0)
    
    in_corpus = sum(1 for d in all_rel_docs if d in corpus)
    
    queries_with_rel = sum(
        1 for qid in queries 
        if qid in qrels and any(doc_id in corpus for doc_id in qrels[qid] if qrels[qid][doc_id] > 0)
    )
    
    print(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}, Total relevant pairs: {total_relevant}")
    print(f"Avg relevant per query: {total_relevant / len(queries):.2f}")
    print(f"Relevant docs in qrels: {len(all_rel_docs)}, In corpus: {in_corpus} ({100*in_corpus/len(all_rel_docs):.1f}%)")
    print(f"Queries with relevant docs in corpus: {queries_with_rel} ({100*queries_with_rel/len(queries):.1f}%)")


if __name__ == "__main__":
    save_dir = "data"
    
    queries, qrels, relevant_doc_ids = load_mrtydi_queries_and_qrels(save_dir)
    corpus = load_smart_corpus(relevant_doc_ids, save_dir, target_total=100000)
    get_statistics(save_dir)
    print("Done")
