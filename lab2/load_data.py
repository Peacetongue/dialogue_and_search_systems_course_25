import json
import os
from datasets import load_dataset
from tqdm import tqdm


def load_mrtydi_corpus(save_dir: str = "data", max_docs: int = 100000):
    """
    Загружает корпус документов Mr. TyDi для русского языка.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Загрузка корпуса Mr. TyDi (русский, первые {max_docs} документов)...")
    corpus = load_dataset(
        "castorini/mr-tydi-corpus", 
        "russian", 
        split=f"train[:{max_docs}]", 
        trust_remote_code=True
    )
    
    print(f"Загружено документов: {len(corpus)}")
    
    # Сохраняем документы
    documents = {}
    for doc in tqdm(corpus, desc="Обработка документов"):
        doc_id = doc["docid"]
        documents[doc_id] = {
            "docid": doc_id,
            "title": doc.get("title", ""),
            "text": doc["text"]
        }
    
    # Сохраняем в JSON
    corpus_path = os.path.join(save_dir, "corpus.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"Корпус сохранён: {corpus_path} ({len(documents)} документов)")
    return documents


def load_mrtydi_queries(save_dir: str = "data"):
    """
    Загружает запросы и qrels из Mr. TyDi для русского языка.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nЗагрузка запросов Mr. TyDi (русский)...")
    
    # Загружаем dev split (содержит запросы с релевантными документами)
    dataset = load_dataset("castorini/mr-tydi", "russian", trust_remote_code=True)
    
    queries = {}
    qrels = {}  # query_id -> {doc_id: relevance}
    
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
                qrels[query_id][str(doc_id)] = 1
            
            # Отрицательные документы (если есть)
            if "negative_passages" in item:
                for doc_id in item["negative_passages"]:
                    if isinstance(doc_id, dict):
                        doc_id = doc_id.get("docid", "")
                    if str(doc_id) not in qrels[query_id]:
                        qrels[query_id][str(doc_id)] = 0
    
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
                qrels[query_id][str(doc_id)] = 1
    
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
    
    print(f"\nЗапросы сохранены: {queries_path} ({len(queries)} запросов)")
    print(f"Qrels сохранены: {qrels_path}")
    print(f"Qrels (TREC): {qrels_trec_path}")
    
    return queries, qrels


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
    
    print(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}, Total relevant pairs: {total_relevant}")
    print(f"Avg relevant per query: {total_relevant / len(queries):.2f}")


if __name__ == "__main__":
    save_dir = "data"
    
    # Загружаем корпус
    corpus = load_mrtydi_corpus(save_dir)
    
    # Загружаем запросы и qrels
    queries, qrels = load_mrtydi_queries(save_dir)
    
    # Выводим статистику
    get_statistics(save_dir)
