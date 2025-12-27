"""
Индексация документов в ElasticSearch с использованием BM25
- Создаёт индекс с русским анализатором
- Индексирует документы пакетами
"""

import json
import os
import time
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm


def wait_for_elasticsearch(es: Elasticsearch, max_retries: int = 30, delay: int = 2):
    """Ожидает готовности ElasticSearch."""
    for i in range(max_retries):
        try:
            es.info()
            print("ElasticSearch доступен!")
            return True
        except Exception as e:
            print(f"Попытка {i + 1}/{max_retries}: ElasticSearch не готов... ({e})")
        time.sleep(delay)
    raise ConnectionError("Не удалось подключиться к ElasticSearch")


def create_index(es: Elasticsearch, index_name: str = "mrtydi_russian"):
    """
    Создаёт индекс с русским анализатором для BM25.
    """
    # Удаляем индекс, если существует
    if es.indices.exists(index=index_name):
        print(f"Удаление существующего индекса '{index_name}'...")
        es.indices.delete(index=index_name)
    
    # Настройки индекса с русским анализатором
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "filter": {
                    "russian_stop": {
                        "type": "stop",
                        "stopwords": "_russian_"
                    },
                    "russian_stemmer": {
                        "type": "stemmer",
                        "language": "russian"
                    }
                },
                "analyzer": {
                    "russian_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "russian_stop",
                            "russian_stemmer"
                        ]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "docid": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "russian_analyzer"
                },
                "text": {
                    "type": "text",
                    "analyzer": "russian_analyzer"
                },
                "title_text": {
                    "type": "text",
                    "analyzer": "russian_analyzer"
                }
            }
        }
    }
    
    es.indices.create(index=index_name, body=index_settings)
    print(f"Индекс '{index_name}' создан с русским анализатором")


def index_documents(es: Elasticsearch, corpus_path: str, index_name: str = "mrtydi_russian", batch_size: int = 1000):
    """
    Индексирует документы в ElasticSearch.
    """
    print(f"\nЗагрузка документов из {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    print(f"Всего документов для индексации: {len(corpus)}")
    
    def generate_actions():
        for doc_id, doc in corpus.items():
            yield {
                "_index": index_name,
                "_id": doc_id,
                "_source": {
                    "docid": doc_id,
                    "title": doc.get("title", ""),
                    "text": doc["text"],
                    "title_text": f"{doc.get('title', '')} {doc['text']}"
                }
            }
    
    # Индексация с помощью bulk API
    print("Индексация документов...")
    success, failed = 0, 0
    
    for ok, result in tqdm(
        helpers.streaming_bulk(
            es,
            generate_actions(),
            chunk_size=batch_size,
            raise_on_error=False
        ),
        total=len(corpus),
        desc="Индексация"
    ):
        if ok:
            success += 1
        else:
            failed += 1
    
    # Обновляем индекс
    es.indices.refresh(index=index_name)
    
    # Получаем статистику
    stats = es.indices.stats(index=index_name)
    doc_count = stats["indices"][index_name]["primaries"]["docs"]["count"]
    
    print(f"\nИндексация завершена!")
    print(f"Успешно: {success}, Ошибок: {failed}")
    print(f"Документов в индексе: {doc_count}")


def test_search(es: Elasticsearch, index_name: str = "mrtydi_russian"):
    """Тестовый поиск для проверки работы индекса."""
    query = "столица России"
    
    result = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "title_text": query
                }
            },
            "size": 5
        }
    )
    
    print(f"\nТестовый поиск: '{query}'")
    for hit in result["hits"]["hits"]:
        print(f"Score: {hit['_score']:.4f}")
        print(f"Title: {hit['_source']['title'][:100]}...")
        print(f"Text: {hit['_source']['text'][:200]}...")


if __name__ == "__main__":
    # Подключение к ElasticSearch
    es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
    es_port = os.getenv("ELASTICSEARCH_PORT", "9200")
    es_url = f"http://{es_host}:{es_port}"
    
    print(f"Подключение к ElasticSearch: {es_url}")
    es = Elasticsearch([es_url])
    
    # Ждём готовности ES
    wait_for_elasticsearch(es)
    
    # Создаём индекс
    index_name = "mrtydi_russian"
    create_index(es, index_name)
    
    # Индексируем документы
    corpus_path = "data/corpus.json"
    if os.path.exists(corpus_path):
        index_documents(es, corpus_path, index_name)
        test_search(es, index_name)
    else:
        print(f"Ошибка: файл {corpus_path} не найден. Сначала запустите load_data.py")
