# Лабораторная работа №2: Гибридный поиск и ранжирование

## Описание

Реализация гибридной поисковой системы на основе архитектуры **Retrieve & Re-Rank** с использованием:
- **BM25** (ElasticSearch) для первичного поиска кандидатов
- **Cross-Encoder** и **Sentence Transformer** для переранжирования

**Датасет:** Mr. TyDi (русский язык)
- ~500K документов из русской Википедии
- ~6K запросов с релевантными документами

## Структура проекта

```
lab2/
├── docker-compose.yml          # ElasticSearch
├── requirements.txt            # Python зависимости
├── 01_load_data.py            # Загрузка данных Mr. TyDi
├── 02_index_elasticsearch.py   # Индексация в ElasticSearch
├── 03_search_bm25.py          # Поиск BM25 и метрики
├── 04_rerank.py               # Переранжирование
├── run_pipeline.py            # Запуск всего пайплайна
├── data/                      # Данные (создаётся автоматически)
│   ├── corpus.json
│   ├── queries.json
│   └── qrels.json
└── results/                   # Результаты (создаётся автоматически)
    ├── bm25_run.txt
    ├── bm25_metrics.json
    └── all_metrics.json
```

## Быстрый старт

### 1. Запустите ElasticSearch

```bash
cd lab2
docker-compose up -d
```

Дождитесь запуска (~30 сек):
```bash
curl http://localhost:9200/_cluster/health
```

### 2. Установите зависимости Python

```bash
pip install -r requirements.txt
```

### 3. Запустите пайплайн

```bash
# Весь пайплайн
python run_pipeline.py --all

# Или по частям:
python 01_load_data.py           # Загрузка данных
python 02_index_elasticsearch.py  # Индексация
python 03_search_bm25.py          # Поиск и метрики BM25
python 04_rerank.py               # Переранжирование
```

## Метрики

Вычисляются следующие метрики на @5:
- **Precision@5** - доля релевантных в топ-5
- **Recall@5** - доля найденных релевантных документов
- **MAP@5** - Mean Average Precision
- **MRR@5** - Mean Reciprocal Rank

## Модели для переранжирования

### Cross-Encoder
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Одновременно кодирует пару (запрос, документ)
- Более точный, но медленнее

### Bi-Encoder (Sentence Transformer)
- `paraphrase-multilingual-MiniLM-L12-v2`
- Отдельно кодирует запрос и документы
- Быстрее, поддерживает русский язык

## Архитектура решения

```
┌─────────────────────────────────────────────────────────────────┐
│                        RETRIEVE & RE-RANK                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Запрос   │───>│  BM25 (ES)       │───>│  Top-50 канд.    │  │
│  │           │    │  Retrieve        │    │                  │  │
│  └───────────┘    └──────────────────┘    └────────┬─────────┘  │
│                                                     │            │
│                                                     ▼            │
│                   ┌──────────────────┐    ┌──────────────────┐  │
│                   │  Cross-Encoder/  │───>│  Top-5 релев.    │  │
│                   │  Bi-Encoder      │    │  результаты      │  │
│                   │  Re-Rank         │    │                  │  │
│                   └──────────────────┘    └──────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Результаты

После выполнения пайплайна результаты сохраняются в `results/all_metrics.json`:

```json
{
  "bm25": {
    "precision@5": 0.XXX,
    "recall@5": 0.XXX,
    "map@5": 0.XXX,
    "mrr@5": 0.XXX
  },
  "cross_encoder": { ... },
  "biencoder": { ... }
}
```

## Остановка ElasticSearch

```bash
docker-compose down
# Или с удалением данных:
docker-compose down -v
```

## Ссылки

- [Mr. TyDi Dataset](https://huggingface.co/datasets/castorini/mr-tydi)
- [Sentence Transformers](https://sbert.net/)
- [ElasticSearch Russian Analyzer](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html)
