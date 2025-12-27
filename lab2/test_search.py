import os
import sys
from elasticsearch import Elasticsearch

es = Elasticsearch([f"http://{os.getenv('ELASTICSEARCH_HOST')}:9200"])
query = sys.argv[1] if len(sys.argv) > 1 else "документ"

result = es.search(index="mrtydi_russian", body={"query": {"match": {"title_text": query}}, "size": 5})

for i, hit in enumerate(result["hits"]["hits"], 1):
    print(f"{i}. {hit['_score']:.2f} - {hit['_source']['title'][:80]}")
