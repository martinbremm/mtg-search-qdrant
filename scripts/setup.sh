docker pull qdrant/qdrant

docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

python src/qdrant_indexer.py
