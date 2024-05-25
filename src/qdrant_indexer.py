import logging
import time
import numpy as np
import pandas as pd

from typing import List
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import Batch, Distance, VectorParams


class QdrantManager(QdrantClient):
    def __init__(self):
        super().__init__(
            "localhost",
            port=6333,
            timeout=30
        )

        logging.basicConfig(level=logging.INFO)

    def create_index(self, collection_name: str) -> None:
        logging.info(f"Creating collection: {collection_name}")

        if collection_name not in qdrant_manager.get_collections():
            qdrant_manager.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            logging.info(f"Collection '{collection_name}' created successfully.")

        else:
            logging.info(f"Collection '{collection_name}' already exists.")


    def upload_embedded_documents(self,
                                  collection_name: str,
                                  documents: List[str],
                                  ids: List[int] = None,
                                  payloads: List[dict] = None,
                                  embeddings: List[np.ndarray] = None) -> None:
        logging.info(f"Uploading {len(documents)} documents into collection {collection_name} ...")

        if not embeddings:
            # https://huggingface.co/BAAI/bge-base-en
            embedding_model = TextEmbedding(model_name="BAAI/bge-base-en")
            embeddings: List[np.ndarray] = list(embedding_model.embed(docs))

        try:
            self.upload_collection(
                collection_name=collection_name,
                ids=ids,
                payload=payloads,
                vectors=embeddings,
                parallel=4,
                max_retries=3
                )

            logging.info(f"Successfully uploaded {len(documents)} documents into collection {collection_name}")
        except RuntimeError:
            logging.warning(f"RunTimeError while uploading {len(documents)} documents into collection {collection_name}")


if __name__ == "__main__":
    qdrant_manager = QdrantManager()

    collection_name = "mtg-search"

    qdrant_manager.delete_collection(collection_name=collection_name)

    qdrant_manager.create_index(collection_name=collection_name)

    df = pd.read_parquet("mtgjson/data/cards_embedded.parquet")
    df = df.replace({np.nan: None})


    docs: List[str] = df["text"].to_list()
    ids: List[int] = df.index.to_list()
    #embeddings = df["embeddings"]

    payloads: List[dict] = df.drop(columns=["embeddings"]).to_dict(orient="records")

    start = time.time()
    qdrant_manager.upload_embedded_documents(collection_name=collection_name,
                                             documents=docs,
                                             ids=ids,
                                             payloads=payloads)
    end = time.time()
    logging.info(f"Uploading embeddings took {round(end-start, 3)} seconds.")

    print(qdrant_manager.scroll(
        collection_name=collection_name,
        with_payload=False,
        with_vectors=False
        ))
