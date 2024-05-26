import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

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
        
        self.set_model("sentence-transformers/all-MiniLM-L6-v2")
        # comment this line to use dense vectors only
        self.set_sparse_model("prithivida/Splade_PP_en_v1")

    def create_index(self, collection_name: str) -> None:
        logging.info(f"Creating collection: {collection_name}")

        if collection_name not in qdrant_manager.get_collections():
            qdrant_manager.recreate_collection(
                collection_name=collection_name,
                vectors_config=qdrant_manager.get_fastembed_vector_params(),
                # comment this line to use dense vectors only
                sparse_vectors_config=qdrant_manager.get_fastembed_sparse_vector_params(),  
            )
            logging.info(f"Collection '{collection_name}' created successfully.")

        else:
            logging.info(f"Collection '{collection_name}' already exists.")


    def upload_embedded_documents(self,
                                  collection_name: str,
                                  documents: List[str],
                                  ids: List[int] = None,
                                  metadata: List[dict] = None,
                                  embeddings: List[np.ndarray] = None) -> None:
        logging.info(f"Uploading {len(documents)} documents into collection {collection_name} ...")
            
        try:
            self.add(
                collection_name=collection_name,
                ids=ids if ids else tqdm(range(len(documents))),
                documents=documents,
                metadata=metadata,
                # parallel=0,  # Use all available CPU cores to encode data. 
                # Requires wrapping code into if __name__ == '__main__' block
                )

            logging.info(f"Successfully uploaded {len(documents)} documents into collection {collection_name}")
        except RuntimeError:
            logging.warning(f"RunTimeError while uploading {len(documents)} documents into collection {collection_name}")


if __name__ == "__main__":
    qdrant_manager = QdrantManager()

    collection_name = "mtg-search"

    qdrant_manager.delete_collection(collection_name=collection_name)

    qdrant_manager.create_index(collection_name=collection_name)

    df = pd.read_csv("mtgjson/data/cards_preprocessed.csv")
    df = df.replace({np.nan: None})

    docs: List[str] = df["text"].to_list()
    
    metadata: List[dict] = df.drop(columns=["embeddings"]).to_dict(orient="records") if "embeddings" in df.columns else df.to_dict(orient="records")

    start = time.time()
    qdrant_manager.upload_embedded_documents(collection_name=collection_name,
                                             documents=docs,
                                             metadata=metadata)
    end = time.time()
    
    logging.info(f"Uploading embeddings took {round(end-start, 3)} seconds.")

    print(qdrant_manager.scroll(
        collection_name=collection_name,
        with_payload=False,
        with_vectors=False
        ))
