import logging
import pandas as pd

from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.retrievers import BM25Retriever

class BM25Manager:
    def __init__(self, debug=False):
        logging.basicConfig(level=logging.INFO)

        self.df = pd.read_csv("mtgjson/data/cards_preprocessed.csv")
        self.loader = DataFrameLoader(self.df, page_content_column="name")

        # Use lazy load for larger table, which won't read the full table into memory
        logging.info("Loading names into B25 retriever")
        
        if debug:
            for i in self.loader.lazy_load():
                logging.info(i.page_content)
    
    def retrieve(self, query: str) -> List[Document]:
        retriever = BM25Retriever.from_documents(self.loader.load())

        logging.info(f"Retrieving BM25 documents according to query: {query}")
        return retriever.get_relevant_documents(query)