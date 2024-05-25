import numpy as np
import pandas as pd

from fastembed import TextEmbedding
from typing import List


df = pd.read_csv("mtgjson/data/cards.csv", low_memory=False)

df = df[[
    "colorIdentity", 
    "defense",
    "edhrecRank",
    "edhrecSaltiness",
    "isReprint",
    "keywords",
    "manaCost",
    "manaValue",
    "name", 
    "originalText", 
    "power",
    "printings",
    "rarity",
    "setCode",
    "subtypes",
    "text",
    "type",
    "types",
]]

df = df.dropna(subset="text").reset_index(drop=True)
    
df = df.drop_duplicates(subset="text").reset_index(drop=True)

df = df.replace({np.nan: None})

df.to_csv("mtgjson/data/cards_preprocessed.csv", index=False)


# creating categorical embeddings
rarity_dummy_df = pd.get_dummies(df['rarity'], prefix='rarity', dtype=int)

keywords_dummy_df = pd.get_dummies(df['keywords'], prefix='keywords', dtype=int)

subtypes_dummy_df = pd.get_dummies(df['keywords'], prefix='keywords', dtype=int)

color_dummy_df = pd.get_dummies(df['colorIdentity'], prefix='color', dtype=int)

categorical_embeddings_df = pd.concat([color_dummy_df, rarity_dummy_df, subtypes_dummy_df, keywords_dummy_df], axis=1)

categorical_embeddings: List[float] = categorical_embeddings_df.values.tolist()


# concatenating fastembed and categorical embeddings
docs: List[str] = df["text"].to_list()

embedding_model = TextEmbedding(model_name="BAAI/bge-base-en")
semantic_embeddings: List[np.ndarray] = list(embedding_model.embed(docs))


embeddings = [np.concatenate((emb, np.array(cat_emb))) for emb, cat_emb in zip(semantic_embeddings, categorical_embeddings)]


df["embeddings"] = embeddings

df.to_parquet("mtgjson/data/cards_embedded.parquet", index=False)