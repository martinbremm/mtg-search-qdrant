import numpy as np
import pandas as pd

from fastembed import TextEmbedding
from typing import List


def prepare_df(file_path: str = "mtgjson/data/cards.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        raise Exception(f"Could not create DataFrame, because there was no csv file found at '{file_path}'.")

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
    
    return df


def create_card_embeddings(df: pd.DataFrame, categorical_embeddings: bool = False, file_path: str = "mtgjson/data/cards_embedded.parquet"):
    if categorical_embeddings:
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

    if categorical_embeddings:
        embeddings = [np.concatenate((emb, np.array(cat_emb))) for emb, cat_emb in zip(semantic_embeddings, categorical_embeddings)]
    else:
        embeddings = semantic_embeddings

    df["embeddings"] = embeddings

    print(f"Saving embedded mtg cards to '{file_path}'")
    df.to_parquet(file_path, index=False)
    
    
if __name__ == "__main__":
    df = prepare_df()
    create_card_embeddings(df)