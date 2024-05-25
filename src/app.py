import os
import streamlit as st

from fastembed import TextEmbedding
# from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

from bm25_indexer import BM25Manager


# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# openai_client = OpenAI(
#     base_url="http://localhost:8080/v1",
#     api_key = "sk-no-key-required"
# )


# Setting up FastEmbed
embedding_model = TextEmbedding(model_name="BAAI/bge-base-en")

# Setting up OpenSearch Client
qdrant_manager = QdrantClient()

# Setting up BM25 Manager
bm25_manager = BM25Manager()


# Defining the system prompt
system_prompt = """
You are a highly advanced Magic: The Gathering information system. Given a card's details in the following format:

[Card Name]
[Card Type]
[Mana Cost]
[Card Text]

Answer questions about the specified card or provide relevant information. If the card has multiple faces (e.g., double-faced cards, flip cards), provide details for both faces.

Example:
---
Gideon Jura
Planeswalker - Gideon
{{3}}{{W}}{{W}}
[+2]: During target opponent's next turn, creatures that player controls attack Gideon Jura if able.
[0]: Until end of turn, Gideon Jura becomes a 6/6 Human Soldier creature that's still a planeswalker. Prevent all damage that would be dealt to him this turn.
[-10]: Creatures you control get +1/+1. Put a loyalty counter on each creature you control.

What is Gideon Jura's mana cost?
---

You should respond to the question based on the provided card information. If the information is insufficient or the question is not applicable, you may indicate that the system lacks the necessary details to provide a specific answer.

Here are the cards the search found:
{search_content}
"""


# Streamlit UI
st.title("Magic: The Gathering Search App")

query = st.text_area("Enter your prompt", "Show me cards that counter opponents creature spells.")

if st.button("Generate Output"):

    semantic_search_cards = qdrant_manager.search(
        collection_name="mtg-search",
        # query_filter=models.Filter(
        #     must=[
        #         models.FieldCondition(
        #             key="city",
        #             match=models.MatchValue(
        #                 value="London",
        #             ),
        #         )
        #     ]
        # ),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        query_vector=list(embedding_model.embed(query))[0],
        #limit=10,
    )
    # sorting results according to edhrec rank
    semantic_search_cards = sorted(semantic_search_cards, key=lambda x: x.payload['edhrecRank'], reverse=True)
    
    bm25_search_cards = bm25_manager.retrieve(query=query)

    # response = "".join(response_cards)
    
    # # Generating comprehensive answer with llamafile
    # completion = openai_client.chat.completions.create(
    #     model="LLaMA_CPP",
    #     messages=[
    #         {"role": "system", "content": system_prompt.format(search_content=search_content)},
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # st.text("Generated Output:")
    # st.write(completion.choices[0].message.content)
    
    st.write("Semantic Search")
    st.write([(r.payload.get("name"), r.payload.get("text")) for r in semantic_search_cards])
    
    st.write("BM25")
    st.write([(r.page_content, r.metadata.get("text")) for r in bm25_search_cards])
    