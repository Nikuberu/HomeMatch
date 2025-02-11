from typing import Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def generate_embeddings(input_data: Union[str, list[str]]) -> np.ndarray:
    # Load the Sentence Transformer model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    # Generate embeddings
    embeddings = model.encode(input_data)
    return embeddings


def connect_listing(row: pd.Series) -> str:
    # Connect listing data
    return (
        f"Neighborhood: {row['Neighborhood']}\n"
        f"Price: {row['Price']}\n"
        f"Bedrooms: {row['Bedrooms']}\n"
        f"Bathrooms: {row['Bathrooms']}\n"
        f"House Size: {row['House Size']}\n"
        f"Description: {row['Description']}\n"
        f"Neighborhood Description: {row['Neighborhood Description']}"
    )


def connect_lancedb_result(row: pd.Series) -> str:
    # Connect listing data
    return (
        f"Neighborhood: {row['neighborhood']}\n"
        f"Price: {row['price']}\n"
        f"Bedrooms: {row['bedrooms']}\n"
        f"Bathrooms: {row['bathrooms']}\n"
        f"House Size: {row['house_size']}\n"
        f"Description: {row['description']}\n"
        f"Neighborhood Description: {row['neighborhood_description']}"
    )
