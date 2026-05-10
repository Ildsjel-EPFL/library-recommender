import pandas as pd
import numpy as np
import numpy.typing as npt

from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from data import DATA_DIR

with open(DATA_DIR / "hf_login.txt", "r") as f:
    token = f.read()
    
login(token=token)

def compute_embeddings(df : pd.DataFrame, model_choice : str = "e5", device="cpu") -> npt.NDArray[np.float32]:
    """
    Compute text embeddings for the given dataframe using the specified model choice.
    
    :param df: The input dataframe containing book information.
    :type df: pd.DataFrame
    :param model_choice: The choice of embedding model to use ("minilm", "e5", or "gemma"). Defaults to "e5".
    :type model_choice: str, optional
    :param device: The device to run the embedding model on ("cpu" or "cuda"). Defaults to "cpu".
    :type device: str, optional
    :return: A numpy array of shape (num_books, embedding_dim) containing the computed embeddings.
    :rtype: npt.NDArray[np.float32]
    """
    
    model_mapping = {
        "minilm": "paraphrase-multilingual-MiniLM-L12-v2",
        "e5": "intfloat/multilingual-e5-base",
        "gemma": "google/embeddinggemma-300m"
    }

    print(f"Preparing Text Features using {model_mapping[model_choice]}")
    rich_texts = []
    for _, row in df.iterrows():
        title = str(row.get("Title", row.get("title", ""))).strip()
        author = str(row.get("Author", row.get("author", ""))).strip()
        subjects = str(row.get("Subjects", row.get("subjects", ""))).strip() 
        genres = str(row.get("genres", "")).strip()
        summary = str(row.get("summary", "")).strip()
        
        parts = []
        if title: parts.append(f"Title: {title}.")
        if author: parts.append(f"Author: {author}.")
        if subjects: parts.append(f"Subjects: {subjects}.")
        if genres: parts.append(f"Genres: {genres}.")
        if summary: parts.append(f"Summary: {summary}.")
        
        full_text = " ".join(parts) if parts else "Unknown book."
        
        # E5 models require a prefix for document indexing
        if model_choice == "e5":
            full_text = "passage: " + full_text
            
        rich_texts.append(full_text)

    # Encode Texts
    embedder = SentenceTransformer(model_mapping[model_choice], device=device)
    embeddings = embedder.encode(rich_texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings
 