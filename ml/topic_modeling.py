"""
Topic modeling utilities.

We start with NMF + TF-IDF because it is:
- stable and deterministic
- fast for API use
- easy to deploy

Later, we can add LDA ans LSA as upgrades.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple


from sklearn.feature_extraction.text import TfidfVectorizer # for TF-IDF
from sklearn.decomposition import NMF # for NMF model


@dataclass
class TopicModelConfig:
    n_topics: int = 3
    n_top_words: int = 12

    # Vectorizer params
    max_features: int = 5000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: Tuple[int, int] = (1, 2)

    # Model params
    random_state: int = 42
    max_iter: int = 400
    
#-------- Train Non-Negative Matrix Factorization 
def fit_nmf_topics(cleaned_texts: List[str], config: TopicModelConfig):
    
    """
    Fit TF-IDF + NMF on cleaned texts.

    Returns a dictionary containing:
    - vectorizer
    - model
    - topic_words
    - doc_topic_matrix
    """
    
    # Filter out empty docs
    non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t.strip()]
    texts = [cleaned_texts[i] for i in non_empty_indices]
    
    if len(texts) < 5:
        raise ValueError("Not enough non-empty documents to train topics. Try lowering preprocessing filters.")

    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        ngram_range=config.ngram_range,
    )
    # Main DTM(TF-IDF) matrix X
    DTM = vectorizer.fit_transform(texts)
    
    model = NMF(
        n_components=config.n_topics,
        random_state=config.random_state,
        max_iter=config.max_iter,
        init="nndsvda",
    )
    
    W = model.fit_transform(DTM)  # doc-topic matrix
    H = model.components_        # topic-term matrix
    
    feature_names = vectorizer.get_feature_names_out()
    
    topic_words: List[List[str]] = []
    for topic_idx in range(config.n_topics):
        top_indices = H[topic_idx].argsort()[::-1][:config.n_top_words]
        top_terms = [feature_names[i] for i in top_indices]
        topic_words.append(top_terms)
        
    return {
        "vectorizer": vectorizer,
        "model": model,
        "topic_words": topic_words,
        "doc_topic_matrix": W,
        "kept_doc_indices": non_empty_indices,  # mapping back to original docs
    }
    
# Topics
def dominant_topics(doc_topic_matrix, default_topic: int = -1) -> List[int]:
    """
    Return dominant topic index per document based on max weight.
    """
    return doc_topic_matrix.argmax(axis=1).tolist()