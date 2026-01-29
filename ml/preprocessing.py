from __future__ import annotations

"""
Text preprocessing utilities (production-style)
Goal:
- Provide consistent, reusable text cleaning for any dataset.
- Match my thesis-style cleaning steps (URLs removal, non-alpha cleanup, stopwords, lemmatization, ...)
- Keep everything configurable and safe.

This module is pure Python (no FastAPI imports) so it can be reused in:
- training scripts
- batch jobs
- API services
"""

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional 

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------
# NLTK setup
# -----------------------------

def ensure_nltk_resources() -> None:
    """
    Ensures required NLTK resources exist.
    If missing, downloads them.

    This prevents runtime crashes like:
    LookupError: Resource 'stopwords' not found
    """
    
    required = ["stopwords", "wordner", "omw-1.4"]
    for r in required:
        try:
            nltk.data.find(f"corpora/{r}")
        except LookupError:
            nltk.download(r) 
            
            
# -----------------------------
# Config
# -----------------------------
@dataclass
class PreprocessConfig:
    """
    Preprocessing configuration.
    I will adjust these later from the API request if needed.
    """
    
    language: str = "english"
    min_token_len: int = 4
    
    # Controls what is kept/removed
    remove_urls: bool = True
    keep_only_letters: bool = True  # if True, strip numbers/punctuation
    lowercase: bool = True
    
    # Stopwords and lemmatization
    remove_stopwords: bool = True
    extra_stopwords: List[str] = field(default_factory=list)
    lemmatize: bool = True
    
    

# -----------------------------
# Core preprocessing
# -----------------------------
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def preprocess_text(
    text: str, 
    configuration: PreprocessConfig, 
    stopwords_set: Optional[set] = None, 
    lemmatizer: Optional[WordNetLemmatizer] = None
    ) -> str:
    
    """
    Clean a single text string.

    Steps (similar to thesis pipeline):
    1) lowercase
    2) remove URLs
    3) keep only letters (optional)
    4) tokenize (simple whitespace after cleanup)
    5) remove stopwords
    6) remove short tokens (len < min_token_len)
    7) lemmatize tokens
    8) return cleaned string
    """
    
    if text is None:
        return ""
    
    text_normalized = str(text)
    
    # 1) lowercase
    if configuration.lowercase:
        text_normalized = text_normalized.lower()
        
    # 2) remove URLs
    if configuration.remove_urls:
        text_normalized = URL_REGEX.sub(" ", text_normalized)
        
    # 3) keep only letters 
    if configuration.keep_only_letters:
        # Replace anything that isn't a letter with space
        # Includes digits and punctuation
        s = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ\s]", " ", text_normalized)
        
    # Normalize whitespace
    text_normalized = re.sub(r"\s+", " ", text_normalized).strip()
    if not text_normalized:
        return ""
    
    # 4) tokenize
    tokens = text_normalized.split(" ")
    

    # Prepare stopwords set if needed
    if configuration.remove_stopwords:
        if stopwords_set is None:
            ensure_nltk_resources()
            stopwords_set = set(stopwords.words(configuration.language))
            stopwords_set.update({word.lower() for word in configuration.extra_stopwords})
    else:
        stopwords_set = set()
        
        
    
    # Prepare lemmatizer
    if configuration.lemmatize and lemmatizer is None:
        ensure_nltk_resources()
        lemmatizer = WordNetLemmatizer()

    cleaned_tokens: List[str] = []

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        
        # 5) Stopword filtering
        if configuration.remove_stopwords and token in stopwords_set:
            continue

        # 6) Token length filtering
        if len(token) < configuration.min_token_len:
            continue
        
        # 7) Lemmatization
        if configuration.lemmatize and lemmatizer is not None:
            token = lemmatizer.lemmatize(token)

        cleaned_tokens.append(token)
        
    return " ".join(cleaned_tokens)




def preprocess_many(
    texts: Iterable[str],
    configuration: PreprocessConfig,
) -> List[str]:
    """
    Preprocess multiple documents efficiently by
    reusing stopwords and lemmatizer.
    """
    ensure_nltk_resources()
    
    stopwords_set = set()
    if configuration.remove_stopwords:
        stopwords_set = set(stopwords.words(configuration.language))
        stopwords_set.update(word.lower() for word in configuration.extra_stopwords)

    
    lemmatizer = WordNetLemmatizer() if configuration.lemmatize else None

    return [
        preprocess_text(
            text,
            configuration,
            stopwords_set=stopwords_set,
            lemmatizer=lemmatizer,
        )
        for text in texts
    ]

