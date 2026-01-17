# -*- coding: utf-8 -*-
"""
Entity Filtering (EF) module for processing retrieved captions
This module extracts key entities from retrieved captions using frequency-based filtering,
similar to IFCap's Entity Filtering approach.
"""

import nltk
import numpy as np
import zipfile
from typing import List, Tuple, Optional
from nltk.stem import WordNetLemmatizer

# Download required NLTK data if not available or corrupted
def _ensure_nltk_data():
    """Ensure NLTK data is available, re-download if corrupted."""
    nltk_data_list = [
        ('punkt', 'tokenizers/punkt'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('wordnet', 'corpora/wordnet')
    ]
    
    for data_name, data_path in nltk_data_list:
        try:
            nltk.data.find(data_path)
        except (LookupError, zipfile.BadZipFile, Exception):
            # Data not found or corrupted, try to download
            try:
                nltk.download(data_name, quiet=True)
            except Exception:
                # If download fails, continue - will fail at runtime if needed
                pass

# Ensure NLTK data is available (lazy check - only when actually needed)
# We'll check in the function when needed instead of at import time
# This avoids blocking imports if NLTK data is corrupted


def extract_entities_from_captions(
    captions: List[str],
    lemmatizer: Optional[WordNetLemmatizer] = None
) -> List[Tuple[int, str]]:
    """
    Extract entities from a list of captions and count their frequencies.
    
    Args:
        captions: List of caption strings
        lemmatizer: Optional WordNetLemmatizer instance (created if None)
    
    Returns:
        List of tuples (frequency, entity) sorted by frequency in descending order
    """
    # Ensure NLTK data is available (lazy check)
    _ensure_nltk_data()
    
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    
    detected_entities = {}
    
    for cap in captions:
        # Tokenize and POS tag
        pos_tags = nltk.pos_tag(nltk.word_tokenize(cap))
        
        # Extract nouns (NN, NNS)
        for word, pos in pos_tags:
            if pos == 'NN' or pos == 'NNS':
                entity = lemmatizer.lemmatize(word.lower().strip())
                if entity not in detected_entities:
                    detected_entities[entity] = 0
                detected_entities[entity] += 1
    
    # Convert to list of (frequency, entity) tuples and sort by frequency
    if detected_entities:
        freq_entity_list = [(freq, entity) for entity, freq in detected_entities.items()]
        freq_entity_list.sort(reverse=True)
        return freq_entity_list
    else:
        return []


def filter_entities_by_threshold(
    freq_entities: List[Tuple[int, str]],
    threshold: int = 1
) -> List[str]:
    """
    Filter entities by frequency threshold.
    
    Args:
        freq_entities: List of (frequency, entity) tuples
        threshold: Minimum frequency threshold
    
    Returns:
        List of entity names (strings) with frequency >= threshold
    """
    filtered = list(filter(lambda x: x[0] >= threshold, freq_entities))
    if filtered:
        _, entities = zip(*filtered)
        return list(entities)
    else:
        return []


def filter_entities_normal(
    freq_entities: List[Tuple[int, str]],
    alpha: float = 1.0
) -> List[str]:
    """
    Filter entities using normal distribution (mean + alpha * std).
    
    Args:
        freq_entities: List of (frequency, entity) tuples
        alpha: Multiplier for standard deviation
    
    Returns:
        List of filtered entity names
    """
    if not freq_entities:
        return []
    
    freq, _ = zip(*freq_entities)
    mean = np.mean(freq)
    std = np.std(freq)
    threshold = mean + std * alpha
    
    filtered = list(filter(lambda x: x[0] > threshold, freq_entities))
    if filtered:
        _, entities = zip(*filtered)
        return list(entities)
    else:
        return []


def filter_entities_log_normal(
    freq_entities: List[Tuple[int, str]],
    alpha: float = 1.0
) -> List[str]:
    """
    Filter entities using log-normal distribution (log_mean + alpha * log_std).
    
    Args:
        freq_entities: List of (frequency, entity) tuples
        alpha: Multiplier for standard deviation
    
    Returns:
        List of filtered entity names
    """
    if not freq_entities:
        return []
    
    freq, _ = zip(*freq_entities)
    log_freq = np.log(freq)
    mean = np.mean(log_freq)
    var = np.mean(np.square(log_freq - mean))
    std = var ** 0.5
    threshold = mean + std * alpha
    
    filtered = list(filter(lambda x: np.log(x[0]) > threshold, freq_entities))
    if filtered:
        _, entities = zip(*filtered)
        return list(entities)
    else:
        return []


def retrieve_concepts_ef(
    select_memory_captions: List[str],
    filter_method: str = 'threshold',
    threshold: int = 1,
    alpha: float = 1.0,
    max_entities: int = 5
) -> List[str]:
    """
    Extract key concepts from retrieved captions using Entity Filtering (EF) method.
    
    This function replaces the complex Filter stage of Retrieve-then-Filter
    with a simpler frequency-based Entity Filtering approach.
    
    Args:
        select_memory_captions: List of retrieved caption strings
        filter_method: Filtering method ('threshold', 'normal', 'log_normal')
        threshold: Frequency threshold (used when filter_method='threshold')
        alpha: Alpha parameter for normal/log_normal filtering
        max_entities: Maximum number of entities to return
    
    Returns:
        List of key entity names (strings)
    
    Example:
        >>> captions = [
        ...     "A cute girl is sitting on a bed with a pink blanket.",
        ...     "A young woman lies on a bed covered with a pink blanket."
        ... ]
        >>> entities = retrieve_concepts_ef(captions, filter_method='threshold', threshold=1)
        >>> print(entities)  # ['bed', 'blanket', 'girl', ...]
    """
    # Step 1: Extract entities and count frequencies
    freq_entities = extract_entities_from_captions(select_memory_captions)
    
    if not freq_entities:
        return []
    
    # Step 2: Filter entities based on method
    if filter_method == 'threshold':
        filtered_entities = filter_entities_by_threshold(freq_entities, threshold)
    elif filter_method == 'normal':
        filtered_entities = filter_entities_normal(freq_entities, alpha)
    elif filter_method == 'log_normal':
        filtered_entities = filter_entities_log_normal(freq_entities, alpha)
    else:
        raise ValueError(f"Unknown filter_method: {filter_method}. Must be 'threshold', 'normal', or 'log_normal'")
    
    # Step 3: Limit to max_entities
    if len(filtered_entities) > max_entities:
        filtered_entities = filtered_entities[:max_entities]
    
    return filtered_entities


# Alias for compatibility
retrieve_concepts_entity_filtering = retrieve_concepts_ef
