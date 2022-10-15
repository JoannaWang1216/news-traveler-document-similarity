import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def process_tfidf_similarity(base_document: str, documents: list, select_count: int):
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform([base_document] + documents)

    cosine_similarities = cosine_similarity(
        tfidf_matrix[0:1], tfidf_matrix[1:]
    ).flatten()

    related_docs_indices = np.argsort(cosine_similarities)[-select_count:][::-1]

    return related_docs_indices
