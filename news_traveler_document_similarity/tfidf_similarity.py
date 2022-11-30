from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def process_tfidf_similarity(base_document: str, document_to_compare: str) -> float:
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform([base_document, document_to_compare])

    cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return cos_similarity
