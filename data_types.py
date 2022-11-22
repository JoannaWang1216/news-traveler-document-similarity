from typing import Literal, Optional, TypedDict


class OppositeNewsRequest(TypedDict):
    content: str
    keyword: str


class SentimentRequest(TypedDict):
    content: str


class ErrorResponse(TypedDict):
    message: str


class InternalErrorResponse(TypedDict):
    message: str
    debug: str


class GatewayTimeoutResponse(TypedDict):
    reason: str


class Sentiment(TypedDict):
    kind: Literal["positive", "neutral", "negative"]
    confidence: float


class News(TypedDict):
    source: Optional[str]
    author: Optional[str]
    title: str
    content: str
    url: str
    urlToImage: Optional[str]
    publishedAt: Optional[str]


class NewsWithSentiment(TypedDict):
    source: str
    author: Optional[str]
    title: str
    description: str
    content: str
    url: str
    urlToImage: Optional[str]
    publishedAt: Optional[str]
    sentiment: Sentiment


class NewsWithSentimentAndBias(TypedDict):
    source: str
    author: Optional[str]
    title: str
    description: str
    content: str
    url: str
    urlToImage: Optional[str]
    publishedAt: Optional[str]
    sentiment: Sentiment
    bias: float


class SearchOkResponse(TypedDict):
    count: int
    results: list[News]


class OppositeSentimentNewsOkResponse(TypedDict):
    count: int
    results: list[NewsWithSentiment]


class SentimentAndBiasOkResponse(TypedDict):
    sentiment: Sentiment
    bias: float


class SentimentOkResponse(TypedDict):
    sentiment: Sentiment


class BiasOkResponse(TypedDict):
    bias: float


class NewsDataApiParam(TypedDict):
    country: Optional[str]
    category: Optional[str]
    language: Optional[str]
    domain: Optional[str]
    q: str
    qInTitle: Optional[str]
    page: Optional[int]


class NewsApiParam(TypedDict):
    q: str
    searchIn: Optional[list[str]]
    sources: Optional[list[str]]
    domains: Optional[list[str]]
    excludeDomains: Optional[list[str]]
    startFrom: Optional[str]
    endTo: Optional[str]
    language: Optional[str]
    sortBy: Optional[Literal["relevancy", "popularity", "publishedAt"]]
    pageSize: Optional[int]
    page: Optional[int]


class SearchSuccess(TypedDict):
    news: list[News]


class SearchError(TypedDict):
    status_code: int
    message: str


class SentimentAndBiasSuccess(TypedDict):
    sentiment: Sentiment
    bias: float


class SentimentAndBiasError(TypedDict):
    status_code: int
    message: str
    sentiment: Optional[Sentiment]
    bias: Optional[float]


class BiasAnalysisSuccess(TypedDict):
    value: float


class BiasAnalysisError(TypedDict):
    status_code: int
    message: str


class SentimentAnalysisSuccess(TypedDict):
    value: Sentiment


class SentimentAnalysisError(TypedDict):
    status_code: int
    message: str


class SentimentAnalysisResult(TypedDict):
    status_code: int
    value: Sentiment
