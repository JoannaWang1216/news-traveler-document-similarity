import json
import os
import random
from typing import Any, Callable, Literal, Optional, TypedDict, TypeVar, Union

import requests
from flask import Flask, request
from newsdataapi import NewsDataApiClient


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
    title: Optional[str]
    description: Optional[str]
    content: Optional[str]
    url: Optional[str]
    urlToImage: Optional[str]
    publishedAt: Optional[str]


class NewsWithSentiment(TypedDict):
    source: Optional[str]
    author: Optional[str]
    title: Optional[str]
    description: Optional[str]
    content: Optional[str]
    url: Optional[str]
    urlToImage: Optional[str]
    publishedAt: Optional[str]
    sentiment: Sentiment
    bias: float


class SearchResponse(TypedDict):
    count: int
    results: list[News]


class OppositeNewsResponse(TypedDict):
    count: int
    results: list[NewsWithSentiment]


class SentimentResponse(TypedDict):
    sentiment: Sentiment
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


# A workaround for not using NotRequired
def generate_newsdataapi_param(
    q,
    country=None,
    category=None,
    language=None,
    domain=None,
    q_in_title=None,
    page=None,
) -> NewsDataApiParam:
    return {
        "q": q,
        "country": country,
        "category": category,
        "language": language,
        "domain": domain,
        "qInTitle": q_in_title,
        "page": page,
    }


# A workaround for not using NotRequired
def generate_newapi_param(
    q,
    search_in=None,
    sources=None,
    domains=None,
    exclude_domains=None,
    start_from=None,
    end_to=None,
    language=None,
    sort_by=None,
    page_size=None,
    page=None,
) -> NewsApiParam:
    return {
        "q": q,
        "searchIn": search_in,
        "sources": sources,
        "domains": domains,
        "excludeDomains": exclude_domains,
        "startFrom": start_from,
        "endTo": end_to,
        "language": language,
        "sortBy": sort_by,
        "pageSize": page_size,
        "page": page,
    }


SearchParam = TypeVar("SearchParam", NewsDataApiParam, NewsApiParam)


def request_newsdataapi(params: NewsDataApiParam) -> Union[SearchSuccess, SearchError]:
    api_key = os.getenv("NEWSDATAAPI_KEY")
    api = NewsDataApiClient(apikey=api_key)
    response = api.news_api(**params)
    if response["status"] == "error":
        return {
            "status_code": 400,
            "message": f'{response["results"]["code"]}, {response["results"]["message"]}',
        }
    return {
        "news": [
            {
                "source": news["source_id"],
                "author": ",".join(news["creator"]),
                "title": news["title"],
                "description": news["description"],
                "content": news["content"],
                "url": news["link"],
                "urlToImage": news["image_url"],
                "publishedAt": news["pubDate"],
            }
            for news in response["results"]
        ]
    }


def request_newsapi(params: NewsApiParam) -> Union[SearchSuccess, SearchError]:
    api_key = os.getenv("NEWSAPI_KEY")
    _params: dict[str, Any] = {k: v for k, v in params.items() if v is not None} | {
        "apiKey": api_key
    }
    response = requests.get(
        url="https://newsapi.org/v2/everything",
        params=_params,
        timeout=5,
    )
    if not response.ok:
        return {
            "status_code": response.status_code,
            "message": f'{response.json()["code"]}, {response.json()["message"]}',
        }
    return {
        "news": [
            {
                "source": news["source"],
                "author": news["author"],
                "title": news["title"],
                "description": news["description"],
                "content": news["content"],
                "url": news["url"],
                "urlToImage": news["urlToImage"],
                "publishedAt": news["publishedAt"],
            }
            for news in response.json()["articles"]
        ]
    }


def search_news(
    params: SearchParam,
    call_api: Callable[[SearchParam], Union[SearchSuccess, SearchError]],
) -> Union[SearchSuccess, SearchError]:
    return call_api(params)


# Usage:
search_result = search_news(
    generate_newapi_param("Taiwan", language="en"), request_newsapi
)
search_result2 = search_news(
    generate_newsdataapi_param("Taiwan", language="en"), request_newsdataapi
)


def send_newsapi_request(keyword: str) -> tuple[dict, int]:
    api_key = os.getenv("NEWSAPI_KEY")
    params = {"q": keyword, "sortBy": "publishedAt", "apiKey": api_key}
    response = requests.get(
        url="https://newsapi.org/v2/everything", params=params, timeout=5
    )
    if response.ok:
        return json.loads(response.content.decode("utf-8")), response.status_code
    return {"errorMessage": response.content.decode("utf-8")}, response.status_code


def parse_newsapi_response(newsapi_response: dict) -> dict:
    response = {"totalResults": newsapi_response["totalResults"], "results": []}
    for news in newsapi_response["articles"]:
        response["results"].append(
            {
                "semanticInfo": {},
                "source": news["source"],
                "author": news["author"],
                "title": news["title"],
                "description": news["description"],
                "url": news["url"],
                "urlToImage": news["urlToImage"],
                "publishedAt": news["publishedAt"],
                "content": news["content"],
            }
        )
    return response


def send_newsdataapi_request(keyword: str) -> tuple[dict, int]:
    api_key = os.getenv("NEWSDATAAPI_KEY")
    api = NewsDataApiClient(apikey=api_key)
    response = api.news_api(q=keyword, country="us")
    if response["status"] != "success":
        return response, 400
    return response, 200


def parse_newsdataapi_response(newsdataapi_response: dict) -> dict:
    response = {"totalResults": newsdataapi_response["totalResults"], "results": []}
    for news in newsdataapi_response["results"]:
        response["results"].append(
            {
                "semanticInfo": {},
                "source": news["source_id"],
                "author": news["creator"],
                "title": news["title"],
                "description": news["description"],
                "url": news["link"],
                "urlToImage": news["image_url"],
                "publishedAt": news["pubDate"],
                "content": news["content"],
            }
        )
    return response


def send_biasapi_request(article: str) -> tuple[dict, int]:
    return {}, 200


def send_toneapi_request(articles: list[str]) -> tuple[list, int]:
    return [
        {
            "label": random.choice(["POS", "NEU", "NEG"]),
            "score": random.uniform(0, 1),
        }
    ], 200


def parse_semantic_response(biasapi_response: dict, tone_response: dict) -> dict:
    tone = {}
    if tone_response["label"] == "POS":
        tone["class"] = "positive"
    elif tone_response["label"] == "NEU":
        tone["class"] = "neural"
    else:
        tone["class"] = "negative"
    tone["confidence"] = tone_response["score"]
    return {
        "bias": random.choice(["left", "leanLeft", "center", "leanRight", "right"]),
        "tone": tone,
    }


def filter_opposite_semantic(response: dict, current_semantic: dict) -> dict:
    filtered_response = {}
    filtered_response["results"] = list(
        filter(
            lambda news: news["semanticInfo"]["bias"] != current_semantic["bias"]
            and news["semanticInfo"]["tone"]["class"]
            != current_semantic["tone"]["class"],
            response["results"],
        )
    )
    filtered_response["totalResults"] = len(list(filtered_response["results"]))
    return filtered_response


app = Flask(__name__)


@app.route("/get-news-semantic", methods=["GET", "POST"])
def get_news_semantic() -> tuple[str, int]:
    try:
        request_content = json.loads(request.data.decode("utf-8"))
        if "selectedNews" not in request_content:
            raise ValueError
        news_content = request_content["selectedNews"]["content"]
        biasapi_response, status_code = send_biasapi_request(news_content)
        if status_code != 200:
            return f"bias API status_code {status_code}", 500
        toneapi_response, status_code = send_toneapi_request([news_content])
        if status_code != 200:
            return f"tone API status_code {status_code}", 500
        return (
            json.dumps(parse_semantic_response(biasapi_response, toneapi_response[0])),
            200,
        )
    except ValueError:
        return "Invalid json format", 400
    except RuntimeError:
        return "Internal error", 500


@app.route("/opposite-semantic-news", methods=["GET", "POST"])
def get_opposite_news() -> tuple[str, int]:

    request_content = json.loads(request.data.decode("utf-8"))
    if "keyword" not in request_content:
        raise ValueError
    if "selectedNews" not in request_content:
        raise ValueError
    keyword = request_content["keyword"]
    response, status_code = send_newsdataapi_request(keyword)
    if status_code != 200:
        return f"newsAPI status_code {status_code}", 500
    parsed_response = parse_newsdataapi_response(response)
    for news in parsed_response["results"]:
        biasapi_response, status_code = send_biasapi_request(news["content"])
        if status_code != 200:
            return f"bias API status_code {status_code}", 500
        toneapi_response, status_code = send_toneapi_request(news["content"])
        if status_code != 200:
            return f"tone API status_code {status_code}", 500
        news["semanticInfo"] = parse_semantic_response(
            biasapi_response, toneapi_response[0]
        )
    filtered_response = filter_opposite_semantic(
        parsed_response, request_content["selectedNews"]["semanticInfo"]
    )
    return json.dumps(filtered_response), 200


@app.route("/search", methods=["GET", "POST"])
def search() -> tuple[str, int]:
    try:
        request_content = json.loads(request.data.decode("utf-8"))
        if "keyword" not in request_content:
            raise ValueError
        keyword = request_content["keyword"]
        response, status_code = send_newsdataapi_request(keyword)
        if status_code != 200:
            return f"newsAPI status_code {status_code}", 500
        parsed_response = parse_newsdataapi_response(response)
        for news in parsed_response["results"]:
            biasapi_response, status_code = send_biasapi_request(news["content"])
            if status_code != 200:
                return f"bias API status_code {status_code}", 500
            toneapi_response, status_code = send_toneapi_request(news["content"])
            if status_code != 200:
                return f"tone API status_code {status_code}", 500
            news["semanticInfo"] = parse_semantic_response(
                biasapi_response, toneapi_response[0]
            )
        return json.dumps(parsed_response), 200
    except ValueError:
        return "Invalid json format", 400
    except RuntimeError:
        return "Internal error", 500
