import json
import os
import random
from typing import Literal, Optional, TypedDict

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
    source: str
    author: Optional[str]
    title: str
    description: str
    content: str
    url: str
    urlToImage: Optional[str]
    publishedAt: str


class NewsWithSentiment(TypedDict):
    source: str
    author: Optional[str]
    title: str
    description: str
    content: str
    url: str
    urlToImage: Optional[str]
    publishedAt: str
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
