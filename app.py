import http
import json
import os
from typing import Any, Callable, TypeVar, Union, cast

import requests
from dotenv import load_dotenv
from flask import Flask, request
from newsdataapi import NewsDataApiClient
from werkzeug.exceptions import BadRequestKeyError

from data_types import (
    BiasAnalysisError,
    BiasAnalysisSuccess,
    BiasOkResponse,
    ErrorResponse,
    GatewayTimeoutResponse,
    InternalErrorResponse,
    News,
    NewsApiParam,
    NewsDataApiParam,
    NewsWithSentiment,
    OppositeSentimentNewsOkResponse,
    SearchError,
    SearchOkResponse,
    SearchSuccess,
    SentimentAnalysisError,
    SentimentAnalysisSuccess,
    SentimentAndBiasError,
    SentimentAndBiasOkResponse,
    SentimentAndBiasSuccess,
    SentimentOkResponse,
)
from news_traveler_sentiment_analysis.sentiment_analysis import (
    sentiment_analysis_per_document,
)

load_dotenv()

NEWSDATAAPI_KEY = os.environ["NEWSDATAAPI_KEY"]
NEWSAPI_KEY = os.environ["NEWSAPI_KEY"]
BIASAPI_KEY = os.environ["BIASAPI_KEY"]


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
    collected_news: list[News] = []
    call_count = 0
    api = NewsDataApiClient(apikey=NEWSDATAAPI_KEY)
    while len(collected_news) < 10 and call_count < 5:
        response = api.news_api(**params)
        if response["status"] == "error":
            return {
                "status_code": http.HTTPStatus.BAD_REQUEST,
                "message": f'{response["results"]["code"]}, {response["results"]["message"]}',
            }
        collected_news.extend(
            [
                {
                    "source": news["source_id"],
                    "author": ",".join(news["creator"])
                    if news["creator"]
                    else news["source_id"],
                    "title": news["title"],
                    "description": news["description"],
                    "content": news["content"],
                    "url": news["link"],
                    "urlToImage": news["image_url"],
                    "publishedAt": news["pubDate"],
                }
                for news in response["results"]
                if news["source_id"]
                and news["title"]
                and news["description"]
                and news["content"]
                and news["link"]
            ]
        )
        if "nextPage" not in response:
            break
        params.update({"page": response["nextPage"]})
        call_count += 1
    return {"news": collected_news}


def request_newsapi(params: NewsApiParam) -> Union[SearchSuccess, SearchError]:
    _params: dict[str, Any] = {k: v for k, v in params.items() if v is not None} | {
        "apiKey": NEWSAPI_KEY
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
            if news["source"]
            and news["title"]
            and news["description"]
            and news["content"]
            and news["url"]
        ]
    }


def request_biasapi(article: str) -> Union[BiasAnalysisSuccess, BiasAnalysisError]:
    response = requests.post(
        "https://api.thebipartisanpress.com/api/endpoints/beta/robert",
        data={"API": BIASAPI_KEY, "Text": article},
        timeout=20,
    )
    if response.ok:
        return {"value": float(response.content.decode("utf-8")) / 42}
    else:
        return {
            "status_code": response.status_code,
            "message": response.content.decode("utf-8"),
        }


def request_biasapi_mock(article: str) -> Union[BiasAnalysisSuccess, BiasAnalysisError]:
    return {"value": (abs(hash(article)) % 100) / 50.0 - 1.0}


def request_sentimentapi(
    article: str,
) -> Union[SentimentAnalysisSuccess, SentimentAnalysisError]:
    result = sentiment_analysis_per_document(article)
    return {
        "value": {
            "kind": "positive"
            if result["label"] == "POS"
            else "neutral"
            if result["label"] == "NEU"
            else "negative",
            "confidence": result["score"],
        }
    }


def analyze_sentiment_and_bias(
    article: str,
    call_biasapi: Callable[[str], Union[BiasAnalysisSuccess, BiasAnalysisError]],
    call_sentimentapi: Callable[
        [str], Union[SentimentAnalysisSuccess, SentimentAnalysisError]
    ],
) -> Union[SentimentAndBiasSuccess, SentimentAndBiasError]:
    bias_result = call_biasapi(article)
    sentiment_result = call_sentimentapi(article)
    if "value" in bias_result and "value" in sentiment_result:
        bias_result = cast(BiasAnalysisSuccess, bias_result)
        sentiment_result = cast(SentimentAnalysisSuccess, sentiment_result)
        return {
            "bias": bias_result["value"],
            "sentiment": sentiment_result["value"],
        }
    return {
        "bias": None
        if "value" not in bias_result
        else cast(BiasAnalysisSuccess, bias_result)["value"],
        "sentiment": None
        if "value" not in sentiment_result
        else cast(SentimentAnalysisSuccess, sentiment_result)["value"],
        "status_code": http.HTTPStatus.BAD_REQUEST,
        "message": f'bias: {"" if "message" not in bias_result else cast(BiasAnalysisError, bias_result)["message"]}'
        f'sentiment: {"" if "message" not in sentiment_result else cast(SentimentAnalysisError, sentiment_result)["message"]}',
    }


def analyze_sentiment(
    article: str,
    call_sentimentapi: Callable[
        [str], Union[SentimentAnalysisSuccess, SentimentAnalysisError]
    ],
) -> Union[SentimentAnalysisSuccess, SentimentAnalysisError]:
    return call_sentimentapi(article)


def analyze_bias(
    article: str,
    call_biasapi: Callable[[str], Union[BiasAnalysisSuccess, BiasAnalysisError]],
) -> Union[BiasAnalysisSuccess, BiasAnalysisError]:
    return call_biasapi(article)


def search_news(
    params: SearchParam,
    call_api: Callable[[SearchParam], Union[SearchSuccess, SearchError]],
) -> Union[SearchSuccess, SearchError]:
    return call_api(params)


app = Flask(__name__)


@app.route("/sentiment", methods=["POST"])
def get_news_sentiment() -> tuple[
    Union[
        ErrorResponse,
        InternalErrorResponse,
        GatewayTimeoutResponse,
        SentimentOkResponse,
    ],
    int,
]:
    try:
        article = json.loads(request.data.decode("utf-8"))["content"]
    except json.JSONDecodeError as e:
        return {"message": f"json decode error: {e.msg}"}, http.HTTPStatus.BAD_REQUEST
    except UnicodeDecodeError as e:
        return {
            "message": f"string decode error: {e.reason}"
        }, http.HTTPStatus.BAD_REQUEST
    except KeyError as e:
        return {"message": "key not found: content"}, http.HTTPStatus.BAD_REQUEST
    analyze_result = analyze_sentiment(article, request_sentimentapi)
    if "status_code" not in analyze_result:
        analyze_result = cast(SentimentAnalysisSuccess, analyze_result)
        return {"sentiment": analyze_result["value"]}, http.HTTPStatus.OK
    analyze_result = cast(SentimentAnalysisError, analyze_result)
    return {
        "message": analyze_result["message"],
        "debug": "",
    }, http.HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/bias", methods=["POST"])
def get_news_bias() -> tuple[
    Union[
        ErrorResponse,
        InternalErrorResponse,
        GatewayTimeoutResponse,
        BiasOkResponse,
    ],
    int,
]:
    try:
        article = json.loads(request.data.decode("utf-8"))["content"]
    except json.JSONDecodeError as e:
        return {"message": f"json decode error: {e.msg}"}, http.HTTPStatus.BAD_REQUEST
    except UnicodeDecodeError as e:
        return {
            "message": f"string decode error: {e.reason}"
        }, http.HTTPStatus.BAD_REQUEST
    except KeyError as e:
        return {"message": "key not found: content"}, http.HTTPStatus.BAD_REQUEST
    analyze_result = analyze_bias(article, request_biasapi)
    if "status_code" not in analyze_result:
        analyze_result = cast(BiasAnalysisSuccess, analyze_result)
        return {"bias": analyze_result["value"]}, http.HTTPStatus.OK
    analyze_result = cast(BiasAnalysisError, analyze_result)
    return {
        "message": analyze_result["message"],
        "debug": "",
    }, http.HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/sentiment-and-bias", methods=["POST"])
def get_news_sentiment_and_bias() -> tuple[
    Union[
        ErrorResponse,
        InternalErrorResponse,
        GatewayTimeoutResponse,
        SentimentAndBiasOkResponse,
    ],
    int,
]:
    try:
        article = json.loads(request.data.decode("utf-8"))["content"]
    except json.JSONDecodeError as e:
        return {"message": f"json decode error: {e.msg}"}, http.HTTPStatus.BAD_REQUEST
    except UnicodeDecodeError as e:
        return {
            "message": f"string decode error: {e.reason}"
        }, http.HTTPStatus.BAD_REQUEST
    except KeyError as e:
        return {"message": "key not found: content"}, http.HTTPStatus.BAD_REQUEST
    analyze_result = analyze_sentiment_and_bias(
        article, request_biasapi_mock, request_sentimentapi
    )
    if "status_code" not in analyze_result:
        analyze_result = cast(SentimentAndBiasSuccess, analyze_result)
        return {
            "sentiment": analyze_result["sentiment"],
            "bias": analyze_result["bias"],
        }, http.HTTPStatus.OK
    analyze_result = cast(SentimentAndBiasError, analyze_result)
    return {
        "message": "biasapi error "
        if analyze_result["bias"] is None
        else "" + "sentimentapi error "
        if analyze_result["sentiment"] is None
        else "" + f": {analyze_result['message']}",
        "debug": "",
    }, http.HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/opposite-sentiment-news", methods=["POST"])
def get_opposite_news() -> tuple[
    Union[
        ErrorResponse,
        InternalErrorResponse,
        GatewayTimeoutResponse,
        OppositeSentimentNewsOkResponse,
    ],
    int,
]:
    try:
        request_content = json.loads(request.data.decode("utf-8"))
    except json.JSONDecodeError as e:
        return {"message": f"json decode error: {e.msg}"}, http.HTTPStatus.BAD_REQUEST
    except UnicodeDecodeError as e:
        return {
            "message": f"string decode error: {e.reason}"
        }, http.HTTPStatus.BAD_REQUEST
    try:
        article = request_content["content"]
    except KeyError as e:
        return {"message": "key not found: content"}, http.HTTPStatus.BAD_REQUEST
    try:
        keyword = request_content["keyword"]
    except KeyError as e:
        return {"message": "key not found: keyword"}, http.HTTPStatus.BAD_REQUEST
    search_result = search_news(
        generate_newsdataapi_param(keyword, language="en"), request_newsdataapi
    )
    if "news" not in search_result:
        search_result = cast(SearchError, search_result)
        return {
            "message": f"newsdataapi status_code {search_result['status_code']} "
            f"with message {search_result['message']}",
            "debug": "",
        }, http.HTTPStatus.INTERNAL_SERVER_ERROR
    search_result = cast(SearchSuccess, search_result)
    analyze_result = analyze_sentiment(article, request_sentimentapi)
    if "status_code" in analyze_result:
        analyze_result = cast(SentimentAnalysisError, analyze_result)
        return {
            "message": analyze_result["message"]
        }, http.HTTPStatus.INTERNAL_SERVER_ERROR
    current_sentiment = cast(SentimentAnalysisSuccess, analyze_result)["value"]
    filtered_results: list[NewsWithSentiment] = []
    for news in search_result["news"]:
        if len(filtered_results) == 3:
            break
        analyze_result = analyze_sentiment(news["content"], request_sentimentapi)
        if "status_code" in analyze_result:
            analyze_result = cast(SentimentAnalysisError, analyze_result)
            return {
                "message": analyze_result["message"]
            }, http.HTTPStatus.INTERNAL_SERVER_ERROR
        analyze_result = cast(SentimentAnalysisSuccess, analyze_result)
        filtered_news = cast(
            NewsWithSentiment, cast(dict, news) | {"sentiment": analyze_result["value"]}
        )
        if analyze_result["value"]["kind"] != current_sentiment["kind"]:
            filtered_results.append(filtered_news)
    return {
        "results": filtered_results,
        "count": len(filtered_results),
    }, http.HTTPStatus.OK


@app.route("/search", methods=["GET"])
def search() -> tuple[
    Union[
        ErrorResponse, InternalErrorResponse, GatewayTimeoutResponse, SearchOkResponse
    ],
    int,
]:
    try:
        keyword = request.args["query"]
    except BadRequestKeyError:
        return {"message": "key not found: query"}, http.HTTPStatus.BAD_REQUEST
    search_result = search_news(
        generate_newsdataapi_param(keyword, language="en"), request_newsdataapi
    )
    if "news" in search_result:
        search_result = cast(SearchSuccess, search_result)
        return {
            "results": search_result["news"],
            "count": len(search_result["news"]),
        }, http.HTTPStatus.OK
    search_result = cast(SearchError, search_result)
    return {
        "message": f"newsdataapi status_code {search_result['status_code']} "
        f"with message {search_result['message']}",
        "debug": "",
    }, http.HTTPStatus.INTERNAL_SERVER_ERROR
