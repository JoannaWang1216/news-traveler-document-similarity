import http
import json
import os
import random
import re
from typing import Any, Callable, Final, TypeVar, Union, cast

import requests
from dotenv import load_dotenv
from flask import Flask, request
from flask_caching import Cache
from newsdataapi import NewsDataApiClient, newsdataapi_exception
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
    SearchError,
    SearchOkResponse,
    SearchSuccess,
    SearchWithFilterOkResponse,
    SearchWithSentimentSuccess,
    SentimentAnalysisError,
    SentimentAnalysisSuccess,
    SentimentAndBiasError,
    SentimentAndBiasOkResponse,
    SentimentAndBiasSuccess,
    SentimentLabel,
    SentimentOkResponse,
    SimilarityAnalysisError,
    SimilarityAnalysisSuccess,
)
from news_traveler_document_similarity.tfidf_similarity import process_tfidf_similarity
from news_traveler_sentiment_analysis.sentiment_analysis import (
    sentiment_analysis_per_document,
)

CACHE_DEFAULT_TIMEOUT = 60 * 60 * 24  # 1 day

config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": CACHE_DEFAULT_TIMEOUT,
}

app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)

load_dotenv()

newsdataapi_keys_re = re.compile(r"NEWSDATAAPI_KEY_(\d+)")
NEWSDATAAPI_KEY = [
    value for key, value in os.environ.items() if newsdataapi_keys_re.match(key)
]
print("Available keys:", len(NEWSDATAAPI_KEY))
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


@cache.memoize(CACHE_DEFAULT_TIMEOUT)
def _request_newsdataapi(params: NewsDataApiParam) -> Any:
    api = NewsDataApiClient(apikey=random.choice(NEWSDATAAPI_KEY))
    return api.news_api(**params)


@cache.memoize(CACHE_DEFAULT_TIMEOUT)
def request_newsdataapi(
    params: NewsDataApiParam, count: int, exact_count: bool
) -> Union[SearchSuccess, SearchError]:
    MAX_CALL_COUNT: Final = 5
    collected_news: list[News] = []
    call_count = 0
    while len(collected_news) < count and call_count < MAX_CALL_COUNT:
        try:
            response = _request_newsdataapi(params)
        except newsdataapi_exception.NewsdataException as e:
            return {
                "status_code": http.HTTPStatus.INTERNAL_SERVER_ERROR,
                "message": json.loads(str(e).replace("'", '"'))["results"]["message"],
            }
        if response["status"] == "error":
            return {
                "status_code": http.HTTPStatus.BAD_REQUEST,
                "message": f'{response["results"]["code"]}, {response["results"]["message"]}',
            }
        collected_news.extend(
            [
                cast(
                    News,
                    {
                        "source": news["source_id"],
                        "author": ",".join(news["creator"])
                        if news["creator"]
                        else news["source_id"],
                        "title": news["title"],
                        "content": news["content"]
                        if news["description"] is None
                        else news["description"]
                        if news["content"] is None
                        else news["content"]
                        if len(news["content"]) > len(news["description"])
                        else news["description"],
                        "url": news["link"],
                        "urlToImage": news["image_url"],
                        "publishedAt": news["pubDate"],
                    },
                )
                for news in response["results"]
                if news["title"]
                and (news["description"] or news["content"])
                and news["link"]
            ]
        )
        if response["nextPage"] is None:
            break
        params.update({"page": response["nextPage"]})
        call_count += 1
    if exact_count:
        collected_news = collected_news[:count]
    if len(collected_news) >= count:
        return {"news": collected_news, "nextOffset": response["nextPage"]}
    return {"news": collected_news, "nextOffset": None}


@cache.memoize(CACHE_DEFAULT_TIMEOUT)
def request_newsapi(
    params: NewsApiParam, count: int, exact_count: bool  # type: ignore
) -> Union[SearchSuccess, SearchError]:
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
                "content": news["content"]
                if news["description"] is None
                else news["description"]
                if news["content"] is None
                else news["content"]
                if len(news["content"]) > len(news["description"])
                else news["description"],
                "url": news["url"],
                "urlToImage": news["urlToImage"],
                "publishedAt": news["publishedAt"],
            }
            for news in response.json()["articles"]
            if news["title"]
            and (news["description"] or news["content"])
            and news["url"]
        ],
        "nextOffset": None,
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


def request_similarityapi(
    base_article: str, article: str, threshold: float
) -> Union[SimilarityAnalysisSuccess, SimilarityAnalysisError]:
    return {"is_similar": process_tfidf_similarity(base_article, article, threshold)}


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
    count: int,
    call_api: Callable[[SearchParam, int, bool], Union[SearchSuccess, SearchError]],
) -> Union[SearchSuccess, SearchError]:
    return call_api(params, count, True)


def search_news_with_filter(
    params: SearchParam,
    count: int,
    call_newsapi: Callable[[SearchParam, int, bool], Union[SearchSuccess, SearchError]],
    call_sentimentapi: Callable[
        [str], Union[SentimentAnalysisSuccess, SentimentAnalysisError]
    ],
    call_similarityapi: Callable[
        [str, str, float], Union[SimilarityAnalysisSuccess, SimilarityAnalysisError]
    ],
    sentiment_labels: list[SentimentLabel],
    similarity_threshold: float,
    base_article: str,
) -> Union[SearchWithSentimentSuccess, SearchError]:
    collected_news: list[NewsWithSentiment] = []
    while len(collected_news) < count:
        result = call_newsapi(params, 10, False)
        if "news" in result:
            result = cast(SearchSuccess, result)
            collected_news.extend(
                [
                    cast(
                        NewsWithSentiment,
                        cast(dict, news)
                        | {
                            "sentiment": cast(
                                dict, call_sentimentapi(news["content"])["value"]
                            )
                        },
                    )
                    for news in result["news"]
                    if cast(
                        SimilarityAnalysisSuccess,
                        call_similarityapi(
                            base_article, news["content"], similarity_threshold
                        ),
                    )["is_similar"]
                    and cast(
                        SentimentAnalysisSuccess, call_sentimentapi(news["content"])
                    )["value"]["kind"]
                    in sentiment_labels
                    and news["content"] != base_article
                ]
            )
            if result["nextOffset"] is None:
                break
            params["page"] = result["nextOffset"]
        else:
            return cast(SearchError, result)
    return {
        "news": collected_news[:count],
        "nextOffset": cast(SearchSuccess, result)["nextOffset"],
    }


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
def search_with_filters() -> tuple[
    Union[
        ErrorResponse,
        InternalErrorResponse,
        GatewayTimeoutResponse,
        SearchWithFilterOkResponse,
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
    try:
        count = request_content["count"]
    except KeyError as e:
        return {"message": "key not found: count"}, http.HTTPStatus.BAD_REQUEST
    try:
        similarity_threshold = request_content["similarityThreshold"]
    except KeyError as e:
        return {
            "message": "key not found: similarityThreshold"
        }, http.HTTPStatus.BAD_REQUEST
    try:
        if not all(
            sentiment in ["positive", "negative", "neutral"]
            for sentiment in request_content["sentimentFilter"]
        ):
            raise ValueError(
                "sentimentFilter must be a list of positive, negative, neutral"
            )
        sentiment_filter: list[SentimentLabel] = cast(
            list[SentimentLabel],
            request_content["sentimentFilter"]
            if request_content["sentimentFilter"]
            else ["positive", "negative", "neutral"],
        )
    except KeyError as e:
        return {
            "message": "key not found: sentimentFilter"
        }, http.HTTPStatus.BAD_REQUEST
    except ValueError as e:
        return {"message": str(e)}, http.HTTPStatus.BAD_REQUEST
    search_result = search_news_with_filter(
        generate_newsdataapi_param(keyword, language="en"),
        count,
        request_newsdataapi,
        request_sentimentapi,
        request_similarityapi,
        sentiment_filter,
        similarity_threshold,
        article,
    )
    if "news" not in search_result:
        search_result = cast(SearchError, search_result)
        return {
            "message": f"newsdataapi status_code {search_result['status_code']} "
            f"with message {search_result['message']}",
            "debug": "",
        }, http.HTTPStatus.INTERNAL_SERVER_ERROR
    search_result = cast(SearchWithSentimentSuccess, search_result)

    return {
        "results": search_result["news"],
        "count": len(search_result["news"]),
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
        count = int(request.args["count"])
        if count < 1:
            raise ValueError("count must be > 1")
        offset = int(request.args["offset"]) if "offset" in request.args else None
        if offset is not None and offset < 0:
            raise ValueError("offset must be >= 0")
    except BadRequestKeyError as e:
        e.show_exception = True
        return {"message": "key not found: " + e.args[0]}, http.HTTPStatus.BAD_REQUEST
    except ValueError as e:
        return {"message": e.args[0]}, http.HTTPStatus.BAD_REQUEST
    search_result = search_news(
        generate_newsdataapi_param(keyword, language="en", page=offset),
        count,
        request_newsdataapi,
    )
    if "news" in search_result:
        search_result = cast(SearchSuccess, search_result)
        return {
            "results": search_result["news"],
            "count": len(search_result["news"]),
            "nextOffset": search_result["nextOffset"],
        }, http.HTTPStatus.OK
    search_result = cast(SearchError, search_result)
    return {
        "message": f"newsdataapi status_code {search_result['status_code']} "
        f"with message {search_result['message']}",
        "debug": "",
    }, http.HTTPStatus.INTERNAL_SERVER_ERROR
