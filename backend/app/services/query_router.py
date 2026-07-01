from enum import StrEnum
import re


class QueryType(StrEnum):
    LOCAL_FACT_SEARCH = "local_fact_search"
    DOCUMENT_OVERVIEW = "document_overview"
    INDEX_LOOKUP = "index_lookup"
    AGGREGATE_ANALYSIS = "aggregate_analysis"
    GRAPH_ANALYSIS = "graph_analysis"


AGGREGATE_PATTERNS = [
    r"top\s*\d+",
    r"가장 많이",
    r"빈도",
    r"통계",
    r"frequency",
    r"most referenced",
    r"모든 문서",
    r"전체 문서",
    r"전 문서",
    r"찾은 모든 페이지",
    r"문서별로",
    r"표로 정리",
    r"전부 찾아",
    r"전체에서 찾아",
]
GRAPH_PATTERNS = [
    r"그래프",
    r"도표",
    r"그림",
    r"figure",
    r"chart",
    r"graph",
    r"plot",
    r"x축",
    r"y축",
    r"x-axis",
    r"y-axis",
    r"추세",
    r"그래프의 의미",
]
OVERVIEW_PATTERNS = [r"목적", r"개요", r"구조", r"목차", r"요약", r"contents", r"table of contents", r"purpose", r"overview"]
INDEX_PATTERNS = [r"author index", r"subject index", r"master author", r"master subject", r"symbols? standard", r"conversion", r"unit", r"참조되는 페이지", r"등장", r"의미", r"symbol"]


def classify_query(question: str) -> QueryType:
    text = question.lower()
    if any(re.search(pattern, text) for pattern in GRAPH_PATTERNS):
        return QueryType.GRAPH_ANALYSIS
    if any(re.search(pattern, text) for pattern in AGGREGATE_PATTERNS):
        return QueryType.AGGREGATE_ANALYSIS
    if any(re.search(pattern, text) for pattern in OVERVIEW_PATTERNS):
        return QueryType.DOCUMENT_OVERVIEW
    if any(re.search(pattern, text) for pattern in INDEX_PATTERNS):
        return QueryType.INDEX_LOOKUP
    # Very short symbol/name lookups should favor exact index retrieval.
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", question)
    if len(tokens) <= 3 and any(len(token) <= 3 for token in tokens):
        return QueryType.INDEX_LOOKUP
    return QueryType.LOCAL_FACT_SEARCH
