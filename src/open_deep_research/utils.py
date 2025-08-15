import os
import re
import asyncio
import json
import datetime
import requests
import random
import concurrent.futures
import hashlib
import pickle
from pathlib import Path
import aiohttp
from aiohttp import ClientTimeout
import httpx
import time
from typing import List, Optional, Dict, Any, Union, Literal, Annotated, cast, overload
from enum import Enum
from urllib.parse import unquote
from collections import defaultdict, OrderedDict
import itertools
import logging

from exa_py import Exa
from linkup import LinkupClient
from tavily import AsyncTavilyClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient as AsyncAzureAISearchClient
from azure.search.documents.models import VectorQuery
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from open_deep_research.knowledge_base import KnowledgeBaseManager
from open_deep_research.configuration import Configuration
from open_deep_research.state import Section
from open_deep_research.prompts import SUMMARIZATION_PROMPT
from open_deep_research.prompts import (
    general_overview,
    regional_overview,
    industry_overview,
    regional_detail,
    industry_detail,
)
import tiktoken


def route_prompt(chapter_name, section_name: str) -> str:
    """Route the prompt to the appropriate section based on chapter and section names."""
    sys_prompts = {
        "总体概况": general_overview,
        "地区概况": regional_overview,
        "行业概况": industry_overview,
    }
    if section_name in sys_prompts:
        logging.info(f"Routing prompt to system prompt for section: {section_name}")
        return sys_prompts[section_name]

    chapter_prompts = {"地区产品质量状况": regional_detail, "行业产品质量状况": industry_detail}
    if chapter_name in chapter_prompts:
        logging.info(f"Routing prompt to chapter prompt for chapter: {chapter_name}")
        return chapter_prompts[chapter_name]
    return ""


def get_model(configurable: Configuration, mode: str) -> BaseChatModel:
    """Initialize and return the chat model based on the configuration."""
    mode_map = {
        "planner": ("planner_model", "planner_provider", "planner_model_kwargs"),
        "writer": ("writer_model", "writer_provider", "writer_model_kwargs"),
        "summarizer": (
            "summarization_model",
            "summarization_model_provider",
            "summarization_model_kwargs",
        ),
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid mode: {mode}. Expected one of {list(mode_map.keys())}.")
    model_attr, provider_attr, kwargs_attr = mode_map[mode]
    model_name = get_config_value(getattr(configurable, model_attr))
    model_provider = get_config_value(getattr(configurable, provider_attr))
    model_kwargs = get_config_value(getattr(configurable, kwargs_attr) or {})
    return init_chat_model(model=model_name, model_provider=model_provider, **model_kwargs)


async def reduce_source_str(input_string: str, max_tokens: int = 60000) -> str:
    """Helper function to reduce the length of a string to fit within a token limit.

    Args:
        input_string (str): input string to be reduced.
        max_tokens (int, optional): maximum token number. Defaults to 60000.

    Returns:
        str: reduced string that fits within the token limit.
    """

    async def get_encoding_async():
        return await asyncio.to_thread(tiktoken.get_encoding, "cl100k_base")

    encoder = await get_encoding_async()  # DeepSeek/OpenAI 使用类似的分词器
    token_count = len(encoder.encode(input_string))
    logging.info(f"输入字符串的 Token 数: {token_count}")
    if token_count > max_tokens:
        len_str = int(max_tokens / token_count * len(input_string))
        input_string = input_string[:len_str]
        logging.warning(f"超过最大限制 {max_tokens}，已缩减字符串长度到 {len_str} 字符")
    return input_string


@overload
def get_config_value(value: str) -> str: ...
@overload
def get_config_value(value: Dict[str, Any]) -> Dict[str, Any]: ...
@overload
def get_config_value(value: Enum) -> str: ...


def get_config_value(value: Union[str, Dict[str, Any], Enum]) -> Union[str, Dict[str, Any]]:
    """
    Helper function to handle string, dict, and enum cases of configuration values
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value


def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": ["max_results", "topic"],
        "perplexity": [],  # Perplexity accepts no additional parameters
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "linkup": ["depth"],
        "googlesearch": ["max_results"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}


def deduplicate_and_format_sources(
    search_response,
    max_tokens_per_source=5000,
    include_raw_content=True,
    deduplication_strategy: Literal["keep_first", "keep_last"] = "keep_first",
):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.

    Args:
        search_responses: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool
        deduplication_strategy: Whether to keep the first or last search result for each unique URL
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response["results"])

    # Deduplicate by URL
    if deduplication_strategy == "keep_first":
        unique_sources = {}
        for source in sources_list:
            if source["url"] not in unique_sources:
                unique_sources[source["url"]] = source
    elif deduplication_strategy == "keep_last":
        unique_sources = {source["url"]: source for source in sources_list}
    else:
        raise ValueError(f"Invalid deduplication strategy: {deduplication_strategy}")

    # Format output
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*80}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-'*80}\n"  # Subsection separator
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        formatted_text += f"{'='*80}\n\n"  # End section separator

    return formatted_text.strip()


def format_sections(sections: list[Section], final: bool = False) -> str:
    """Format a list of sections into a string"""
    chapter_no = set()
    i = 0
    k = 0
    sections_str = ""
    for j, section in enumerate(sections):
        if section.chapter_name not in chapter_no:
            chapter_no.add(section.chapter_name)
            i += 1
            k = j
            if not final:
                sections_str += (
                    f"\n# 第{i}章 {section.chapter_name}\n"
                    + f"## 第{j + 1 - k}节 {section.name}\n"
                    + f"### 概要: {section.description}\n"
                    + f"### 内容：{section.content if section.content else '[未完成]'}\n"
                )
            else:
                sections_str += (
                    f"\n# 第{i}章 {section.chapter_name}\n"
                    + f"## 第{j + 1 - k}节 {section.name}\n"
                    + f"{section.content if section.content else '[未完成]'}\n"
                )
        else:
            if not final:
                sections_str += (
                    f"\n## 第{j + 1 - k}节 {section.name}\n"
                    + f"### 概要: {section.description}\n"
                    + f"### 内容：{section.content if section.content else '[未完成]'}\n"
                )
            else:
                sections_str += (
                    f"\n## 第{j + 1 - k}节 {section.name}\n"
                    + f"{section.content if section.content else '[未完成]'}\n"
                )
    return sections_str


def process_references_from_sections(sections: List[Section]) -> tuple[str, List[Section]]:
    """
    处理多个Section对象中的引用资料，提取所有引用，按照正文中的引用顺序重新排序。

    Args:
        sections: Section对象列表，每个对象的content属性包含正文和引用资料

    Returns:
        tuple: (统一的资料来源字符串, 处理后的Section对象列表)
    """
    # 用于存储所有引用及其在文档中的位置
    all_citations = []
    # 用于存储每个节的引用资料
    section_references = {}
    # 记录当前处理的文本位置
    current_position = 0

    # 第一遍处理：提取每个节的引用资料和正文中的引用
    for section_idx, section in enumerate(sections):
        content = section.content

        # 查找该节中的资料来源部分
        source_match = re.search(r"### 资料来源\n((?:[^\n#]|(?:\n(?!#{1,3})))+)", content, re.DOTALL)
        if source_match:
            source_content = source_match.group(1)

            # 提取该节中的所有引用条目
            references = re.findall(r"\[(.*?)\](.*?)(?=\n\[|\n*$)", source_content, re.DOTALL)

            # 保存该节的引用信息
            section_refs = {}
            for ref_id, ref_content in references:
                section_refs[ref_id] = ref_content.strip()

            section_references[section_idx] = section_refs

            # 计算正文部分（去掉资料来源部分）
            section_content_without_source = content[: source_match.start()]

            # 查找正文中的所有引用标记及其位置
            for citation_match in re.finditer(r"\[(\d+)\]", section_content_without_source):
                citation_id = citation_match.group(1)
                if citation_id in section_refs:
                    all_citations.append(
                        {
                            "global_position": current_position + citation_match.start(),
                            "old_id": citation_id,
                            "section_idx": section_idx,
                            "content": section_refs[citation_id],
                        }
                    )

            # 更新Section对象，移除资料来源部分
            sections[section_idx].content = section_content_without_source

        # 更新当前位置
        current_position += len(content)

    # 按照在全局正文中的位置排序引用
    all_citations.sort(key=lambda x: x["global_position"])

    # 按正文中引用的顺序创建有序字典，合并相同内容
    ordered_refs = OrderedDict()
    old_to_new_id = {}  # {(section_idx, old_id): new_id}
    new_id = 1

    for citation in all_citations:
        section_idx = citation["section_idx"]
        old_id = citation["old_id"]
        content = citation["content"]

        # 检查是否已经有相同内容的引用
        content_already_referenced = False
        for existing_new_id, existing_content in ordered_refs.items():
            if existing_content == content:
                old_to_new_id[(section_idx, old_id)] = existing_new_id
                content_already_referenced = True
                break

        # 如果内容是新的，分配新的引用ID
        if not content_already_referenced:
            new_id_str = str(new_id)
            old_to_new_id[(section_idx, old_id)] = new_id_str
            ordered_refs[new_id_str] = content
            new_id += 1

    # 第二遍处理：替换每个节中的引用编号
    for section_idx, section in enumerate(sections):
        content = section.content

        # 替换该节正文中的引用
        updated_content = content

        # 查找该节中的所有引用
        if section_idx in section_references:
            section_citations = []
            for citation_match in re.finditer(r"\[(\d+)\]", content):
                old_id = citation_match.group(1)
                if old_id in section_references[section_idx]:
                    start_pos = citation_match.start()
                    end_pos = citation_match.end()
                    section_citations.append((start_pos, end_pos, old_id))

            # 从后向前替换，避免位置偏移问题
            for start_pos, end_pos, old_id in sorted(section_citations, reverse=True):
                if (section_idx, old_id) in old_to_new_id:
                    new_id = old_to_new_id[(section_idx, old_id)]
                    updated_content = updated_content[:start_pos] + f"[{new_id}]" + updated_content[end_pos:]

        # 更新Section对象
        sections[section_idx].content = updated_content

    # 生成统一的资料来源部分
    references_section = "\n# 资料来源\n"
    for ref_id, ref_content in ordered_refs.items():
        references_section += f"[{ref_id}] {ref_content}\n\n"

    return references_section, sections


@traceable
async def tavily_search_async(
    search_queries,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> List[Dict]:
    """
    Performs concurrent web searches with the Tavily API

    Args:
        search_queries (List[str]): List of search queries to process
        max_results (int): Maximum number of results to return
        topic (Literal["general", "news", "finance"]): Topic to filter results by
        include_raw_content (bool): Whether to include raw content in the results

    Returns:
        List[dict]: List of search responses from Tavily API:
            {
                'query': str,
                'follow_up_questions': None,
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the webpage
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full page content if available
                    },
                    ...
                ]
            }
    """
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []

    async def safe_search(query):
        try:
            logging.debug(f"Starting Tavily search for query: {query}")
            result = await tavily_async_client.search(
                query,
                max_results=max_results,
                include_raw_content=True,
                include_images=False,
                topic=topic,
                timeout=300,
            )
            logging.debug(f"Completed Tavily search for query: {query}")
            return result
        except Exception as e:
            logging.error(f"Tavily search failed for query '{query}': {str(e)}", exc_info=True)
            # Return a placeholder result to keep index alignment
            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e),
            }

    for query in search_queries:
        search_tasks.append(safe_search(query))

    logging.debug(
        f"Starting {len(search_tasks)} concurrent Tavily searches with max_results={max_results}, topic={topic}, include_raw_content={include_raw_content}"
    )
    search_docs = await asyncio.gather(*search_tasks)
    logging.debug(f"Completed {len(search_docs)} Tavily searches")
    return search_docs


@traceable
async def azureaisearch_search_async(
    search_queries: List[str],
    max_results: int = 5,
    topic: str = "general",
    include_raw_content: bool = True,
) -> List[Dict]:
    """
    Performs concurrent web searches using the Azure AI Search API.

    Args:
        search_queries (List[str]): list of search queries to process
        max_results (int): maximum number of results to return for each query
        topic (str): semantic topic filter for the search.
        include_raw_content (bool)

    Returns:
        List[dict]: list of search responses from Azure AI Search API, one per query.
    """
    # configure and create the Azure Search client
    # ensure all environment variables are set
    if not all(
        var in os.environ
        for var in [
            "AZURE_AI_SEARCH_ENDPOINT",
            "AZURE_AI_SEARCH_INDEX_NAME",
            "AZURE_AI_SEARCH_API_KEY",
        ]
    ):
        raise ValueError(
            "Missing required environment variables for Azure Search API which are: AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_INDEX_NAME, AZURE_AI_SEARCH_API_KEY"
        )
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT", "")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "")
    credential = AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY", ""))

    reranker_key = "@search.reranker_score"

    async with AsyncAzureAISearchClient(endpoint, index_name, credential) as client:

        async def do_search(query: str) -> Dict:

            # search query
            paged = await client.search(
                search_text=query,
                vector_queries=[VectorQuery(fields="vector", kind="text", text=query, exhaustive=True)],
                semantic_configuration_name="fraunhofer-rag-semantic-config",
                query_type="semantic",
                select=["url", "title", "chunk", "creationTime", "lastModifiedTime"],
                top=max_results,
            )
            # async iterator to get all results
            items = [doc async for doc in paged]
            # Umwandlung in einfaches Dict-Format
            results = [
                {
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "content": doc.get("chunk"),
                    "score": doc.get(reranker_key),
                    "raw_content": doc.get("chunk") if include_raw_content else None,
                }
                for doc in items
            ]
            return {"query": query, "results": results}

        # parallelize the search queries
        tasks = [do_search(q) for q in search_queries]
        return await asyncio.gather(*tasks)


@traceable
def perplexity_search(search_queries):
    """Search the web using the Perplexity API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process

    Returns:
        List[dict]: List of search responses from Perplexity API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
    }

    search_docs = []
    for query in search_queries:

        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Search the web and provide factual information with sources.",
                },
                {"role": "user", "content": query},
            ],
        }

        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse the response
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", ["https://perplexity.ai"])

        # Create results list for this query
        results = []

        # First citation gets the full content
        results.append(
            {
                "title": f"Perplexity Search, Source 1",
                "url": citations[0],
                "content": content,
                "raw_content": content,
                "score": 1.0,  # Adding score to match Tavily format
            }
        )

        # Add additional citations without duplicating content
        for i, citation in enumerate(citations[1:], start=2):
            results.append(
                {
                    "title": f"Perplexity Search, Source {i}",
                    "url": citation,
                    "content": "See primary source for full content",
                    "raw_content": None,
                    "score": 0.5,  # Lower score for secondary sources
                }
            )

        # Format response to match Tavily structure
        search_docs.append(
            {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": results,
            }
        )

    return search_docs


@traceable
async def exa_search(
    search_queries,
    max_characters: Optional[int] = None,
    num_results=5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    subpages: Optional[int] = None,
):
    """Search the web using the Exa API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        max_characters (int, optional): Maximum number of characters to retrieve for each result's raw content.
                                       If None, the text parameter will be set to True instead of an object.
        num_results (int): Number of search results per query. Defaults to 5.
        include_domains (List[str], optional): List of domains to include in search results.
            When specified, only results from these domains will be returned.
        exclude_domains (List[str], optional): List of domains to exclude from search results.
            Cannot be used together with include_domains.
        subpages (int, optional): Number of subpages to retrieve per result. If None, subpages are not retrieved.

    Returns:
        List[dict]: List of search responses from Exa API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """
    # Check that include_domains and exclude_domains are not both specified
    if include_domains and exclude_domains:
        raise ValueError("Cannot specify both include_domains and exclude_domains")

    # Initialize Exa client (API key should be configured in your .env file)
    exa = Exa(api_key=f"{os.getenv('EXA_API_KEY')}")

    # Define the function to process a single query
    async def process_query(query):
        # Use run_in_executor to make the synchronous exa call in a non-blocking way
        loop = asyncio.get_event_loop()

        # Define the function for the executor with all parameters
        def exa_search_fn():
            # Build parameters dictionary
            kwargs = {
                # Set text to True if max_characters is None, otherwise use an object with max_characters
                "text": True if max_characters is None else {"max_characters": max_characters},
                "summary": True,  # This is an amazing feature by EXA. It provides an AI generated summary of the content based on the query
                "num_results": num_results,
            }

            # Add optional parameters only if they are provided
            if subpages is not None:
                kwargs["subpages"] = subpages

            if include_domains:
                kwargs["include_domains"] = include_domains
            elif exclude_domains:
                kwargs["exclude_domains"] = exclude_domains

            return exa.search_and_contents(query, **kwargs)

        response = await loop.run_in_executor(None, exa_search_fn)

        # Format the response to match the expected output structure
        formatted_results = []
        seen_urls = set()  # Track URLs to avoid duplicates

        # Helper function to safely get value regardless of if item is dict or object
        def get_value(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default) if hasattr(item, key) else default

        # Access the results from the SearchResponse object
        results_list = get_value(response, "results", [])

        # First process all main results
        for result in results_list:
            # Get the score with a default of 0.0 if it's None or not present
            score = get_value(result, "score", 0.0)

            # Combine summary and text for content if both are available
            text_content = get_value(result, "text", "")
            summary_content = get_value(result, "summary", "")

            content = text_content
            if summary_content:
                if content:
                    content = f"{summary_content}\n\n{content}"
                else:
                    content = summary_content

            title = get_value(result, "title", "")
            url = get_value(result, "url", "")

            # Skip if we've seen this URL before (removes duplicate entries)
            if url in seen_urls:
                continue

            seen_urls.add(url)

            # Main result entry
            result_entry = {
                "title": title,
                "url": url,
                "content": content,
                "score": score,
                "raw_content": text_content,
            }

            # Add the main result to the formatted results
            formatted_results.append(result_entry)

        # Now process subpages only if the subpages parameter was provided
        if subpages is not None:
            for result in results_list:
                subpages_list = get_value(result, "subpages", [])
                for subpage in subpages_list:
                    # Get subpage score
                    subpage_score = get_value(subpage, "score", 0.0)

                    # Combine summary and text for subpage content
                    subpage_text = get_value(subpage, "text", "")
                    subpage_summary = get_value(subpage, "summary", "")

                    subpage_content = subpage_text
                    if subpage_summary:
                        if subpage_content:
                            subpage_content = f"{subpage_summary}\n\n{subpage_content}"
                        else:
                            subpage_content = subpage_summary

                    subpage_url = get_value(subpage, "url", "")

                    # Skip if we've seen this URL before
                    if subpage_url in seen_urls:
                        continue

                    seen_urls.add(subpage_url)

                    formatted_results.append(
                        {
                            "title": get_value(subpage, "title", ""),
                            "url": subpage_url,
                            "content": subpage_content,
                            "score": subpage_score,
                            "raw_content": subpage_text,
                        }
                    )

        # Collect images if available (only from main results to avoid duplication)
        images = []
        for result in results_list:
            image = get_value(result, "image")
            if image and image not in images:  # Avoid duplicate images
                images.append(image)

        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": images,
            "results": formatted_results,
        }

    # Process all queries sequentially with delay to respect rate limit
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (0.25s = 4 requests per second, well within the 5/s limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(0.25)

            result = await process_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing query '{query}': {str(e)}")
            # Add a placeholder result for failed queries to maintain index alignment
            search_docs.append(
                {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": [],
                    "error": str(e),
                }
            )

            # Add additional delay if we hit a rate limit error
            if "429" in str(e):
                print("Rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(1.0)  # Add a longer delay if we hit a rate limit

    return search_docs


@traceable
async def arxiv_search_async(search_queries, load_max_docs=5, get_full_documents=True, load_all_available_meta=True):
    """
    Performs concurrent searches on arXiv using the ArxivRetriever.

    Args:
        search_queries (List[str]): List of search queries or article IDs
        load_max_docs (int, optional): Maximum number of documents to return per query. Default is 5.
        get_full_documents (bool, optional): Whether to fetch full text of documents. Default is True.
        load_all_available_meta (bool, optional): Whether to load all available metadata. Default is True.

    Returns:
        List[dict]: List of search responses from arXiv, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL (Entry ID) of the paper
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str|None  # Full paper content if available
                    },
                    ...
                ]
            }
    """

    async def process_single_query(query):
        try:
            # Create retriever for each query
            retriever = ArxivRetriever(
                arxiv_search=None,  # Use default arxiv search
                arxiv_exceptions=None,  # Use default exception handling
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta,
            )

            # Run the synchronous retriever in a thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))

            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0

            for i, doc in enumerate(docs):
                # Extract metadata
                metadata = doc.metadata

                # Use entry_id as the URL (this is the actual arxiv link)
                url = metadata.get("entry_id", "")

                # Format content with all useful metadata
                content_parts = []

                # Primary information
                if "Summary" in metadata:
                    content_parts.append(f"Summary: {metadata['Summary']}")

                if "Authors" in metadata:
                    content_parts.append(f"Authors: {metadata['Authors']}")

                # Add publication information
                published = metadata.get("Published", "")
                published_str = (
                    published.isoformat() if hasattr(published, "isoformat") else str(published) if published else ""
                )
                if published_str:
                    content_parts.append(f"Published: {published_str}")

                # Add additional metadata if available
                if "primary_category" in metadata:
                    content_parts.append(f"Primary Category: {metadata['primary_category']}")

                if "categories" in metadata and metadata["categories"]:
                    content_parts.append(f"Categories: {', '.join(metadata['categories'])}")

                if "comment" in metadata and metadata["comment"]:
                    content_parts.append(f"Comment: {metadata['comment']}")

                if "journal_ref" in metadata and metadata["journal_ref"]:
                    content_parts.append(f"Journal Reference: {metadata['journal_ref']}")

                if "doi" in metadata and metadata["doi"]:
                    content_parts.append(f"DOI: {metadata['doi']}")

                # Get PDF link if available in the links
                pdf_link = ""
                if "links" in metadata and metadata["links"]:
                    for link in metadata["links"]:
                        if "pdf" in link:
                            pdf_link = link
                            content_parts.append(f"PDF: {pdf_link}")
                            break

                # Join all content parts with newlines
                content = "\n".join(content_parts)

                result = {
                    "title": metadata.get("Title", ""),
                    "url": url,  # Using entry_id as the URL
                    "content": content,
                    "score": base_score - (i * score_decrement),
                    "raw_content": doc.page_content if get_full_documents else None,
                }
                results.append(result)

            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": results,
            }
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{query}': {str(e)}")
            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e),
            }

    # Process queries sequentially with delay to respect arXiv rate limit (1 request per 3 seconds)
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (3 seconds per ArXiv's rate limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(3.0)

            result = await process_single_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{query}': {str(e)}")
            search_docs.append(
                {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": [],
                    "error": str(e),
                }
            )

            # Add additional delay if we hit a rate limit error
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("ArXiv rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(5.0)  # Add a longer delay if we hit a rate limit

    return search_docs


@traceable
async def pubmed_search_async(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    """
    Performs concurrent searches on PubMed using the PubMedAPIWrapper.

    Args:
        search_queries (List[str]): List of search queries
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        email (str, optional): Email address for PubMed API. Required by NCBI.
        api_key (str, optional): API key for PubMed API for higher rate limits.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from PubMed, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL to the paper on PubMed
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str       # Full abstract content
                    },
                    ...
                ]
            }
    """

    async def process_single_query(query):
        try:
            # print(f"Processing PubMed query: '{query}'")

            # Create PubMed wrapper for the query
            wrapper = PubMedAPIWrapper(
                parse=None,  # Use default parsing
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "your_email@example.com",
                api_key=api_key if api_key else "",
            )

            # Run the synchronous wrapper in a thread pool
            loop = asyncio.get_event_loop()

            # Use wrapper.lazy_load instead of load to get better visibility
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))

            print(f"Query '{query}' returned {len(docs)} results")

            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0

            for i, doc in enumerate(docs):
                # Format content with metadata
                content_parts = []

                if doc.get("Published"):
                    content_parts.append(f"Published: {doc['Published']}")

                if doc.get("Copyright Information"):
                    content_parts.append(f"Copyright Information: {doc['Copyright Information']}")

                if doc.get("Summary"):
                    content_parts.append(f"Summary: {doc['Summary']}")

                # Generate PubMed URL from the article UID
                uid = doc.get("uid", "")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""

                # Join all content parts with newlines
                content = "\n".join(content_parts)

                result = {
                    "title": doc.get("Title", ""),
                    "url": url,
                    "content": content,
                    "score": base_score - (i * score_decrement),
                    "raw_content": doc.get("Summary", ""),
                }
                results.append(result)

            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": results,
            }
        except Exception as e:
            # Handle exceptions with more detailed information
            error_msg = f"Error processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            import traceback

            print(traceback.format_exc())  # Print full traceback for debugging

            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e),
            }

    # Process all queries with a reasonable delay between them
    search_docs = []

    # Start with a small delay that increases if we encounter rate limiting
    delay = 1.0  # Start with a more conservative delay

    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests
            if i > 0:  # Don't delay the first request
                # print(f"Waiting {delay} seconds before next query...")
                await asyncio.sleep(delay)

            result = await process_single_query(query)
            search_docs.append(result)

            # If query was successful with results, we can slightly reduce delay (but not below minimum)
            if result.get("results") and len(result["results"]) > 0:
                delay = max(0.5, delay * 0.9)  # Don't go below 0.5 seconds

        except Exception as e:
            # Handle exceptions gracefully
            error_msg = f"Error in main loop processing PubMed query '{query}': {str(e)}"
            print(error_msg)

            search_docs.append(
                {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": [],
                    "error": str(e),
                }
            )

            # If we hit an exception, increase delay for next query
            delay = min(5.0, delay * 1.5)  # Don't exceed 5 seconds

    return search_docs


@traceable
async def linkup_search(search_queries, depth: Literal["standard", "deep"] = "standard"):
    """
    Performs concurrent web searches using the Linkup API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        depth (str, optional): "standard" (default)  or "deep". More details here https://docs.linkup.so/pages/documentation/get-started/concepts

    Returns:
        List[dict]: List of search responses from Linkup API, one per query. Each response has format:
            {
                'results': [            # List of search results
                    {
                        'title': str,   # Title of the search result
                        'url': str,     # URL of the result
                        'content': str, # Summary/snippet of content
                    },
                    ...
                ]
            }
    """
    client = LinkupClient()
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
            client.async_search(
                query,
                depth,
                output_type="searchResults",
            )
        )

    search_results = []
    for response in await asyncio.gather(*search_tasks):
        search_results.append(
            {
                "results": [
                    {"title": result.name, "url": result.url, "content": result.content} for result in response.results
                ],
            }
        )

    return search_results


@traceable
async def google_search_async(
    search_queries: Union[str, List[str]], max_results: int = 5, include_raw_content: bool = True
):
    """
    Performs concurrent web searches using Google.
    Uses Google Custom Search API if environment variables are set, otherwise falls back to web scraping.

    Args:
        search_queries (List[str]): List of search queries to process
        max_results (int): Maximum number of results to return per query
        include_raw_content (bool): Whether to fetch full page content

    Returns:
        List[dict]: List of search responses from Google, one per query
    """

    # Check for API credentials from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    use_api = bool(api_key and cx)

    # Handle case where search_queries is a single string
    if isinstance(search_queries, str):
        search_queries = [search_queries]

    # Define user agent generator
    def get_useragent():
        """Generates a random user agent string."""
        lynx_version = f"Lynx/{random.randint(2, 3)}.{random.randint(8, 9)}.{random.randint(0, 2)}"
        libwww_version = f"libwww-FM/{random.randint(2, 3)}.{random.randint(13, 15)}"
        ssl_mm_version = f"SSL-MM/{random.randint(1, 2)}.{random.randint(3, 5)}"
        openssl_version = f"OpenSSL/{random.randint(1, 3)}.{random.randint(0, 4)}.{random.randint(0, 9)}"
        return f"{lynx_version} {libwww_version} {ssl_mm_version} {openssl_version}"

    # Create executor for running synchronous operations
    executor = None if use_api else concurrent.futures.ThreadPoolExecutor(max_workers=5)

    # Use a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(5 if use_api else 2)

    async def search_single_query(query):
        async with semaphore:
            try:
                results = []

                # API-based search
                if use_api:
                    # The API returns up to 10 results per request
                    for start_index in range(1, max_results + 1, 10):
                        # Calculate how many results to request in this batch
                        num = min(10, max_results - (start_index - 1))

                        # Make request to Google Custom Search API
                        params = {
                            "q": query,
                            "key": api_key,
                            "cx": cx,
                            "start": start_index,
                            "num": num,
                        }
                        print(f"Requesting {num} results for '{query}' from Google API...")

                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                "https://www.googleapis.com/customsearch/v1", params=params
                            ) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    print(f"API error: {response.status}, {error_text}")
                                    break

                                data = await response.json()

                                # Process search results
                                for item in data.get("items", []):
                                    result = {
                                        "title": item.get("title", ""),
                                        "url": item.get("link", ""),
                                        "content": item.get("snippet", ""),
                                        "score": None,
                                        "raw_content": item.get("snippet", ""),
                                    }
                                    results.append(result)

                        # Respect API quota with a small delay
                        await asyncio.sleep(0.2)

                        # If we didn't get a full page of results, no need to request more
                        if not data.get("items") or len(data.get("items", [])) < num:
                            break

                # Web scraping based search
                else:
                    # Add delay between requests
                    await asyncio.sleep(0.5 + random.random() * 1.5)
                    print(f"Scraping Google for '{query}'...")

                    # Define scraping function
                    def google_search(query, max_results):
                        try:
                            lang = "en"
                            safe = "active"
                            start = 0
                            fetched_results = 0
                            fetched_links = set()
                            search_results = []

                            while fetched_results < max_results:
                                # Send request to Google
                                resp = requests.get(
                                    url="https://www.google.com/search",
                                    headers={"User-Agent": get_useragent(), "Accept": "*/*"},
                                    params={
                                        "q": query,
                                        "num": max_results + 2,
                                        "hl": lang,
                                        "start": start,
                                        "safe": safe,
                                    },
                                    cookies={
                                        "CONSENT": "PENDING+987",  # Bypasses the consent page
                                        "SOCS": "CAESHAgBEhIaAB",
                                    },
                                )
                                resp.raise_for_status()

                                # Parse results
                                soup = BeautifulSoup(resp.text, "html.parser")
                                result_block = soup.find_all("div", class_="ezO2md")
                                new_results = 0

                                for result in result_block:
                                    if not isinstance(result, Tag):
                                        continue
                                    link_tag = result.find("a", href=True)
                                    title_tag = cast(Tag, link_tag).find("span", class_="CVA68e") if link_tag else None
                                    description_tag = result.find("span", class_="FrIlee")

                                    if link_tag and title_tag and description_tag:
                                        link_tag = cast(Tag, link_tag)
                                        link = unquote(str(link_tag["href"]).split("&")[0].replace("/url?q=", ""))

                                        if link in fetched_links:
                                            continue

                                        fetched_links.add(link)
                                        title = title_tag.text
                                        description = description_tag.text

                                        # Store result in the same format as the API results
                                        search_results.append(
                                            {
                                                "title": title,
                                                "url": link,
                                                "content": description,
                                                "score": None,
                                                "raw_content": description,
                                            }
                                        )

                                        fetched_results += 1
                                        new_results += 1

                                        if fetched_results >= max_results:
                                            break

                                if new_results == 0:
                                    break

                                start += 10
                                time.sleep(1)  # Delay between pages

                            return search_results

                        except Exception as e:
                            print(f"Error in Google search for '{query}': {str(e)}")
                            return []

                    # Execute search in thread pool
                    loop = asyncio.get_running_loop()
                    search_results = await loop.run_in_executor(executor, lambda: google_search(query, max_results))

                    # Process the results
                    results = search_results

                # If requested, fetch full page content asynchronously (for both API and web scraping)
                if include_raw_content and results:
                    content_semaphore = asyncio.Semaphore(3)

                    async with aiohttp.ClientSession() as session:
                        fetch_tasks = []

                        async def fetch_full_content(result):
                            async with content_semaphore:
                                url = result["url"]
                                headers = {
                                    "User-Agent": get_useragent(),
                                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                                }

                                try:
                                    await asyncio.sleep(0.2 + random.random() * 0.6)
                                    async with session.get(
                                        url, headers=headers, timeout=ClientTimeout(total=10)
                                    ) as response:
                                        if response.status == 200:
                                            # Check content type to handle binary files
                                            content_type = response.headers.get("Content-Type", "").lower()

                                            # Handle PDFs and other binary files
                                            if (
                                                "application/pdf" in content_type
                                                or "application/octet-stream" in content_type
                                            ):
                                                # For PDFs, indicate that content is binary and not parsed
                                                result["raw_content"] = (
                                                    f"[Binary content: {content_type}. Content extraction not supported for this file type.]"
                                                )
                                            else:
                                                try:
                                                    # Try to decode as UTF-8 with replacements for non-UTF8 characters
                                                    html = await response.text(errors="replace")
                                                    soup = BeautifulSoup(html, "html.parser")
                                                    result["raw_content"] = soup.get_text()
                                                except UnicodeDecodeError as ude:
                                                    # Fallback if we still have decoding issues
                                                    result["raw_content"] = f"[Could not decode content: {str(ude)}]"
                                except Exception as e:
                                    print(f"Warning: Failed to fetch content for {url}: {str(e)}")
                                    result["raw_content"] = f"[Error fetching content: {str(e)}]"
                                return result

                        for result in results:
                            fetch_tasks.append(fetch_full_content(result))

                        updated_results = await asyncio.gather(*fetch_tasks)
                        results = updated_results
                        print(f"Fetched full content for {len(results)} results")

                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": results,
                }
            except Exception as e:
                print(f"Error in Google search for query '{query}': {str(e)}")
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": [],
                }

    try:
        # Create tasks for all search queries
        search_tasks = [search_single_query(query) for query in search_queries]

        # Execute all searches concurrently
        search_results = await asyncio.gather(*search_tasks)

        return search_results
    finally:
        # Only shut down executor if it was created
        if executor:
            executor.shutdown(wait=False)


async def scrape_pages(titles: List[str], urls: List[str]) -> str:
    """
    Scrapes content from a list of URLs and formats it into a readable markdown document.

    This function:
    1. Takes a list of page titles and URLs
    2. Makes asynchronous HTTP requests to each URL
    3. Converts HTML content to markdown
    4. Formats all content with clear source attribution

    Args:
        titles (List[str]): A list of page titles corresponding to each URL
        urls (List[str]): A list of URLs to scrape content from

    Returns:
        str: A formatted string containing the full content of each page in markdown format,
             with clear section dividers and source attribution
    """

    # Create an async HTTP client
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        pages = []

        # Fetch each URL and convert to markdown
        for url in urls:
            try:
                # Fetch the content
                response = await client.get(url)
                response.raise_for_status()

                # Convert HTML to markdown if successful
                if response.status_code == 200:
                    # Handle different content types
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type:
                        # Convert HTML to markdown
                        markdown_content = markdownify(response.text)
                        pages.append(markdown_content)
                    else:
                        # For non-HTML content, just mention the content type
                        pages.append(f"Content type: {content_type} (not converted to markdown)")
                else:
                    pages.append(f"Error: Received status code {response.status_code}")

            except Exception as e:
                # Handle any exceptions during fetch
                pages.append(f"Error fetching URL: {str(e)}")

        # Create formatted output
        formatted_output = f"Search results: \n\n"

        for i, (title, url, page) in enumerate(zip(titles, urls, pages)):
            formatted_output += f"\n\n--- SOURCE {i+1}: {title} ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"FULL CONTENT:\n {page}"
            formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output


@tool
async def duckduckgo_search(search_queries: List[str]):
    """Perform searches using DuckDuckGo with retry logic to handle rate limits

    Args:
        search_queries (List[str]): List of search queries to process

    Returns:
        str: A formatted string of search results
    """

    async def process_single_query(query):
        # Execute synchronous search in the event loop's thread pool
        loop = asyncio.get_event_loop()

        def perform_search():
            max_retries = 3
            retry_count = 0
            backoff_factor = 2.0
            last_exception = None

            while retry_count <= max_retries:
                try:
                    results = []
                    with DDGS() as ddgs:
                        # Change query slightly and add delay between retries
                        if retry_count > 0:
                            # Random delay with exponential backoff
                            delay = backoff_factor**retry_count + random.random()
                            print(f"Retry {retry_count}/{max_retries} for query '{query}' after {delay:.2f}s delay")
                            time.sleep(delay)

                            # Add a random element to the query to bypass caching/rate limits
                            modifiers = [
                                "about",
                                "info",
                                "guide",
                                "overview",
                                "details",
                                "explained",
                            ]
                            modified_query = f"{query} {random.choice(modifiers)}"
                        else:
                            modified_query = query

                        # Execute search
                        ddg_results = list(ddgs.text(modified_query, max_results=5))

                        # Format results
                        for i, result in enumerate(ddg_results):
                            results.append(
                                {
                                    "title": result.get("title", ""),
                                    "url": result.get("href", ""),
                                    "content": result.get("body", ""),
                                    "score": 1.0 - (i * 0.1),  # Simple scoring mechanism
                                    "raw_content": result.get("body", ""),
                                }
                            )

                        # Return successful results
                        return {
                            "query": query,
                            "follow_up_questions": None,
                            "answer": None,
                            "images": [],
                            "results": results,
                        }
                except Exception as e:
                    # Store the exception and retry
                    last_exception = e
                    retry_count += 1
                    print(f"DuckDuckGo search error: {str(e)}. Retrying {retry_count}/{max_retries}")

                    # If not a rate limit error, don't retry
                    if "Ratelimit" not in str(e) and retry_count >= 1:
                        print(f"Non-rate limit error, stopping retries: {str(e)}")
                        break

            # If we reach here, all retries failed
            print(f"All retries failed for query '{query}': {str(last_exception)}")
            # Return empty results but with query info preserved
            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(last_exception),
            }

        return await loop.run_in_executor(None, perform_search)

    # Process queries with delay between them to reduce rate limiting
    search_docs = []
    urls = []
    titles = []
    for i, query in enumerate(search_queries):
        # Add delay between queries (except first one)
        if i > 0:
            delay = 2.0 + random.random() * 2.0  # Random delay 2-4 seconds
            await asyncio.sleep(delay)

        # Process the query
        result = await process_single_query(query)
        search_docs.append(result)

        # Safely extract URLs and titles from results, handling empty result cases
        if result["results"] and len(result["results"]) > 0:
            for res in result["results"]:
                if "url" in res and "title" in res:
                    urls.append(res["url"])
                    titles.append(res["title"])

    # If we got any valid URLs, scrape the pages
    if urls:
        return await scrape_pages(titles, urls)
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)


# @tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: Optional[RunnableConfig] = None,
) -> str:
    """
    Fetches results from Tavily search API.

    Args:
        queries (List[str]): List of search queries
        max_results (int): Maximum number of results to return
        topic (Literal['general', 'news', 'finance']): Topic to filter results by

    Returns:
        str: A formatted string of search results
    """
    # Use tavily_search_async with include_raw_content=True to get content directly
    search_results = await tavily_search_async(queries, max_results=max_results, topic=topic, include_raw_content=True)

    # Format the search results directly using the raw_content already provided
    formatted_output = f"\n搜索结果: \n\n"

    # Deduplicate results by URL
    unique_results = {}
    """
    'url': {
        'title': str,             # Title of the search result
        'url': str,               # URL of the result
        'content': str,           # Summary/snippet of content
        'raw_content': str|None,  # Full content if available
        'score': float,           # Relevance score (if available)
        'query': str,             # Original query used to fetch this result
    }
    """
    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = {**result, "query": response["query"]}

    async def noop():
        return None

    configurable = Configuration.from_runnable_config(config)
    logging.info(f"sum or split in configurable in tavily_search: {configurable.process_search_results}")
    max_tokens = configurable.max_tokens
    max_char_to_include = 10_000

    # TODO: share this behavior across all search implementations / tools
    if configurable.process_search_results == "summarize":
        if configurable.summarization_model_provider == "anthropic":
            extra_kwargs = {"betas": ["extended-cache-ttl-2025-04-11"]}
        else:
            extra_kwargs = get_config_value(configurable.summarization_model_kwargs or {})

        logging.debug(f"Using summarization model: {configurable.summarization_model}")
        summarization_model = init_chat_model(
            model=configurable.summarization_model,
            model_provider=configurable.summarization_model_provider,
            max_retries=configurable.max_structured_output_retries,
            **cast(Dict, extra_kwargs),
        )
        summarization_tasks = []
        for result in unique_results.values():
            if not result.get("raw_content"):
                summarization_tasks.append(noop())
            else:
                reduced_str = await reduce_source_str(result["raw_content"], max_tokens=max_tokens)
                summarization_tasks.append(summarize_webpage(summarization_model, reduced_str))

        logging.debug(
            f"Summarizing {len(summarization_tasks)} search results with model {configurable.summarization_model}"
        )
        summaries = await asyncio.gather(*summarization_tasks)
        logging.debug("Summarization completed, stitching results together.")
        unique_results = {
            url: {
                "title": result["title"],
                "content": result["content"] if summary is None else summary,
            }
            for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
        }
    elif configurable.process_search_results == "split_and_rerank":

        # embeddings = cast(Embeddings, init_embeddings("openai:text-embedding-3-small"))
        async def get_embeddings_async():
            return await asyncio.to_thread(
                init_embeddings, "huggingface:BAAI/bge-small-zh", model_kwargs={"local_files_only": True}
            )

        embeddings = cast(Embeddings, await get_embeddings_async())
        # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
        results_by_query = itertools.groupby(unique_results.values(), key=lambda x: x["query"])
        all_retrieved_docs = []
        for query, query_results in results_by_query:
            retrieved_docs = await split_and_rerank_search_results(embeddings, query, cast(List, query_results))
            all_retrieved_docs.extend(retrieved_docs)

        stitched_docs = stitch_documents_by_url(all_retrieved_docs)
        unique_results = {
            doc.metadata["url"]: {"title": doc.metadata["title"], "content": doc.page_content} for doc in stitched_docs
        }

    # Format the unique results
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        if result.get("raw_content"):
            formatted_output += f"FULL CONTENT:\n{result['raw_content'][:(max_char_to_include//len(unique_results))]}"  # Limit content size
        formatted_output += "\n\n" + "-" * 80 + "\n"

    if unique_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


@tool
async def azureaisearch_search(queries: List[str], max_results: int = 5, topic: str = "general") -> str:
    """
    Fetches results from Azure AI Search API.

    Args:
        queries (List[str]): List of search queries

    Returns:
        str: A formatted string of search results
    """
    # Use azureaisearch_search_async with include_raw_content=True to get content directly
    search_results = await azureaisearch_search_async(
        queries, max_results=max_results, topic=topic, include_raw_content=True
    )

    # Format the search results directly using the raw_content already provided
    formatted_output = f"Search results: \n\n"

    # Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = result

    # Format the unique results
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        if result.get("raw_content"):
            formatted_output += f"FULL CONTENT:\n{result['raw_content'][:30000]}"  # Limit content size
        formatted_output += "\n\n" + "-" * 80 + "\n"

    if unique_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


async def select_and_execute_search(
    search_api: str,
    query_list: list[str],
    params_to_pass: dict,
    config: Optional[RunnableConfig] = None,
) -> str:
    """Select and execute the appropriate search API.

    Args:
        search_api: Name of the search API to use
        query_list: List of search queries to execute
        params_to_pass: Parameters to pass to the search API

    Returns:
        Formatted string containing search results

    Raises:
        ValueError: If an unsupported search API is specified
    """
    if search_api == "tavily":
        # Tavily search tool used with both workflow and agent
        # and returns a formatted source string
        # return await tavily_search.ainvoke(
        #     {"queries": query_list, **params_to_pass}, config=config
        # )
        return await tavily_search(queries=query_list, **params_to_pass, config=config)
    elif search_api == "duckduckgo":
        # DuckDuckGo search tool used with both workflow and agent
        return await duckduckgo_search.ainvoke({"search_queries": query_list})
    elif search_api == "perplexity":
        search_results = perplexity_search(query_list, **params_to_pass)
    elif search_api == "exa":
        search_results = await exa_search(query_list, **params_to_pass)
    elif search_api == "arxiv":
        search_results = await arxiv_search_async(query_list, **params_to_pass)
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(query_list, **params_to_pass)
    elif search_api == "linkup":
        search_results = await linkup_search(query_list, **params_to_pass)
    elif search_api == "googlesearch":
        search_results = await google_search_async(query_list, **params_to_pass)
    elif search_api == "azureaisearch":
        search_results = await azureaisearch_search_async(query_list, **params_to_pass)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    return deduplicate_and_format_sources(
        search_results, max_tokens_per_source=4000, deduplication_strategy="keep_first"
    )


class Summary(BaseModel):
    summary: str
    key_excerpts: list[str]


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content."""
    try:
        user_input_content = "请总结以下网页内容：\n"
        if isinstance(model, ChatAnthropic):
            user_input_content = [
                {
                    "type": "text",
                    "text": user_input_content,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ]
        logging.debug(f"Summarizing webpage content{webpage_content[:100]}...")  # Log first 100 chars for context
        summary = cast(
            Summary,
            await model.with_structured_output(Summary)
            .with_retry(stop_after_attempt=2)
            .ainvoke(
                [
                    {
                        "role": "system",
                        "content": SUMMARIZATION_PROMPT.format(webpage_content=webpage_content),
                    },
                    {"role": "user", "content": user_input_content},
                ]
            ),
        )
        logging.debug("Webpage content{webpage_content[:100]} summarized successfully.")
    except:
        logging.error("Failed to summarize webpage content{webpage_content[:100]}, returning raw content instead.")
        # fall back on the raw content
        return webpage_content

    def format_summary(summary: Summary):
        excerpts_str = "\n".join(f"- {e}" for e in summary.key_excerpts)
        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{excerpts_str}\n</key_excerpts>"""

    return format_summary(summary)


async def split_and_rerank_search_results(
    embeddings: Embeddings,
    query: str,
    search_results: list[dict],
    max_chunks: int = 5,
    use_local_kb: bool = True,
    save_to_local: bool = True,
) -> List[Document]:
    """
    使用新的知识库管理策略的版本 - 修正整合逻辑
    """
    if not use_local_kb:
        # 如果不使用本地知识库，直接处理搜索结果
        return _process_search_results_only(embeddings, query, search_results, max_chunks)

    # 使用知识库管理器
    kb_manager = KnowledgeBaseManager(embeddings)

    # 加载或创建知识库
    existing_kb, existing_docs, kb_id, is_new = await kb_manager.load_or_create_knowledge_base(query)

    # 处理搜索结果
    new_documents = []
    if search_results:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, add_start_index=True)

        search_documents = [
            Document(
                page_content=result.get("raw_content") or result["content"],
                metadata={
                    "url": result["url"],
                    "title": result["title"],
                    "source": "search_result",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "kb_id": kb_id,  # 添加知识库ID标识
                },
            )
            for result in search_results
        ]

        new_splits = text_splitter.split_documents(search_documents)
        new_documents.extend(new_splits)

    # 去重和合并文档
    all_documents = existing_docs.copy()
    existing_urls = {doc.metadata.get("url", "") for doc in existing_docs}

    new_unique_docs = []
    for doc in new_documents:
        url = doc.metadata.get("url", "")
        if url not in existing_urls:
            all_documents.append(doc)
            new_unique_docs.append(doc)
            existing_urls.add(url)

    # 日志信息
    if new_unique_docs:
        if is_new:
            logging.info(f"Adding {len(new_unique_docs)} documents to new knowledge base {kb_id}")
        else:
            logging.info(f"Integrating {len(new_unique_docs)} new documents into existing knowledge base {kb_id}")
    else:
        logging.info(f"No new unique documents to add to knowledge base {kb_id}")

    # 创建或更新向量存储
    if existing_kb and new_unique_docs:
        # 情况1: 现有知识库 + 新文档 -> 整合到现有知识库
        try:
            existing_kb.add_documents(new_unique_docs)
            vector_store = existing_kb
            logging.info(f"Successfully integrated {len(new_unique_docs)} new documents into existing KB {kb_id}")
        except Exception as e:
            logging.error(f"Failed to integrate documents into existing vector store {kb_id}: {e}")
            # 降级处理：重建向量存储
            logging.info(f"Rebuilding vector store for KB {kb_id} with all {len(all_documents)} documents")
            vector_store = InMemoryVectorStore(embeddings)
            vector_store.add_documents(all_documents)
    elif existing_kb and not new_unique_docs:
        # 情况2: 现有知识库 + 无新文档 -> 直接使用现有知识库
        vector_store = existing_kb
        logging.info(f"Using existing knowledge base {kb_id} without modifications")
    elif not existing_kb and new_unique_docs:
        # 情况3: 新知识库 + 新文档 -> 创建新向量存储
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(all_documents)
        logging.info(f"Created new knowledge base {kb_id} with {len(all_documents)} documents")
    else:
        # 情况4: 新知识库 + 无新文档 -> 创建空向量存储（不太可能出现）
        vector_store = InMemoryVectorStore(embeddings)
        logging.warning(f"Created empty knowledge base {kb_id}")

    # 检索相关文档
    if all_documents:
        retrieved_docs = vector_store.similarity_search(query, k=max_chunks)

        # 添加检索元数据
        for doc in retrieved_docs:
            doc.metadata["retrieved_at"] = datetime.datetime.now().isoformat()
            doc.metadata["query"] = query
            doc.metadata["kb_id"] = kb_id

        # 保存更新后的知识库（只有在有新内容或是新知识库时才保存）
        if save_to_local and (new_unique_docs or is_new):
            await kb_manager.save_knowledge_base(kb_id, vector_store, query, all_documents)

            # 更新日志信息
            if is_new:
                logging.info(f"Saved new knowledge base {kb_id} with {len(all_documents)} documents")
            else:
                logging.info(f"Updated existing knowledge base {kb_id}, now contains {len(all_documents)} documents")

        logging.info(f"Retrieved {len(retrieved_docs)} chunks from KB {kb_id} (total: {len(all_documents)} docs)")
        return retrieved_docs
    else:
        logging.warning(f"No documents available for retrieval in KB {kb_id}")
        return []


def _process_search_results_only(
    embeddings: Embeddings, query: str, search_results: list[dict], max_chunks: int = 5
) -> List[Document]:
    """
    仅处理搜索结果，不使用本地知识库的辅助函数
    """
    if not search_results:
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, add_start_index=True)

    search_documents = [
        Document(
            page_content=result.get("raw_content") or result["content"],
            metadata={
                "url": result["url"],
                "title": result["title"],
                "source": "search_result",
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )
        for result in search_results
    ]

    # 分割文档
    splits = text_splitter.split_documents(search_documents)

    # 创建临时向量存储
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(splits)

    # 检索相关文档
    retrieved_docs = vector_store.similarity_search(query, k=max_chunks)

    # # 添加检索元数据
    # for doc in retrieved_docs:
    #     doc.metadata["retrieved_at"] = datetime.datetime.now().isoformat()
    #     doc.metadata["query"] = query

    logging.info(f"Processed {len(splits)} chunks without local KB, retrieved {len(retrieved_docs)}")
    return retrieved_docs


def stitch_documents_by_url(documents: list[Document]) -> list[Document]:
    url_to_docs: defaultdict[str, list[Document]] = defaultdict(list)
    url_to_snippet_hashes: defaultdict[str, set[str]] = defaultdict(set)
    for doc in documents:
        snippet_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
        url = doc.metadata["url"]
        # deduplicate snippets by the content
        if snippet_hash in url_to_snippet_hashes[url]:
            continue

        url_to_docs[url].append(doc)
        url_to_snippet_hashes[url].add(snippet_hash)

    # stitch retrieved chunks into a single doc per URL
    stitched_docs = []
    for docs in url_to_docs.values():
        stitched_doc = Document(
            page_content="\n\n".join([f"...{doc.page_content}..." for doc in docs]),
            metadata=docs[0].metadata,
        )
        stitched_docs.append(stitched_doc)

    return stitched_docs


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    today = datetime.datetime.now()
    try:
        # Unix style format
        return today.strftime("%a %b %-d, %Y")
    except ValueError:
        # Windows style
        return today.strftime("%a %b %d, %Y").replace(" 0", " ")


async def load_mcp_server_config(path: str) -> dict:
    """Load MCP server configuration from a file."""

    def _load():
        with open(path, "r") as f:
            config = json.load(f)
        return config

    config = await asyncio.to_thread(_load)
    return config


async def main():

    queries = [
        "湖南省制造业产品质量合格率2024年94%突破 政策背景 重要意义",
        "2024年湖南省产品质量监管创新措施 制造业质量提升举措",
    ]
    config = RunnableConfig()
    config["configurable"] = {"process_search_results": "split_and_rerank"}
    # full_inspect_kb(queries[0])
    # source_str = await tavily_search(queries, config=config)
    # print(source_str)
    # """
    # 'url': {
    #     'title': str,             # Title of the search result
    #     'url': str,               # URL of the result
    #     'content': str,           # Summary/snippet of content
    # }
    # """
    # for url, result in unique_results.items():
    #     logging.info(f"\n\n--- SOURCE: {result['title']} ---")
    #     logging.info(f"URL: {url}")
    #     logging.info(f"content:\n{result['content']}")


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_dir = logs_dir / f"log_{timestamp}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(filename)s | %(funcName)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_dir,
        encoding="utf-8",
        filemode="w",
    )
    # demo_knowledge_base_management()
    asyncio.run(main())
