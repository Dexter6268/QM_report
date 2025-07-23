import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict, Literal
from langchain_core.runnables import RunnableConfig

DEFAULT_REPORT_STRUCTURE = """请使用以下结构撰写用户提供主题的报告：

1. 引言（无需检索）
    - 简要介绍主题领域

2. 主体部分：
    - 每个小节应聚焦于主题的一个子话题

"""

DEFAULT_CONCLUSION_STRUCTURE = """请使用以下结构撰写用户提供主题的报告:
1. 问题分析
    - 内容应针对于现有的报告内容
2. 政策建议
    - 内容应对应于问题分析
"""


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"
    NONE = "none"


@dataclass(kw_only=True)
class WorkflowConfiguration:
    """Configuration for the workflow/graph-based implementation (graph.py)."""

    # Common configuration
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    conclusion_structure: str = DEFAULT_CONCLUSION_STRUCTURE
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str = "deepseek"
    summarization_model: str = "deepseek-chat"
    summarization_model_kwargs: Optional[Dict[str, Any]] = None
    max_structured_output_retries: int = 3
    include_source_str: bool = False

    # Workflow-specific configuration
    number_of_queries: int = 2  # Number of search queries to generate per iteration
    max_search_depth: int = 2  # Maximum number of reflection + search iterations
    planner_provider: str = "deepseek"
    planner_model: str = "deepseek-chat"
    planner_model_kwargs: Optional[Dict[str, Any]] = None
    writer_provider: str = "deepseek"
    writer_model: str = "deepseek-chat"
    writer_model_kwargs: Optional[Dict[str, Any]] = None
    max_tokens: int = 5000  # Maximum tokens for each section of the report

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "WorkflowConfiguration":
        """Create a WorkflowConfiguration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})


@dataclass(kw_only=True)
class MultiAgentConfiguration:
    """Configuration for the multi-agent implementation (multi_agent.py)."""

    # Common configuration
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str = "anthropic"
    summarization_model: str = "claude-3-5-haiku-latest"
    include_source_str: bool = False

    # Multi-agent specific configuration
    number_of_queries: int = 2  # Number of search queries to generate per section
    supervisor_model: str = "anthropic:claude-3-7-sonnet-latest"
    researcher_model: str = "anthropic:claude-3-7-sonnet-latest"
    ask_for_clarification: bool = False  # Whether to ask for clarification from the user
    # MCP server configuration
    mcp_server_config: Optional[Dict[str, Any]] = None
    mcp_prompt: Optional[str] = None
    mcp_tools_to_include: Optional[list[str]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "MultiAgentConfiguration":
        """Create a MultiAgentConfiguration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})


# Keep the old Configuration class for backward compatibility
Configuration = WorkflowConfiguration
