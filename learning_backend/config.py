"""Runtime configuration and dependency construction."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .store import ContextStore


@dataclass(slots=True)
class RuntimeDependencies:
    llm: ChatOpenAI
    embeddings: OpenAIEmbeddings
    tavily_search: TavilySearchResults
    context_store: ContextStore


def build_runtime(
    *,
    llm_model: str = "gpt-4o",
    embedding_model: str = "text-embedding-3-large",
    tavily_max_results: int = 3,
) -> RuntimeDependencies:
    """Create the external dependencies used by the learning workflow."""

    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

    return RuntimeDependencies(
        llm=ChatOpenAI(model=llm_model, temperature=0),
        embeddings=OpenAIEmbeddings(model=embedding_model),
        tavily_search=TavilySearchResults(max_results=tavily_max_results),
        context_store=ContextStore(),
    )
