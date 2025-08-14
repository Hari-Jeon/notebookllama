from dotenv import load_dotenv
import logging
import os

from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.openai import OpenAIResponses
from llama_index.llms.ollama import Ollama
from typing import Union, cast

logger = logging.getLogger(__name__)
load_dotenv()

LLM = None
RETR = None
QE = None

if (
    os.getenv("LLAMACLOUD_API_KEY", None)
    and os.getenv("LLAMACLOUD_PIPELINE_ID", None)
):
    RETR = LlamaCloudIndex(
        api_key=os.getenv("LLAMACLOUD_API_KEY"),
        pipeline_id=os.getenv("LLAMACLOUD_PIPELINE_ID", None)
    ).as_retriever()

if os.getenv("OPENAI_API_KEY", None):
    LLM = OpenAIResponses(
        model=os.getenv("OPENAI_API_MODEL", "gpt-4.1"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
elif os.getenv("OLLAMA_BASE_URL", None):
    LLM = Ollama(
        model=os.getenv("OLLAMA_MODEL", "llama2"),
        base_url=os.getenv("OLLAMA_BASE_URL")
    )

if RETR and LLM:
    QE = CitationQueryEngine(
        retriever=RETR,
        llm=LLM,
        citation_chunk_size=256,
        citation_chunk_overlap=50,
    )
else:
    logger.warning("Missing LLM or retriever - Query Engine not initialized")


async def query_index(question: str) -> Union[str, None]:
    response = await QE.aquery(question)
    response = cast(Response, response)
    sources = []
    if not response.response:
        return None
    if response.source_nodes is not None:
        sources = [node.text for node in response.source_nodes]
    return (
        "## Answer\n\n"
        + response.response
        + "\n\n## Sources\n\n- "
        + "\n- ".join(sources)
    )
