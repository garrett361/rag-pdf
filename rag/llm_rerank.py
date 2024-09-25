from typing import Any, List, Optional, Union

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.llms.openllm import OpenLLM
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from rag._defaults import DEFAULT_SCORE_PROMPT
from rag._utils import get_llama3_1_score_str, get_llm_answer


def get_llama3_1_score(
    excerpt: str,
    query: str,
    tokenizer: PreTrainedTokenizer,
    llm: OpenLLM,
    system_prompt: str = DEFAULT_SCORE_PROMPT,
) -> str:
    prefix = get_llama3_1_score_str(
        excerpt=excerpt, query=query, tokenizer=tokenizer, system_prompt=system_prompt
    )
    out = get_llm_answer(llm, prefix)
    try:
        score = int(out.text.strip())
    except Exception as e:
        print(f"Exception {e} raised on scoring {excerpt=}, {query=}")
        score = 0
    return score


class LLama31Reranker(BaseNodePostprocessor):
    """
    LLama 3.1 reranker class. Assumes and endpoint (OpenLLM class) is being used for now.
    """

    llm: LLM = Field(description="The LLM to rerank with.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    tokenizer: Union[AutoTokenizer, PreTrainedTokenizerFast] = Field(description="Tokenizer")
    system_prompt: str = Field(description="System prompt")
    _model: Any = PrivateAttr()

    def __init__(
        self,
        llm: OpenLLM,
        tokenizer: AutoTokenizer,
        top_n: int = 2,
        system_prompt: str = DEFAULT_SCORE_PROMPT,
    ):
        super().__init__(top_n=top_n, llm=llm, tokenizer=tokenizer, system_prompt=system_prompt)

    @classmethod
    def class_name(cls) -> str:
        return "LLama31Reranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        scores = []
        for n in nodes:
            score = get_llama3_1_score(
                excerpt=n.text,
                query=query_bundle.query_str,
                tokenizer=self.tokenizer,
                llm=self.llm,
                system_prompt=self.system_prompt,
            )
            scores.append(score)

        reranked_scores_and_nodes = list(
            sorted(zip(scores, nodes, strict=True), key=lambda x: -x[0])
        )
        # for score, node in reranked_scores_and_nodes:
        #     print_in_box(f"Query: {query_bundle.query_str}\n\nScore: {score}\n\n{node.text=}")

        reranked_nodes = [node for _, node in reranked_scores_and_nodes][: self.top_n]

        return reranked_nodes
