from typing import Any, List, Optional, Union

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.llms.openllm import OpenLLM
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from rag._defaults import DEFAULT_SCORE_PROMPT
from rag._utils import get_llama3_1_score_str, get_llm_answer, print_in_box


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


# TODO: @garrett.goon - Remove min_score test code
class LLama31Reranker(BaseNodePostprocessor):
    """
    LLama 3.1 reranker class. Assumes and endpoint (OpenLLM class) is being used for now.
    """

    llm: LLM = Field(description="The LLM to rerank with.")
    top_n: Optional[int] = Field(description="Number of nodes to return sorted by score.")
    min_score: Optional[int] = Field(description="Minimum score to keep")
    tokenizer: Union[AutoTokenizer, PreTrainedTokenizerFast] = Field(description="Tokenizer")
    system_prompt: str = Field(description="System prompt")
    _model: Any = PrivateAttr()

    def __init__(
        self,
        llm: OpenLLM,
        tokenizer: AutoTokenizer,
        top_n: Optional[int] = None,
        min_score: Optional[int] = None,
        system_prompt: str = DEFAULT_SCORE_PROMPT,
    ):
        super().__init__(
            top_n=top_n,
            min_score=min_score,
            llm=llm,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
        )
        if sum((bool(top_n), bool(min_score))) != 1:
            raise ValueError("Exactly one of top_n or min_score must be provided.")

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

        # Debug/sanity check printing
        crossed_min_score_threshold = self.min_score is None
        for score, node in reranked_scores_and_nodes:
            if not crossed_min_score_threshold and score < self.min_score:
                crossed_min_score_threshold = True
                threshold_str = f" MIN SCORE {self.min_score} CUTOFFF "
                print(f"{threshold_str:X^80}")
            print_in_box(
                f"Query: {query_bundle.query_str}\n\nLLM Score: {score}\n\nSimilarity Score: {round((node.score * 100),3)}%\n\n{node.text=}"
            )

        if self.top_n:
            reranked_nodes = [node for _, node in reranked_scores_and_nodes][: self.top_n]
        elif self.min_score:
            reranked_nodes = [
                node for score, node in reranked_scores_and_nodes if score >= self.min_score
            ]
            # Safety valve for the case where no node passes the threshold. Return the best single
            # node.
            if not reranked_nodes:
                print(f"No nodes surpass score threshold ({self.min_score}). Returning best node.")
                reranked_nodes = [node for _, node in reranked_scores_and_nodes][:1]

        return reranked_nodes
