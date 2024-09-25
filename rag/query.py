import argparse
import os
from copy import deepcopy
from typing import Optional

import pandas as pd
import weaviate
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openllm import OpenLLM
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from transformers import AutoTokenizer

from rag._defaults import (
    DEFAULT_ALPHA,
    DEFAULT_CUTOFF,
    DEFAULT_HF_CHAT_MODEL,
    DEFAULT_HF_EMBED_MODEL,
    DEFAULT_MAX_NEW_TOKS,
    DEFAULT_TEMP,
    DEFAULT_TOP_K_RETRIEVER,
    DEFAULT_TOP_P,
)
from rag._utils import (
    get_llama3_1_instruct_str,
    get_llm_answer,
    get_local_llm,
    get_tag_from_dir,
    print_references,
)
from rag.llm_rerank import LLama31Reranker


class QuestionAnsweredNodePostprocessor(BaseNodePostprocessor):
    """
    Creates new nodes with the question answered by the extract appended.
    """

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> list[NodeWithScore]:
        # subtracts 1 from the score
        copied_nodes = [deepcopy(n) for n in nodes]
        for cn in copied_nodes:
            cn.node.text += (
                f"\n\nQuestion answered by above extract: {cn.metadata['QuestionAnswered']}"
            )

        return copied_nodes


def load_data(
    embedding_model_path: str, weaviate_client: weaviate.WeaviateClient
) -> tuple[VectorStoreIndex, dict]:
    if embedding_model_path.startswith("http"):
        print(f"\nUsing Embedding API model endpoint: {embedding_model_path}\n")
        embed_model = OpenAIEmbedding(api_base=embedding_model_path, api_key="dummy")
    else:
        print(f"\nEmbedding model: {embedding_model_path}\n")
        embed_model = HuggingFaceEmbedding(model_name=embedding_model_path)

    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name="Documents")

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    # Fetch all objects from the Weaviate class
    collection = weaviate_client.collections.get("Documents")
    results = []
    for item in collection.iterator():
        results.append(item)

    return index, results


def create_retriever(
    index: VectorStoreIndex, cutoff: str, top_k_retriever: int, alpha=0.5, filters=None
) -> VectorIndexRetriever:
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k_retriever,
        filters=filters,
        vector_store_query_mode="hybrid",
        alpha=alpha,
        # SimilarityPostprocessor not working with weaviate==0.1.0,
        # llama-index-vector-stores-weaviate==1.1.1 See
        # https://github.com/run-llama/llama_index/issues/14728
        # node_postprocessors=[SimilarityPostprocessor(similarity=cutoff)],
    )
    return retriever


def get_nodes(
    query: str,
    retriever: VectorIndexRetriever,
    reranker: Optional[BaseNodePostprocessor] = None,
    cutoff: Optional[float] = None,
) -> list[NodeWithScore]:
    """
    Retrieve the most relevant chunks, given the query.
    """
    # Wrap in a QueryBundle class in order to use reranker.
    # NOTE: @garrett.goon - Liam tried wrapping with query_str below, but found it didn't help.
    # query_str = f"Represent this sentence for searching relevant passages: {query}"
    query_bundle = QueryBundle(query)
    nodes = retriever.retrieve(query_bundle)
    # Weaviate returns nodes in reversed relevant order
    nodes = nodes[::-1]
    # Sanity check that the nodes are now sorted in descending order
    scores = [n.score for n in nodes]
    assert sorted(scores) == list(reversed(scores)), f"{sorted(scores)=}, {list(reversed(scores))=}"

    # TODO: @garrett.goon - Delete this hack for filtering nodes based on a cutoff for
    # weaviate indexes. See https://github.com/run-llama/llama_index/issues/14728
    if cutoff:
        filtered_nodes = [n for n in nodes if n.score >= cutoff]
        # If no nodes survive the filter, just take the best node left to avoid erroring
        if filtered_nodes:
            nodes = filtered_nodes
        else:
            print("No node passed the cutoff, using best node")
            nodes = nodes[:1]

    if reranker is not None:
        print("------------------\n")
        print(f"Reranking {len(nodes)} nodes down to {reranker.top_n} ")
        print("\n------------------")
        # Append the generated question to the node text prior to re-ranking so that the re-ranker
        # has additional, hopefully relevant text to match against.
        question_appended_nodes = QuestionAnsweredNodePostprocessor().postprocess_nodes(
            nodes, query_bundle
        )
        filtered_nodes = reranker.postprocess_nodes(question_appended_nodes, query_bundle)
        filtered_node_ids = {fn.node.id_ for fn in filtered_nodes}
        # Then return the original nodes without the question appended, so that generation does not
        # rely on additional info not preset in the original chunks.
        nodes = [n for n in nodes if n.node.id_ in filtered_node_ids]

        print("------------------\n\n")
        print(f"After reranking, {len(nodes)} nodes: {[n.node.id_ for n in nodes]=} ")
        print("\n\n------------------")

    # print(f"NODES: {query=}, {cutoff=}, {[n.node.id_ for n in nodes]=}")
    return nodes


if __name__ == "__main__":
    print("\n**********  QUERYING **********\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Query to ask of the llm")
    parser.add_argument("--path-to-db", type=str, default="db", help="path to chroma db")
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default=DEFAULT_HF_EMBED_MODEL,
        help="local path or URL to sentence transformer model",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_HF_CHAT_MODEL,
        help="local path or URL to chat model",
    )
    parser.add_argument(
        "--chat-model-endpoint",
        default=None,
        type=str,
        help="HTTP path to model endpoint, if serving",
    )
    parser.add_argument(
        "--top-k-retriever",
        default=DEFAULT_TOP_K_RETRIEVER,
        type=int,
        help="top k results for retriever",
    )
    parser.add_argument(
        "--top-k-reranker",
        default=None,
        type=int,
        help="top k results for reranker",
    )
    parser.add_argument(
        "--temp",
        default=DEFAULT_TEMP,
        type=float,
        help="Generation temp",
    )
    parser.add_argument(
        "--top-p",
        default=DEFAULT_TOP_P,
        type=float,
        help="top p probability for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        default=DEFAULT_MAX_NEW_TOKS,
        type=int,
        help="Max generation toks",
    )
    parser.add_argument(
        "--cutoff",
        default=DEFAULT_CUTOFF,
        type=float,
        help="Filter out docs with score below cutoff.",
    )
    parser.add_argument(
        "--use-4bit-quant",
        action="store_true",
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--streaming",
        help="stream responses",
        action="store_true",
    )
    parser.add_argument(
        "--folder",
        default=None,
        type=str,
        help="Only use documents initially under that folder name.",
    )
    parser.add_argument(
        "--query-file",
        default=None,
        type=str,
        help="txt file containing a single query per line or xlsx file containing queries in first columns. Overrides the --query argument.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        type=str,
        help="Save queries output to csv files under that path.",
    )
    # https://medium.com/llamaindex-blog/llamaindex-enhancing-retrieval-performance-with-alpha-tuning-in-hybrid-search-in-rag-135d0c9b8a00
    parser.add_argument(
        "--alpha",
        default=DEFAULT_ALPHA,
        type=float,
        help="Controls the balance between keyword (alpha=0.0) and vector (alpha=1.0) search",
    )
    parser.add_argument(
        "--st-reranker",
        action="store_true",
        help="Use a SentenceTransformerRerank model, if reranking. Default is to use the chat LLM to rerank.",
    )

    args = parser.parse_args()
    if args.st_reranker and args.top_k_reranker:
        raise ValueError("--top-k-reranker must be set when --st-reranker is chosen")

    if sum((bool(args.query), bool(args.query_file))) != 1:
        raise ValueError("Exactly one of --query or --query-file must be provided.")

    if "Meta-Llama-3.1" not in args.model_name:
        # Only tested with Meta-Llama-3.1 so far. The system prompt and tokenization would need to
        # be adjusted for other models.
        raise ValueError(f"Script expects a Llama-3.1 model, not {args.model_name}")

    if args.top_k_reranker and args.top_k_reranker > args.top_k_retriever:
        raise ValueError("top_k_reranker, if provided, must be smaller than top_k_retriever.")

    with weaviate.WeaviateClient(
        embedded_options=weaviate.embedded.EmbeddedOptions(persistence_data_path=args.path_to_db)
    ) as weaviate_client:
        index, chunks = load_data(args.embedding_model_path, weaviate_client)

        all_tags = []
        for c in chunks:
            eltags = c.properties["tag"]
            if eltags not in all_tags:
                all_tags.append(eltags)
        print("\nAll tags found: " + str(all_tags) + "\n")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        generate_kwargs = {
            "do_sample": True,
            "temperature": args.temp,
            "top_p": args.top_p,
        }
        if args.chat_model_endpoint:
            print(f"\nUsing hosted LLM at: {args.chat_model_endpoint}\n")
            llm = OpenLLM(
                model=args.model_name,
                api_base=args.chat_model_endpoint,
                api_key="fake",
                generate_kwargs=generate_kwargs,
                max_tokens=args.max_new_tokens,
            )
        else:
            print(f"\nUsing local {args.model_name} LLM\n")
            llm = get_local_llm(
                args.model_name,
                tokenizer,
                args.max_new_tokens,
                args.use_4bit_quant,
                generate_kwargs,
            )
        if args.top_k_reranker is None:
            print("Using no reranker")
            reranker = None
        elif args.st_reranker:
            # TODO: @garrett.goon - Don't hardcode ST model.
            reranker = SentenceTransformerRerank(
                model="BAAI/bge-reranker-large", top_n=args.top_k_reranker
            )
            print(f"Using {reranker.__class__.__name__}")
        else:
            reranker = LLama31Reranker(llm=llm, tokenizer=tokenizer, top_n=args.top_k_reranker)
            print(f"Using {reranker.__class__.__name__}")

        if args.output_folder and not os.path.exists(args.output_folder):
            print(args.output_folder + " does not exist yet, creating it...")
            os.makedirs(args.output_folder)

        # Get the list of queries from the queries file or the query arg
        if args.query_file:
            print("Using " + args.query_file + " as query list")

            if args.query_file[-4:] == ".txt":
                query_file = open(args.query_file, "r")
                query_lines = query_file.readlines()
                query_file.close()

                query_list = []
                for query in query_lines:
                    query_list.append(query.replace("\n", ""))
            elif args.query_file[-5:] == ".xlsx":
                query_df = pd.read_excel(args.query_file, header=None)
                query_list = [q for q in query_df[0]]
            else:
                print("Format of query file not supported")
        else:
            query_list = [args.query]

        # Filter by the provided tag or loop over all tags
        if args.folder:
            tags = [get_tag_from_dir(args.folder)]
        else:
            tags = all_tags

        # Sanity check
        for tag in tags:
            if tag not in all_tags:
                raise ValueError(
                    f"Invalid folder. Corresponding {tag=} not found in set of all tags: {all_tags}."
                )

        for tag in tags:
            # Tracking results individually for each tag
            d = {}
            d["Queries"] = []
            d["Answers"] = []
            d["Main Source"] = []

            d["Queries"] = query_list

            for query in query_list:
                # After moving to weaviate, needed to change key="Tag" to the lower-cased key="tag"
                filters = MetadataFilters(
                    filters=[MetadataFilter(key="tag", value=tag)], condition="or"
                )
                retriever = create_retriever(
                    index=index,
                    cutoff=args.cutoff,
                    top_k_retriever=args.top_k_retriever,
                    alpha=args.alpha,
                    filters=filters,
                )
                nodes = get_nodes(query, retriever, reranker=reranker, cutoff=args.cutoff)
                print_references(nodes)

                prefix = get_llama3_1_instruct_str(query, nodes, tokenizer)
                print("\n\nApply query to " + tag + " folder only")
                output_response = get_llm_answer(llm, prefix, streaming=False)
                print("**************************\n\n")
                print(f"\n\n{query=}\n\n{tag=}\n\n{prefix=}\n\n{output_response.text=}\n")
                print("\n\n**************************")

                d["Answers"].append(output_response.text)
                d["Main Source"].append(
                    nodes[0].node.metadata["Source"]
                    + ", page "
                    + str(nodes[0].node.metadata["PageNumber"])
                )

            if args.output_folder:
                output_df = pd.DataFrame(data=d)

                suffix = "all_documents_mixed"
                if tag:
                    suffix = tag
                elif args.folder:
                    suffix = args.folder

                xlsx_name = args.output_folder + "/extracted_info_" + suffix + ".xlsx"
                print("Saving output to " + xlsx_name)
                output_df.to_excel(xlsx_name, index=False)
