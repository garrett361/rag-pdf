import argparse
from pprint import pprint
from textwrap import dedent

import chromadb
import torch
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import AutoTokenizer, PreTrainedTokenizer

from rag._defaults import DEFAULT_HF_CHAT_MODEL, DEFAULT_HF_EMBED_MODEL, DEFAULT_SYTEM_PROMPT


def get_llama3_1_instruct_str(
    query: str,
    context_str: str,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str = DEFAULT_SYTEM_PROMPT,
) -> str:
    # https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3
    context_and_query = f"""
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dedent(context_and_query).strip("\n")},
    ]
    return tokenizer.decode(tokenizer.apply_chat_template(messages, add_generation_prompt=True))


def get_llm(
    model_name: str,
    stopping_ids: list[int],
    temp: float,
    max_new_tokens: int,
    top_p: float,
    use_4bit_quant: bool,
) -> HuggingFaceLLM:
    pprint(f"Using HF model: {model_name}")

    generate_kwargs = {
        "do_sample": True,
        "temperature": temp,
        "top_p": top_p,
    }
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if use_4bit_quant:
        if not torch.cuda.is_available():
            raise ValueError("--use-4bit-quant requires a GPU")
        from transformers import BitsAndBytesConfig

        model_kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        }
    else:
        model_kwargs = {"torch_dtype": torch.bfloat16}
    llm = HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        generate_kwargs=generate_kwargs,
        max_new_tokens=max_new_tokens,
        stopping_ids=stopping_ids,
        model_kwargs=model_kwargs,
    )
    pprint(f"Loaded model {model_name}")
    return llm


def load_data(
    embedding_model_path: str, path_to_db: str
) -> tuple[VectorStoreIndex, chromadb.GetResult]:
    if embedding_model_path.startswith("http"):
        pprint(f"Using Embedding API model endpoint: {embedding_model_path}")
        embed_model = OpenAIEmbedding(api_base=embedding_model_path, api_key="dummy")
    else:
        pprint(f"Embedding model: {embedding_model_path}")
        embed_model = HuggingFaceEmbedding(model_name=embedding_model_path)
    chroma_client = chromadb.PersistentClient(path_to_db)
    chroma_collection = chroma_client.get_collection(name="documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index, chroma_collection.get()


DEFAULT_SYTEM_PROMPT = """
You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
"""
DEFAULT_SYTEM_PROMPT = dedent(DEFAULT_SYTEM_PROMPT).strip("\n")


def create_query_engine(cutoff: float, top_k: int, filters=None):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k, filters=filters)
    # "no_text": just return the retrieved nodes without LLM processing
    response_synthesizer = get_response_synthesizer(response_mode="no_text", streaming=False)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity=cutoff)],
    )
    return query_engine


if __name__ == "__main__":
    print("\n**********  QUERYING **********\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Query to ask of the llm")
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
        "--top-k",
        default=5,
        type=int,
        help="top k results for retriever",
    )
    parser.add_argument(
        "--temp",
        default=0.2,
        type=float,
        help="Generation temp",
    )
    parser.add_argument(
        "--top-p",
        default=None,
        type=float,
        help="top p probability for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        default=250,
        type=int,
        help="Max generation toks",
    )
    parser.add_argument(
        "--cutoff",
        default=0.6,
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
    args = parser.parse_args()

    index, _ = load_data(args.embedding_model_path, args.path_to_db)
    query_engine = create_query_engine(cutoff=args.cutoff, top_k=args.top_k, filters=None)
    # Wrap in a QueryBundle class in order to use reranker.
    query = QueryBundle(args.query)
    retrieved_nodes = query_engine.query(query).source_nodes

    reranker = LLMRerank(top_n=args.top_k)

    context_str = ""
    for node in retrieved_nodes:
        print(f"Context: {node.metadata}")
        context_str += node.text.replace("\n", "  \n")
    print(f"\nUsing {context_str=}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prefix = get_llama3_1_instruct_str(args.query, context_str, tokenizer)

    print(f"\n{prefix=}\n")

    stopping_ids = [
        tokenizer.eos_token_id,
    ]
    llm = get_llm(
        args.model_name,
        stopping_ids,
        args.temp,
        args.max_new_tokens,
        args.top_p,
        args.use_4bit_quant,
    )
    output_response = llm.complete(prefix)
    print(f"\n{output_response.text=}\n")

    print("\n **** REFERENCES **** \n")
    for i in range(len(retrieved_nodes)):
        title = retrieved_nodes[i].node.metadata["Source"]
        page = retrieved_nodes[i].node.metadata["Page Number"]
        text = retrieved_nodes[i].node.text
        commit = retrieved_nodes[i].node.metadata["Commit"]
        doctag = retrieved_nodes[i].node.metadata["Tag"]
        newtext = text.encode("unicode_escape").decode("unicode_escape")
        out_title = f"**Source:** {title}  \n **Page:** {page}  \n **Similarity Score:** {round((retrieved_nodes[i].score * 100),3)}% \n"
        out_text = f"**Text:**  \n {newtext}  \n"
        title = title.replace(" ", "%20")

        pprint(f"{out_title=}")
