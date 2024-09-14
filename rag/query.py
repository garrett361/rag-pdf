import argparse
from pprint import pprint
from textwrap import dedent

import chromadb
import torch
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import AutoTokenizer

from rag._defaults import DEFAULT_HF_CHAT_MODEL, DEFAULT_HF_EMBED_MODEL

DEFAULT_INSTRUCTIONS = "If you don't know the answer to a question, please don't share false information. \n Limit your response to {} tokens."
DEFAULT_SYTEM_PROMPT = """
You are a helpful assistant. Your role is to answer questions about one
or more document. Several passages from the documents will be provided,
followed by a question.  Please answer the question based only on the
information provided in the context. If the question cannot be answered based
on the context, say that you cannot answer. Do not make up an answer.
                       """
DEFAULT_SYTEM_PROMPT = dedent(DEFAULT_SYTEM_PROMPT).strip("\n")


def get_llama3_1_instruct_str(
    context_str: str, prompt: str, max_new_tokens: int, system_prompt: str = DEFAULT_SYTEM_PROMPT
) -> str:
    """
    From: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1

    dedent and strip for proper formatting: https://www.youtube.com/watch?v=ph1pfBB6kOI

    Finding we still need to tab over the text to avoid unwanted whitespace.
    """

    out = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt} Be concise. Answer in fewer than {max_new_tokens} words.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

The relevant context from the documents is below:

{context_str}

{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
    """
    return dedent(out).strip("\n")


def get_prompt(max_new_tokens: int) -> str:
    return DEFAULT_INSTRUCTIONS.format(max_new_tokens)


def get_llm(
    model_name: str, temp: float, max_new_tokens: int, top_p: float, use_4bit_quant: bool
) -> HuggingFaceLLM:
    pprint(f"Using HF model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    stopping_ids = [
        tokenizer.eos_token_id,
    ]
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


def output_stream(llm_stream):
    for chunk in llm_stream:
        yield chunk.delta


if __name__ == "__main__":
    print("\n**********  QUERYING **********\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="prompt to ask of the llm")
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
    output = query_engine.query(args.prompt)
    context_str = ""
    for node in output.source_nodes:
        print(f"Context: {node.metadata}")
        new_text = node.text.replace("\n", "  \n")
        print(f"\n{new_text=}\n")
        context_str += new_text
    print(f"Using {context_str=}")

    prefix = get_llama3_1_instruct_str(context_str, args.prompt, args.max_new_tokens)

    print(f"\n{prefix=}\n")
    llm = get_llm(args.model_name, args.temp, args.max_new_tokens, args.top_p, args.use_4bit_quant)
    output_response = llm.complete(prefix)
    pprint(f"{output_response.text=}")

    print("\n **** REFERENCES **** \n")
    references = output.source_nodes
    for i in range(len(references)):
        title = references[i].node.metadata["Source"]
        page = references[i].node.metadata["Page Number"]
        text = references[i].node.text
        commit = references[i].node.metadata["Commit"]
        doctag = references[i].node.metadata["Tag"]
        newtext = text.encode("unicode_escape").decode("unicode_escape")
        out_title = f"**Source:** {title}  \n **Page:** {page}  \n **Similarity Score:** {round((references[i].score * 100),3)}% \n"
        out_text = f"**Text:**  \n {newtext}  \n"
        title = title.replace(" ", "%20")

        pprint(f"{out_title=}")
