import argparse
from typing import Union

import chromadb
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openllm import OpenLLMAPI
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import AutoTokenizer

from rag._defaults import DEFAULT_HF_CHAT_MODEL, DEFAULT_HF_EMBED_MODEL

DEFAULT_INSTRUCTIONS = "If you don't know the answer to a question, please don't share false information. \n Limit your response to 500 tokens."


def get_llm(
    model_name: str,
    temp: float,
    max_length: int,
    top_p: float,
) -> Union[HuggingFaceLLM, OpenLLMAPI]:
    generate_kwargs = {
        "do_sample": True,
        "temperature": temp,
        "top_p": top_p,
        "max_length": max_length,
    }
    if model_name.startswith("http"):
        print(f"Using OpenLLM model endpoint: {model_name}")
        llm = OpenLLMAPI(address=model_name, generate_kwargs=generate_kwargs)
    else:
        print(f"Using HF model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        stopping_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        llm = HuggingFaceLLM(
            model_name=model_name,
            tokenizer_name=model_name,
            generate_kwargs=generate_kwargs,
            stopping_ids=stopping_ids,
        )
        print(f"Loaded model {model_name}")
    return llm


def load_data(
    embedding_model_path: str, path_to_db: str
) -> tuple[VectorStoreIndex, chromadb.GetResult]:
    if embedding_model_path.startswith("http"):
        print(f"Using Embedding API model endpoint: {embedding_model_path}")
        embed_model = OpenAIEmbedding(api_base=embedding_model_path, api_key="dummy")
    else:
        print(f"Embedding model: {embedding_model_path}")
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
    # configure response synthesizer
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
        "--max-length",
        default=250,
        type=int,
        help="Max generation toks",
    )
    parser.add_argument(
        "--cutoff",
        default=0.6,
        type=float,
        help="cutoff for similarity score",
    )
    parser.add_argument(
        "--streaming",
        help="stream responses",
        action="store_true",
    )
    args = parser.parse_args()

    llm = get_llm(args.model_name, args.temp, args.max_length, args.top_p)

    index, chunks = load_data()
    query_engine = create_query_engine(cutoff=args.cutoff, top_k=args.top_k, filters=None)
    output = query_engine.query(args.prompt)
    context_str = ""
    for node in output.source_nodes:
        print(f"Context: {node.metadata}")
        context_str += node.text.replace("\n", "  \n")
    text_qa_template_str_llama3 = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Context information is
        below.
        ---------------------
        {context_str}
        ---------------------
        Using
        the context information, answer the question: {args.prompt}
        {DEFAULT_INSTRUCTIONS}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    # if args.streaming:
    #     output_response = llm.stream_complete(text_qa_template_str_llama3, formatted=True)
    #     with chat_container.chat_message("assistant", avatar="./static/logo.jpeg"):
    #         response = st.write_stream(output_stream(output_response))

    output_response = llm.complete(text_qa_template_str_llama3)
    print(f"{output_response=}")

    print("REFERENCES")
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

        print(f"{out_title=}")
