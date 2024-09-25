import argparse
import json
import os

import chromadb
import weaviate
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from rag._defaults import DEFAULT_HF_EMBED_MODEL, DEFAULT_MAX_EMBED_BSZ


def embed(data_path: str, path_to_db: str, embed_model, weaviate_client) -> None:
    weaviate_client.collections.delete("Documents")
    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name="Documents")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = []
    for dirpath, _, files in os.walk(data_path):
        for file in files:
            input_file = os.path.join(dirpath, file)

            with open(input_file, "r") as f:
                input_text = json.load(f)
                for doc in input_text:
                    if isinstance(doc, dict):
                        if doc["data_type"] == "Table":
                            text = doc["metadata"]["text_as_html"]
                        else:
                            text = doc["content"]
                        source = doc["metadata"]["source"]
                        page_number = doc["metadata"].get("page_number", 1)
                        tag = doc["metadata"].get("tag", "")
                        question_answered = doc["metadata"].get("question_answered", "")
                        metadata = {
                            "Source": source,
                            "PageNumber": page_number,
                            "Commit": os.environ.get("PACH_JOB_ID", ""),
                            "Tag": tag,
                            "QuestionAnswered": question_answered,
                        }
                        docs.append(
                            TextNode(
                                text=text,
                                metadata=metadata,
                                excluded_embed_metadata_keys=[
                                    "Source",
                                    "PageNumber",
                                    "Commit",
                                    "Tag",
                                ],
                                excluded_llm_metadata_keys=[
                                    "Source",
                                    "PageNumber",
                                    "Commit",
                                    "Tag",
                                    "QuestionAnswered",
                                ],
                                metadata_template="{value}",
                            )
                        )

    print("Number of chunks: ", len(docs))

    # Insert nodes into both indices
    index = VectorStoreIndex(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        insert_batch_size=DEFAULT_MAX_EMBED_BSZ,
    )
    print("Indexing done!")
    index.storage_context.persist(persist_dir=path_to_db)
    print(f"Persisting done! Saved at {path_to_db}")
    return weaviate_client


if __name__ == "__main__":
    print("\n**********  EMBEDDING **********\n")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-db",
        type=str,
        default="db/",
        help="path to chromadb",
    )
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default=DEFAULT_HF_EMBED_MODEL,
        help="Embedding model path",
    )

    parser.add_argument("--data-path", type=str, help="Path to json files with unstructured chunks")
    parser.add_argument("--output", help="output directory")
    args = parser.parse_args()

    settings = chromadb.get_settings()
    settings.allow_reset = True
    print(f"creating/loading db at {args.path_to_db}...")
    db = chromadb.PersistentClient(path=args.path_to_db, settings=settings)
    print("Done!")

    if args.embedding_model_path.startswith("http"):
        print(f"\nUsing Embedding API model endpoint: {args.embedding_model_path}\n")
        embed_model = OpenAIEmbedding(api_base=args.embedding_model_path, api_key="dummy")
    else:
        print(f"\nUsing local Embedding model: {args.embedding_model_path}\n")
        embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_path)

    with weaviate.WeaviateClient(
        embedded_options=weaviate.embedded.EmbeddedOptions(persistence_data_path=args.path_to_db)
    ) as weaviate_client:
        embed(args.data_path, args.path_to_db, embed_model, weaviate_client)
