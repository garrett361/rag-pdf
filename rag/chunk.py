import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from loguru import logger
from semantic_router.encoders import HuggingFaceEncoder
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import TYPE_TO_TEXT_ELEMENT_MAP, Element, ElementMetadata
from unstructured.staging.base import convert_to_dict, dict_to_elements

from rag._defaults import DEFAULT_HF_EMBED_MODEL
from rag.rolling_window import UnstructuredSemanticSplitter


def elements_from_rag_dicts(element_dicts: Iterable[dict[str, Any]]) -> list[Element]:
    """Convert a list of rag-schema-dicts to a list of elements."""
    elements: list[Element] = []

    for item in element_dicts:
        element_id: str = item.get("id", None)
        metadata = (
            ElementMetadata()
            if item.get("metadata") is None
            else ElementMetadata.from_dict(item["metadata"])
        )
        if item.get("data_type") in TYPE_TO_TEXT_ELEMENT_MAP:
            ElementCls = TYPE_TO_TEXT_ELEMENT_MAP[item["data_type"]]
            elements.append(
                ElementCls(text=item["content"], element_id=element_id, metadata=metadata)
            )
        else:
            raise ValueError(f"Data type {item.get('data_type')} not in {TYPE_TO_TEXT_ELEMENT_MAP}")

    return elements


def chunk_docs_unstruct(
    combine_text_under_n_chars: int, max_characters: int, new_after_n_chars: int, elements
):
    chunking_settings = {
        "combine_text_under_n_chars": combine_text_under_n_chars,
        "max_characters": max_characters,
        "new_after_n_chars": new_after_n_chars,
    }
    chunked_raw = chunk_by_title(elements=elements, **chunking_settings)
    el = elements[0].to_dict()
    results = convert_to_dict(chunked_raw)
    if "tag" in el["metadata"]:
        for result in results:
            result["metadata"]["tag"] = el["metadata"]
    return results


def chunk_with_rolling_window(
    embedding_model_path: str,
    rolling_min_split: int,
    rolling_max_split: int,
    elements: list[Element],
):
    encoder = HuggingFaceEncoder(name=embedding_model_path)
    splitter = UnstructuredSemanticSplitter(
        encoder=encoder,
        window_size=1,  # Compares each element with the previous one
        min_split_tokens=rolling_min_split,
        max_split_tokens=rolling_max_split,
    )
    elements_dict = convert_to_dict(elements)
    results = splitter(elements_dict)
    return results


def main(
    input: str,
    output: str,
    chunker: str,
    embedding_model_path: str,
    rolling_max_split: int,
    rolling_min_split: int,
    combine_text_under_n_chars: int,
    max_characters: int,
    new_after_n_chars: int,
):
    for dirpath, _, files in os.walk(input):
        for file in files:
            input_file = os.path.join(dirpath, file)
            logger.info(f"Processing {input_file}.....")
            with open(input_file) as file:
                contents = json.load(file)
                if contents[0].get("id") is None:
                    elements_raw = dict_to_elements(contents)
                else:
                    elements_raw = elements_from_rag_dicts(contents)
            if chunker == "rolling_window":
                logger.info(f"Processing {input_file} with rolling window")
                elements = chunk_with_rolling_window(
                    embedding_model_path, rolling_min_split, rolling_max_split, elements_raw
                )
            elif chunker == "unstructured":
                logger.info(f"Processing {input_file} with unstructured")
                elements = chunk_docs_unstruct(
                    combine_text_under_n_chars, max_characters, new_after_n_chars, elements_raw
                )
            else:
                raise ValueError(
                    f"Expected `chunker` to be one of (rolling_window, unstructured), not {chunker}"
                )
            logger.info(f"Finished processing {input_file} with {len(elements)} chunks")
            output_path = os.path.join(output, Path(input_file).stem + ".json")
            with open(output_path, "w") as f:
                logger.info(f"Writing output to {output_path}")
                json.dump(elements, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File Parser")
    parser.add_argument("--input", type=str, help="input directory")
    parser.add_argument("--output", default="./output", help="output directory")
    parser.add_argument("--chunker", default="unstructured", help="chunking engine")
    parser.add_argument("--combine_text_under_n_chars", default=50, help="unstructured setting")
    parser.add_argument("--max_characters", default=750, help="unstructured setting")
    parser.add_argument("--new_after_n_chars", default=500, help="unstructured setting")
    parser.add_argument(
        "--embedding_model_path",
        default=DEFAULT_HF_EMBED_MODEL,
        help="embedding model for rolling_window",
    )
    parser.add_argument(
        "--rolling_min_split", type=int, default=50, help="min split tokens rolling_window"
    )
    parser.add_argument(
        "--rolling_max_split", type=int, default=100, help="max split tokens rolling_window"
    )
    args = parser.parse_args()
    logger.info("Starting processing")
    main(**vars(args))
