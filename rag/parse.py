import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import nltk
import torch
from llama_index.llms.openllm import OpenLLM
from loguru import logger
from transformers import AutoTokenizer
from unstructured.partition.auto import partition

from rag._defaults import (
    DEFAULT_CHUNK_STRAT,
    DEFAULT_COMBINE_TEXT_UNDER_N_CHARS,
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_NEW_AFTER_N_CHARS,
    DEFAULT_SYSTEM_PROMPT,
    INFORMATIVE_PROMPT,
    QA_PROMPT,
)
from rag._utils import get_tag_from_dir, print_in_box
from rag.query import get_local_llm
from rag.rag_schema import DataElement, DataType, Document, Metadata


def elements_to_rag_schema(elements: list, tag=None) -> list[DataElement]:
    output_list = Document()
    for element in elements:
        if isinstance(element, dict):
            el = deepcopy(element)
        else:
            el = element.to_dict()
        if "url" in el["metadata"]:
            source = el["metadata"]["url"]
        elif "filename" in el["metadata"]:
            source = el["metadata"]["filename"]
        else:
            source = "Unknown"
        el["metadata"]["source"] = source
        page_number = el["metadata"].get("page_number", 1)
        url = el["metadata"].get("url", None)
        text_as_html = el["metadata"].get("text_as_html", None)
        output_list.append(
            DataElement(
                id=el["element_id"],
                data_type=DataType(el["type"]),
                content=el["text"],
                metadata=Metadata(
                    source=el["metadata"]["source"],
                    page_number=page_number,
                    url=url,
                    text_as_html=text_as_html,
                    tag=tag,
                ),
            )
        )
    return output_list


def parse(
    input_file,
    output,
    strategy,
    chunking_strategy: str,
    combine_text_under_n_chars: Optional[int] = None,
    max_characters: Optional[int] = None,
    new_after_n_chars: Optional[int] = None,
    tag=None,
) -> None:
    logger.info(f"Processing {input_file}. Using {tag=}")
    elements = partition(
        filename=input_file,
        skip_infer_table_types=[],
        strategy=strategy,
        chunking_strategy=chunking_strategy,
        combine_text_under_n_chars=combine_text_under_n_chars,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
    )
    logger.info("Done parsing elements")
    output_list = elements_to_rag_schema(elements, tag=tag)
    input_file_path = Path(input_file)

    output_path = Path(os.path.join(output, input_file_path.stem + ".json"))
    if not output_path.parent.exists():
        print(f"Creating {input_file_path.parent.absolute()=}")
        os.makedirs(output_path.parent.absolute())
    with open(output_path, "w") as f:
        logger.info(f"Writing output to {output_path}")
        json.dump(output_list, f, indent=4)


# From Liam
def generate_completion(llm, tokenizer, text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    toks = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = tokenizer.decode(toks)
    return llm.complete(text)


# From Liam
def clean_parsed(json_file, llm, tokenizer):
    """
    Filter the results of the parsed function based on whether an LLM thinks the chunk is
    informative or not, and if so, include an LLM generated question that the chunk contains the
    answer to.
    """
    results = []
    with open(json_file, "r") as f:
        input_text = json.load(f)
        for doc in input_text:
            if isinstance(doc, dict):
                if doc["data_type"] == "Table":
                    text = doc["metadata"]["text_as_html"]
                else:
                    text = doc["content"]
            informative = generate_completion(
                llm, tokenizer, INFORMATIVE_PROMPT.format(context=text)
            ).text
            if informative.lower() == "no":
                print_in_box(text, f" UNINFORMATIVE ({informative=}) ", "?", "?")
            else:
                prefix = QA_PROMPT.format(context=text)
                question_answered = generate_completion(llm, tokenizer, prefix).text
                print_in_box(
                    text + "\n\nQuestion Answered: " + question_answered,
                    f" Generating Question({informative=}) ",
                )
                doc["metadata"]["question_answered"] = question_answered
                results.append(doc)
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)


def main(
    input: str,
    output: str,
    strategy: str,
    chunking_strategy: str,
    combine_text_under_n_chars: int,
    max_characters: int,
    new_after_n_chars: int,
    folder_tags: bool = False,
    clean_parse_with_llm: bool = False,
    model_name: Optional[str] = None,
    chat_model_endpoint: Optional[str] = None,
) -> None:
    if chat_model_endpoint:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO: @garrett.goon -  Don't hard code
        generate_kwargs = {"do_sample": False}
        max_new_tokens = 200

        if chat_model_endpoint:
            print(f"\nUsing hosted LLM at: {chat_model_endpoint}\n")
            llm = OpenLLM(
                model=model_name,
                api_base=chat_model_endpoint,
                api_key="fake",
                generate_kwargs=generate_kwargs,
                max_tokens=max_new_tokens,
            )
        elif model_name:
            assert (
                clean_parse_with_llm
            ), "--clean-parse-with-llm must be True if chat_model_endpoint"
            print(f"\nUsing local {model_name} LLM\n")
            llm = get_local_llm(
                model_name,
                tokenizer,
                max_new_tokens,
                use_4bit_quant=False,
                generate_kwargs=generate_kwargs,
            )
    else:
        llm = None
        assert not model_name
        assert not chat_model_endpoint
        print("\nNo LLM used to parse.\n")

    # The expectation is that input is a directory which contains various subdirs and no files.
    # Each subdir should itself only contain files, and not additional subdirs, and the contents of
    # each subdir will be parsed and tagged together.
    input_path = Path(input)
    if input_path.is_file():
        raise ValueError("Input must be a directory, not a file.")
    for subdir in input_path.iterdir():
        if subdir.is_file():
            raise ValueError(
                f"Input dir is expected to contain only subdirectories and no files. Found file {subdir=}"
            )
        tag = get_tag_from_dir(subdir) if folder_tags else None
        for file in subdir.iterdir():
            if file.is_dir():
                raise ValueError(
                    f"Subdirectories may only contain files and no additional subdirs. Found subdir {file=}"
                )
            parse(
                file,
                output,
                strategy,
                chunking_strategy,
                combine_text_under_n_chars,
                max_characters,
                new_after_n_chars,
                tag,
            )
    if clean_parse_with_llm:
        for file in Path(output).rglob("*"):
            clean_parsed(file, llm, tokenizer)


if __name__ == "__main__":
    print("\n**********  PARSING **********\n")
    parser = argparse.ArgumentParser(description="File Parser")
    parser.add_argument(
        "--input", type=str, help="Input directory containing to-be-parsed subdirectories."
    )
    parser.add_argument("--output", default="./output", help="output directory")
    parser.add_argument("--strategy", default="auto", help="parsing strategy")
    parser.add_argument(
        "--chunking_strategy", default=DEFAULT_CHUNK_STRAT, help="chunking strategy"
    )
    parser.add_argument(
        "--combine_text_under_n_chars",
        default=DEFAULT_COMBINE_TEXT_UNDER_N_CHARS,
        type=int,
        help="unstructured setting",
    )
    parser.add_argument(
        "--max_characters", default=DEFAULT_MAX_CHARACTERS, type=int, help="unstructured setting"
    )
    parser.add_argument(
        "--new_after_n_chars",
        default=DEFAULT_NEW_AFTER_N_CHARS,
        type=int,
        help="unstructured setting",
    )
    parser.add_argument(
        "--folder_tags",
        action="store_true",
        help="folder tags",
    )

    # For using an LLM to clean up chunks
    parser.add_argument(
        "--clean-parse-with-llm",
        action="store_true",
        help="name of chat model used to clean chunks",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="name of chat model used to clean chunks",
    )
    parser.add_argument(
        "--chat-model-endpoint",
        default=None,
        type=str,
        help="HTTP path to model endpoint, if serving",
    )

    args = parser.parse_args()

    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    logger.info("Starting processing")
    if args.folder_tags:
        logger.info("Using folder names as tags")

    if args.clean_parse_with_llm and not args.model_name:
        raise ValueError("A --model-name argument must be suped with --clean-parse-with-llm.")

    nltk.download("punkt_tab")

    main(**vars(args))
