import io
import textwrap
from pathlib import Path
from pprint import pprint
from textwrap import dedent
from typing import Union

import torch
from llama_index.core.schema import NodeWithScore
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import PreTrainedTokenizer

from rag._defaults import (
    DEFAULT_SCORE_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
)


def get_llama3_1_instruct_str(
    query: str,
    nodes: list[NodeWithScore],
    tokenizer: PreTrainedTokenizer,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    context_str = ""
    for node in nodes:
        context_str += node.text
        # context_str += node.text.replace("\n", "  \n")

    context_and_query = f"""
 Please use the context provided below to answer the associated questions.
---------------------
{context_str}
---------------------

Query: {query}
Answer:
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dedent(context_and_query).strip("\n")},
    ]
    toks = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    print(f"Prefix: {len(toks)=}")
    return tokenizer.decode(toks)


def get_llama3_1_score_str(
    excerpt: str,
    query: str,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str = DEFAULT_SCORE_PROMPT,
) -> str:
    context_and_query = f"""
---------------------
Excerpt: {excerpt}

Query: {query}
---------------------

How relevant is the except to the query on a scale of 1 to 10?
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dedent(context_and_query).strip("\n")},
    ]
    toks = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return tokenizer.decode(toks)


def get_tag_from_dir(input_dir: Union[str, Path]) -> str:
    input_path = Path(input_dir)
    if input_path.is_file():
        raise ValueError("input expected to be a directory, not a file")
    tag = input_path.parts[-1]
    return tag


def get_local_llm(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int,
    use_4bit_quant: bool,
    generate_kwargs: dict,
) -> HuggingFaceLLM:
    print(f"Using HF model: {model_name}")

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
        stopping_ids=[tokenizer.eos_token_id],
        model_kwargs=model_kwargs,
    )
    pprint(f"Loaded model {model_name}")
    return llm


def get_llm_answer(
    llm,
    prefix: str,
    streaming: bool = False,
) -> str:
    output_response = (
        llm.stream_complete(prefix, formatted=True) if streaming else llm.complete(prefix)
    )
    return output_response


def print_in_box(
    text: str,
    header: str = "",
    horizontal_edge: str = "-",
    vertical_edge: str = "|",
    width: int = 80,
) -> str:
    assert len(horizontal_edge) == 1
    assert len(vertical_edge) == 1
    # Split the text into lines
    lines = []
    for line in text.split("\n"):
        lines.extend(textwrap.wrap(line, width=width, replace_whitespace=False))

    # Find the length of the longest line
    max_length = max(len(line) for line in lines)

    # Create the top and bottom borders
    inner_border_len = max_length + 2
    top_header_segment = horizontal_edge * ((inner_border_len - len(header) + 1) // 2)
    top_border = "+" + top_header_segment + header + top_header_segment + "+"
    bottom_border = "+" + horizontal_edge * inner_border_len + "+"

    # Print the box with surrounding whitespace. StringIO for efficiency.
    print_string = io.StringIO()
    print_string.write("\n")
    print_string.write(top_border)
    print_string.write("\n")
    for line in lines:
        print_string.write(f"{vertical_edge} {line:<{max_length}} {vertical_edge}")
        print_string.write("\n")
    print_string.write(bottom_border)
    print_string.write("\n\n")
    print(print_string.getvalue(), flush=True)


def print_references(nodes):
    print("\n **** REFERENCES **** \n")
    for idx, n in enumerate(nodes):
        title = n.node.metadata["Source"]
        page = n.node.metadata["PageNumber"]
        text = n.node.text
        newtext = text.encode("unicode_escape").decode("unicode_escape")
        out_title = f"**Source:** {title}  \n **Page:** {page}  \n **Similarity Score:** {round((n.score * 100),3)}% \n"
        out_text = f"**Text:**  \n {newtext}  \n"
        print_in_box(f"{out_title=}\n\n{out_text=}", f"Reference: {idx}")
