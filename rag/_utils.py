import io
import textwrap
from pathlib import Path
from typing import Union


def get_tag_from_dir(input_dir: Union[str, Path]) -> str:
    input_path = Path(input_dir)
    if input_path.is_file():
        raise ValueError("input expected to be a directory, not a file")
    tag = input_path.parts[-1]
    return tag


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
    wrapped_lines = textwrap.wrap(text, width=width, replace_whitespace=False)
    lines = []
    for line in wrapped_lines:
        lines.extend(line.split("\n"))

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
