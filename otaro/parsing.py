import logging
import re
from typing import Any

import dirtyjson
from dirtyjson.attributed_containers import AttributedDict, AttributedList

logging.basicConfig()
logger = logging.getLogger("otaro.parsing")
logger.setLevel(logging.INFO)


def find_json(rgx: str, text: str):
    match = re.search(rgx, text)
    if match is None:
        return text
    else:
        return match.groupdict().get("json")


def convert_attributed_container(
    container: Any | AttributedDict | AttributedList | float | int,
):
    if isinstance(container, AttributedList):
        return [convert_attributed_container(i) for i in container]
    elif isinstance(container, AttributedDict):
        dict_container = {**container}
        for k, v in dict_container.items():
            dict_container[k] = convert_attributed_container(v)
        return dict_container
    else:
        return container


def llm_parse_json(text: str):
    """Read LLM output and extract JSON data from it."""

    # First check for ```json
    code_snippet_pattern = r"```json(?P<json>(.|\s|\n)*?)```"
    code_snippet_result = find_json(code_snippet_pattern, text)
    # Then try to find the longer match between [.*?] and {.*?}
    array_pattern = re.compile("(?P<json>\\[.*\\])", re.DOTALL)
    array_result = find_json(array_pattern, text)
    dict_pattern = re.compile("(?P<json>{.*})", re.DOTALL)
    dict_result = find_json(dict_pattern, text)

    if array_result and dict_result and len(dict_result) > len(array_result):
        results = [
            code_snippet_result,
            dict_result,
            array_result,
        ]
    else:
        results = [
            code_snippet_result,
            array_result,
            dict_result,
        ]

    # Try each result in order
    for result in results:
        if result is not None:
            try:
                result_json = dirtyjson.loads(result)
                return convert_attributed_container(result_json)
            except dirtyjson.error.Error:
                pass
            try:
                result = (
                    result.replace("None", "null")
                    .replace("True", "true")
                    .replace("False", "false")
                )
                result_json = dirtyjson.loads(result)
                return convert_attributed_container(result_json)
            except dirtyjson.error.Error:
                continue

    error_message = (
        f"Failed to parse JSON from text {text!r}. Your model may not be capable of"
        " supporting JSON output or our parsing technique could use some work. Try"
        " a different model"
    )
    raise ValueError(error_message)
